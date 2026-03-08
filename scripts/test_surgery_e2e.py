"""End-to-end test: IR-surgery stateful decoder with real encoder output.

Validates that the decoder_stateful_embeds model (created by IR surgery in
test_inputs_embeds_workarounds.py) produces correct transcription when fed
real audio encoder features via inputs_embeds.

Pipeline:
  1. Audio (silence 5s) -> mel spectrogram
  2. mel -> encoder_fp16.xml (CPU) -> audio features [1, 104, 1024]
  3. Build inputs_embeds: embed_tokens[text_token_ids] + audio features
  4. Prefill: decoder_stateful_embeds inputs_embeds -> logits
  5. Greedy decode loop: single-token inputs_embeds -> next token

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/test_surgery_e2e.py'
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/test_surgery_e2e.py path\\to\\audio.wav'
"""

import sys
import os
import time

sys.path.insert(0, r"C:\Apps\Qwen3-ASR")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import openvino as ov

MODEL_DIR = r"C:\Apps\Qwen3-ASR\models"
HF_MODEL_DIR = r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B"

# Constants
SAMPLE_RATE = 16000
T_FIXED = 800
AUDIO_PAD_COUNT = 104

# Special token IDs
IM_START = 151644
IM_END = 151645
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676
ASR_TEXT = 151704
NEWLINE = 198


def load_audio(audio_path):
    if audio_path is not None:
        import librosa
        print(f"Loading audio: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        print(f"  Duration: {len(audio) / SAMPLE_RATE:.2f}s, samples: {len(audio)}")
        return audio.astype(np.float32)
    else:
        print("No audio file provided -- using 5s of silence for smoke test")
        return np.zeros(int(SAMPLE_RATE * 5.0), dtype=np.float32)


def compute_mel(audio):
    from transformers import WhisperFeatureExtractor
    print("Computing mel via WhisperFeatureExtractor...")
    feat_extractor = WhisperFeatureExtractor.from_pretrained(HF_MODEL_DIR)
    features = feat_extractor(audio, sampling_rate=SAMPLE_RATE, padding=True,
                              return_attention_mask=True, return_tensors="np")
    mel = features["input_features"].astype(np.float32)  # [1, 128, T]
    print(f"  Mel shape: {mel.shape}")
    # Pad/trim to T_FIXED
    if mel.shape[2] < T_FIXED:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, T_FIXED - mel.shape[2])))
    elif mel.shape[2] > T_FIXED:
        mel = mel[:, :, :T_FIXED]
    return mel


def build_prompt_tokens(tokenizer):
    system_text = "system\nYou are a helpful assistant."
    system_tokens = [IM_START] + tokenizer.encode(system_text) + [IM_END, NEWLINE]
    user_text = "user\n"
    user_tokens = (
        [IM_START] + tokenizer.encode(user_text)
        + [AUDIO_START] + [AUDIO_PAD] * AUDIO_PAD_COUNT + [AUDIO_END, IM_END, NEWLINE]
    )
    assistant_text = "assistant\n"
    assistant_tokens = [IM_START] + tokenizer.encode(assistant_text)
    return system_tokens + user_tokens + assistant_tokens


def build_inputs_embeds(token_ids, audio_features, embed_table):
    """Build inputs_embeds by looking up text token embeddings and injecting audio features."""
    ids = np.array(token_ids, dtype=np.int64)
    embeds = embed_table[ids]  # [seq_len, 1024]
    # Replace audio_pad positions with encoder features
    audio_pad_positions = np.where(ids == AUDIO_PAD)[0]
    audio_feats = np.array(audio_features[0])  # [104, 1024]
    embeds[audio_pad_positions] = audio_feats
    return embeds  # [seq_len, 1024]


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("  E2E Test: IR-Surgery Stateful Decoder + Encoder")
    print("=" * 70)
    print()

    core = ov.Core()
    print(f"OpenVINO {ov.__version__}")
    print(f"Available devices: {core.available_devices}")
    print()

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    print("Loading encoder (CPU)...")
    encoder = core.compile_model(
        os.path.join(MODEL_DIR, "encoder_fp16.xml"), "CPU"
    )
    encoder_input_name = encoder.inputs[0].any_name

    surgery_model_path = os.path.join(MODEL_DIR, "decoder_stateful_embeds", "openvino_model.xml")
    if not os.path.exists(surgery_model_path):
        print(f"ERROR: Surgery model not found: {surgery_model_path}")
        print("Run test_inputs_embeds_workarounds.py first to create it.")
        sys.exit(1)

    print("Loading IR-surgery decoder (CPU, stateful + inputs_embeds)...")
    decoder_compiled = core.compile_model(
        surgery_model_path, "CPU", {"PERFORMANCE_HINT": "LATENCY"}
    )

    # Print decoder inputs
    input_names = {}
    for inp in decoder_compiled.inputs:
        try:
            name = inp.any_name
        except RuntimeError:
            continue
        input_names[name] = inp
        print(f"  Decoder input: {name} shape={inp.partial_shape} type={inp.element_type}")

    print("\nLoading embedding table...")
    embed_table = np.load(os.path.join(MODEL_DIR, "embed_tokens.npy"))
    print(f"  Shape: {embed_table.shape}, dtype: {embed_table.dtype}")

    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR, trust_remote_code=True)

    # ------------------------------------------------------------------
    # 2. Audio -> Mel -> Encoder
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("Step 1: Audio -> Mel -> Encoder")
    print("-" * 70)
    audio = load_audio(audio_path)
    mel = compute_mel(audio)

    t0 = time.perf_counter()
    enc_result = encoder({encoder_input_name: mel})
    enc_time = time.perf_counter() - t0
    audio_features = list(enc_result.values())[0]
    print(f"  Encoder output: {audio_features.shape}, time={enc_time*1000:.0f}ms")

    # ------------------------------------------------------------------
    # 3. Build prompt
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("Step 2: Build prompt + inputs_embeds")
    print("-" * 70)
    all_tokens = build_prompt_tokens(tokenizer)
    prompt_len = len(all_tokens)
    print(f"  Prompt length: {prompt_len} tokens")

    inputs_embeds = build_inputs_embeds(all_tokens, audio_features, embed_table)
    print(f"  inputs_embeds shape: {inputs_embeds.shape}")

    # ------------------------------------------------------------------
    # 4. Prefill
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("Step 3: Prefill (stateful model with inputs_embeds)")
    print("-" * 70)

    request = decoder_compiled.create_infer_request()
    request.reset_state()

    # Build prefill inputs
    prefill_embeds = inputs_embeds[np.newaxis, :, :]  # [1, prompt_len, 1024]
    prefill_attn = np.ones((1, prompt_len), dtype=np.int64)
    prefill_pos = np.arange(prompt_len, dtype=np.int64).reshape(1, -1)
    prefill_beam = np.array([0], dtype=np.int32)
    prefill_ids = np.zeros((1, prompt_len), dtype=np.int64)  # dummy, not used

    prefill_inputs = {
        "inputs_embeds": prefill_embeds,
        "attention_mask": prefill_attn,
        "position_ids": prefill_pos,
        "beam_idx": prefill_beam,
        "input_ids": prefill_ids,
    }

    t0 = time.perf_counter()
    request.infer(prefill_inputs)
    prefill_time = time.perf_counter() - t0

    logits = request.get_output_tensor(0).data.copy()
    print(f"  Prefill time: {prefill_time*1000:.0f}ms")
    print(f"  Logits shape: {logits.shape}")

    # Get first token
    next_logits = logits[0, prompt_len - 1, :]
    first_token_id = int(np.argmax(next_logits))
    first_token_text = tokenizer.decode([first_token_id])
    print(f"  First predicted token: id={first_token_id}, text='{first_token_text}'")

    # Top-5
    top5_ids = np.argsort(next_logits)[-5:][::-1]
    print(f"  Top-5: {[(int(tid), tokenizer.decode([int(tid)])) for tid in top5_ids]}")

    # ------------------------------------------------------------------
    # 5. Greedy decode loop
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("Step 4: Greedy decode (stateful, KV-cache, single-token steps)")
    print("-" * 70)

    MAX_NEW_TOKENS = 50
    generated_ids = []
    past_len = prompt_len
    gen_start = time.perf_counter()

    current_token_id = first_token_id

    for step in range(MAX_NEW_TOKENS):
        if current_token_id == IM_END:
            print(f"  [Step {step:2d}] <|im_end|> -- generation complete")
            break

        generated_ids.append(current_token_id)
        token_text = tokenizer.decode([current_token_id])
        print(f"  [Step {step:2d}] id={current_token_id:6d}  '{token_text}'")

        # Look up embedding for this token
        token_embed = embed_table[current_token_id]  # [1024]
        single_embeds = token_embed[np.newaxis, np.newaxis, :]  # [1, 1, 1024]

        # Build decode inputs
        decode_attn = np.ones((1, past_len + 1), dtype=np.int64)
        decode_pos = np.array([[past_len]], dtype=np.int64)
        decode_beam = np.array([0], dtype=np.int32)
        decode_ids = np.array([[0]], dtype=np.int64)  # dummy

        decode_inputs = {
            "inputs_embeds": single_embeds.astype(np.float32),
            "attention_mask": decode_attn,
            "position_ids": decode_pos,
            "beam_idx": decode_beam,
            "input_ids": decode_ids,
        }

        request.infer(decode_inputs)
        step_logits = request.get_output_tensor(0).data.copy()
        current_token_id = int(np.argmax(step_logits[0, -1, :]))
        past_len += 1

    # Handle last token if not im_end
    if current_token_id != IM_END and len(generated_ids) < MAX_NEW_TOKENS:
        generated_ids.append(current_token_id)

    gen_time = time.perf_counter() - gen_start

    # ------------------------------------------------------------------
    # 6. Results
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  Results")
    print("=" * 70)

    transcription = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print(f"  Encoder time:    {enc_time*1000:.0f}ms")
    print(f"  Prefill time:    {prefill_time*1000:.0f}ms")
    print(f"  Generation time: {gen_time*1000:.0f}ms ({len(generated_ids)} tokens)")
    if len(generated_ids) > 0:
        print(f"  Avg per token:   {gen_time / len(generated_ids) * 1000:.1f}ms")
    total_time = enc_time + prefill_time + gen_time
    print(f"  Total time:      {total_time*1000:.0f}ms")
    print()
    print(f"  Generated tokens: {generated_ids}")
    print(f"  Transcription:    '{transcription}'")

    # ------------------------------------------------------------------
    # 7. Compare with test_e2e.py (decoder_fp16.xml no-KV-cache)
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  Comparison with decoder_fp16.xml (no KV-cache) approach:")
    print("-" * 70)
    print(f"  Surgery stateful: prefill={prefill_time*1000:.0f}ms + "
          f"decode={gen_time*1000:.0f}ms = {total_time*1000:.0f}ms total")
    print(f"  (Previous benchmark: decoder_fp16.xml CPU = ~1506ms/step, "
          f"~48s for 32 tokens)")
    if len(generated_ids) > 0:
        speedup = 48190.0 / (total_time * 1000) if total_time > 0 else 0
        print(f"  Estimated speedup: ~{speedup:.0f}x for 32 tokens")
    print()

    if audio_path is None:
        print("NOTE: This was a smoke test with silence. Provide audio file:")
        print("  uv run python scripts/test_surgery_e2e.py path/to/audio.wav")
    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
