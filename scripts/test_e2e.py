"""End-to-end OpenVINO inference test for Qwen3-ASR encoder + decoder.

Verifies that the exported OpenVINO IRs produce correct transcription by
running the full pipeline: audio -> mel -> encoder -> prompt build -> decoder
-> greedy decode.

Runs on CPU first (NPU later).

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/test_e2e.py'
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/test_e2e.py path\\to\\audio.wav'
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, r"C:\Apps\Qwen3-ASR")

import openvino as ov

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = r"C:\Apps\Qwen3-ASR\models"
HF_MODEL_DIR = r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B"

ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_fp16.xml")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_fp16.xml")
EMBED_PATH = os.path.join(MODEL_DIR, "embed_tokens.npy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
T_FIXED = 800          # Encoder mel input frames (fixed)
SEQ_LEN = 256          # Decoder sequence length (fixed)
AUDIO_PAD_COUNT = 104  # Encoder output seq_len = 8 chunks * 13 frames

# Special token IDs
IM_START = 151644
IM_END = 151645
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676
NEWLINE = 198  # \n


def load_audio(audio_path: str | None) -> np.ndarray:
    """Load audio from file or generate silence for testing."""
    if audio_path is not None:
        import librosa
        print(f"Loading audio: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        print(f"  Duration: {len(audio) / SAMPLE_RATE:.2f}s, samples: {len(audio)}")
        return audio.astype(np.float32)
    else:
        print("No audio file provided -- using 5s of silence for smoke test")
        return np.zeros(int(SAMPLE_RATE * 5.0), dtype=np.float32)


def compute_mel(audio: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram using the model's processor (WhisperFeatureExtractor).

    Falls back to librosa if the processor is unavailable.
    Returns shape [1, 128, T] where T >= T_FIXED (will be padded/trimmed later).
    """
    # Try WhisperFeatureExtractor directly -- it matches the model's expected
    # mel computation exactly (see preprocessor_config.json).
    try:
        from transformers import WhisperFeatureExtractor
        print("Computing mel via WhisperFeatureExtractor...")
        feat_extractor = WhisperFeatureExtractor.from_pretrained(HF_MODEL_DIR)
        features = feat_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="np",
        )
        mel = features["input_features"]  # [1, 128, T]
        print(f"  Mel shape from WhisperFeatureExtractor: {mel.shape}")
        return mel.astype(np.float32)
    except Exception as e:
        print(f"  WhisperFeatureExtractor failed ({e}), falling back to librosa...")

    # Fallback: manual librosa mel computation matching Whisper config
    import librosa
    print("Computing mel via librosa...")
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=400,       # 25ms window at 16kHz
        hop_length=160,  # 10ms hop at 16kHz
        n_mels=128,
        fmin=0,
        fmax=8000,
    )
    mel = np.log(np.clip(mel, a_min=1e-10, a_max=None))
    mel = mel[np.newaxis, :, :]  # [1, 128, T]
    print(f"  Mel shape from librosa: {mel.shape}")
    return mel.astype(np.float32)


def pad_or_trim_mel(mel: np.ndarray) -> np.ndarray:
    """Pad or trim mel to exactly [1, 128, T_FIXED]."""
    _, n_mels, t = mel.shape
    if t < T_FIXED:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, T_FIXED - t)))
    elif t > T_FIXED:
        mel = mel[:, :, :T_FIXED]
    assert mel.shape == (1, n_mels, T_FIXED), f"Expected [1, {n_mels}, {T_FIXED}], got {mel.shape}"
    return mel


def build_prompt_tokens(tokenizer) -> list[int]:
    """Build the ChatML prompt token sequence for ASR.

    Format:
        <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n
        <|im_start|>user\\n<|audio_start|><|audio_pad|>x104<|audio_end|><|im_end|>\\n
        <|im_start|>assistant\\n
    """
    system_text = "system\nYou are a helpful assistant."
    system_tokens = [IM_START] + tokenizer.encode(system_text) + [IM_END, NEWLINE]

    user_text = "user\n"
    user_tokens = (
        [IM_START]
        + tokenizer.encode(user_text)
        + [AUDIO_START]
        + [AUDIO_PAD] * AUDIO_PAD_COUNT
        + [AUDIO_END, IM_END, NEWLINE]
    )

    assistant_text = "assistant\n"
    assistant_tokens = [IM_START] + tokenizer.encode(assistant_text)

    all_tokens = system_tokens + user_tokens + assistant_tokens
    return all_tokens


def build_decoder_inputs(
    all_tokens: list[int],
    audio_features: np.ndarray,
    embed_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded decoder inputs from token list and encoder output.

    Returns:
        inputs_embeds:  [1, SEQ_LEN, 1024]
        position_ids:   [3, 1, SEQ_LEN]
        attention_mask:  [1, SEQ_LEN]
    """
    prompt_len = len(all_tokens)
    input_ids = np.array(all_tokens, dtype=np.int64)

    # Look up token embeddings
    inputs_embeds = embed_table[input_ids]  # [prompt_len, 1024]

    # Replace audio_pad positions with encoder features
    audio_pad_mask = (input_ids == AUDIO_PAD)
    audio_pad_positions = np.where(audio_pad_mask)[0]
    audio_feats = np.array(audio_features[0])  # [104, 1024]
    inputs_embeds[audio_pad_positions] = audio_feats

    # Pad to SEQ_LEN
    padded_embeds = np.zeros((1, SEQ_LEN, 1024), dtype=np.float32)
    padded_embeds[0, :prompt_len, :] = inputs_embeds

    # Position IDs: mRoPE uses cumsum(attention_mask) - 1 for all 3 channels.
    # For non-padded tokens this is simply 0, 1, 2, ...; padded positions get 1
    # (matching get_rope_index: masked_fill_(attention_mask == 0, 1)).
    pos_1d = np.zeros(SEQ_LEN, dtype=np.int64)
    pos_1d[:prompt_len] = np.arange(prompt_len, dtype=np.int64)
    # Padded positions: use value 1 (matches original model behavior)
    pos_1d[prompt_len:] = 1
    pos_ids = pos_1d.reshape(1, 1, SEQ_LEN)
    pos_ids = np.broadcast_to(pos_ids, (3, 1, SEQ_LEN)).copy()

    # Attention mask
    attn_mask = np.zeros((1, SEQ_LEN), dtype=np.int64)
    attn_mask[0, :prompt_len] = 1

    return padded_embeds, pos_ids, attn_mask


def get_tensor_name(tensor) -> str:
    """Get the name of an OpenVINO tensor, falling back to index if unnamed."""
    try:
        return tensor.any_name
    except RuntimeError:
        return f"<unnamed:{tensor.index}>"


def print_model_io(label: str, model):
    """Print input/output names and shapes of a compiled OpenVINO model."""
    print(f"{label} inputs:")
    for inp in model.inputs:
        print(f"  {get_tensor_name(inp):20s} {inp.partial_shape}")
    print(f"{label} outputs:")
    for out in model.outputs:
        print(f"  {get_tensor_name(out):20s} {out.partial_shape}")


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("Qwen3-ASR End-to-End OpenVINO Inference Test")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    core = ov.Core()
    print(f"OpenVINO {ov.__version__}")
    print(f"Available devices: {core.available_devices}")
    print()

    print("Loading encoder...")
    device = os.environ.get("OV_DEVICE", "CPU")
    cache_config = {"CACHE_DIR": os.path.join(MODEL_DIR, "cache")} if device == "NPU" else {}
    print(f"Using device: {device}")
    encoder = core.compile_model(ENCODER_PATH, device, cache_config)
    print_model_io("Encoder", encoder)
    print()

    print("Loading decoder...")
    decoder = core.compile_model(DECODER_PATH, device, cache_config)
    print_model_io("Decoder", decoder)
    print()

    print("Loading embedding table...")
    embed_table = np.load(EMBED_PATH)  # [151936, 1024]
    print(f"  Shape: {embed_table.shape}, dtype: {embed_table.dtype}")
    print()

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print()

    # ------------------------------------------------------------------
    # 2. Prepare audio -> mel
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Step 1: Audio -> Mel spectrogram")
    print("-" * 60)
    audio = load_audio(audio_path)
    mel = compute_mel(audio)
    mel = pad_or_trim_mel(mel)
    print(f"  Final mel input: {mel.shape} (dtype={mel.dtype})")
    print()

    # ------------------------------------------------------------------
    # 3. Run encoder
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Step 2: Encoder inference")
    print("-" * 60)

    # Discover the actual input name
    encoder_input_name = encoder.inputs[0].any_name
    print(f"  Using input name: '{encoder_input_name}'")

    t0 = time.perf_counter()
    encoder_result = encoder({encoder_input_name: mel})
    encoder_time = time.perf_counter() - t0

    audio_features = list(encoder_result.values())[0]  # [1, 104, 1024]
    print(f"  Output shape: {audio_features.shape}")
    print(f"  Output stats: min={audio_features.min():.4f}, max={audio_features.max():.4f}, "
          f"mean={audio_features.mean():.4f}")
    print(f"  Time: {encoder_time:.3f}s")
    print()

    # ------------------------------------------------------------------
    # 4. Build prompt
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Step 3: Build prompt tokens")
    print("-" * 60)

    all_tokens = build_prompt_tokens(tokenizer)
    prompt_len = len(all_tokens)
    print(f"  Prompt length: {prompt_len} tokens")
    print(f"  Decoder SEQ_LEN: {SEQ_LEN}")

    if prompt_len > SEQ_LEN:
        print(f"  ERROR: prompt ({prompt_len}) exceeds SEQ_LEN ({SEQ_LEN})!")
        sys.exit(1)

    # Show token breakdown
    print(f"  Token preview (first 20): {all_tokens[:20]}")
    audio_pad_positions = [i for i, t in enumerate(all_tokens) if t == AUDIO_PAD]
    print(f"  Audio pad positions: {audio_pad_positions[0]}..{audio_pad_positions[-1]} "
          f"({len(audio_pad_positions)} tokens)")
    print()

    # ------------------------------------------------------------------
    # 5. Build decoder inputs
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Step 4: Build decoder inputs")
    print("-" * 60)

    padded_embeds, pos_ids, attn_mask = build_decoder_inputs(
        all_tokens, audio_features, embed_table
    )
    print(f"  inputs_embeds:  {padded_embeds.shape} (dtype={padded_embeds.dtype})")
    print(f"  position_ids:   {pos_ids.shape} (dtype={pos_ids.dtype})")
    print(f"  attention_mask:  {attn_mask.shape} (dtype={attn_mask.dtype})")
    print()

    # ------------------------------------------------------------------
    # 6. Run decoder (prefill)
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Step 5: Decoder prefill")
    print("-" * 60)

    # Discover actual input names
    decoder_input_names = [inp.any_name for inp in decoder.inputs]
    print(f"  Decoder input names: {decoder_input_names}")

    decoder_inputs = {
        decoder_input_names[0]: padded_embeds,
        decoder_input_names[1]: pos_ids,
        decoder_input_names[2]: attn_mask,
    }

    t0 = time.perf_counter()
    decoder_result = decoder(decoder_inputs)
    prefill_time = time.perf_counter() - t0

    logits = list(decoder_result.values())[0]  # [1, 256, 151936]
    print(f"  Output shape: {logits.shape}")
    print(f"  Time: {prefill_time:.3f}s")

    # First predicted token
    next_logits = logits[0, prompt_len - 1, :]
    first_token_id = int(np.argmax(next_logits))
    first_token_text = tokenizer.decode([first_token_id])
    print(f"  First predicted token: id={first_token_id}, text='{first_token_text}'")

    # Top-5 tokens for debugging
    top5_ids = np.argsort(next_logits)[-5:][::-1]
    print(f"  Top-5 tokens: {[(int(tid), tokenizer.decode([int(tid)])) for tid in top5_ids]}")
    print()

    # ------------------------------------------------------------------
    # 7. Greedy generation loop (no KV-cache, O(n^2))
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Step 6: Greedy generation (no KV-cache)")
    print("-" * 60)

    MAX_NEW_TOKENS = 50
    generated_ids = list(all_tokens)
    gen_start = time.perf_counter()

    # Reuse audio_pad_positions for embedding replacement
    audio_pad_indices = np.array(audio_pad_positions, dtype=np.int64)
    audio_feats_np = np.array(audio_features[0])  # [104, 1024]

    for step in range(MAX_NEW_TOKENS):
        cur_len = len(generated_ids)
        if cur_len > SEQ_LEN:
            print(f"  Reached max context length {SEQ_LEN}, stopping.")
            break

        # Build inputs_embeds
        cur_ids = np.array(generated_ids, dtype=np.int64)
        cur_embeds = embed_table[cur_ids]  # [cur_len, 1024]

        # Replace audio_pad positions with encoder features
        cur_embeds[audio_pad_indices] = audio_feats_np

        # Pad to SEQ_LEN
        padded = np.zeros((1, SEQ_LEN, 1024), dtype=np.float32)
        padded[0, :cur_len, :] = cur_embeds

        # Position IDs (sequential, padded positions = 1)
        pos_1d = np.ones(SEQ_LEN, dtype=np.int64)
        pos_1d[:cur_len] = np.arange(cur_len, dtype=np.int64)
        pos = np.broadcast_to(pos_1d.reshape(1, 1, SEQ_LEN), (3, 1, SEQ_LEN)).copy()

        # Attention mask
        mask = np.zeros((1, SEQ_LEN), dtype=np.int64)
        mask[0, :cur_len] = 1

        # Run decoder
        result = decoder({
            decoder_input_names[0]: padded,
            decoder_input_names[1]: pos,
            decoder_input_names[2]: mask,
        })
        step_logits = list(result.values())[0]

        # Get next token (at position cur_len - 1)
        token_id = int(np.argmax(step_logits[0, cur_len - 1, :]))

        # Check for EOS
        if token_id == IM_END:
            print(f"  [Step {step:2d}] <|im_end|> -- generation complete")
            break

        generated_ids.append(token_id)
        token_text = tokenizer.decode([token_id])
        print(f"  [Step {step:2d}] id={token_id:6d}  '{token_text}'")

    gen_time = time.perf_counter() - gen_start
    num_generated = len(generated_ids) - len(all_tokens)

    # ------------------------------------------------------------------
    # 8. Final result
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Result")
    print("=" * 60)

    generated_only = generated_ids[len(all_tokens):]
    transcription = tokenizer.decode(generated_only, skip_special_tokens=False)

    print(f"  Encoder time:    {encoder_time:.3f}s")
    print(f"  Prefill time:    {prefill_time:.3f}s")
    print(f"  Generation time: {gen_time:.3f}s ({num_generated} tokens)")
    if num_generated > 0:
        print(f"  Avg per token:   {gen_time / num_generated:.3f}s")
    print()
    print(f"  Generated tokens: {generated_only}")
    print(f"  Transcription:    '{transcription}'")
    print()

    if audio_path is None:
        print("NOTE: This was a smoke test with silence. Provide an audio file")
        print("      as argument to test real transcription:")
        print("        uv run python scripts/test_e2e.py path/to/audio.wav")
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
