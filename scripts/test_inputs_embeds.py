"""Investigation: Can we inject audio encoder embeddings into the text decoder?

The critical question for the streaming ASR engine is whether OVModelForCausalLM
supports `inputs_embeds` -- i.e., can we replace <|audio_pad|> token embeddings
with actual audio encoder output features?

Steps:
1. Check exported model inputs (IR-level) for all 3 decoder variants
2. Test OVModelForCausalLM forward pass with inputs_embeds
3. Test generate() with inputs_embeds
4. Benchmark inputs_embeds on CPU (32 tokens)
5. Compare with input_ids baseline
6. Simulate real ASR prompt with mixed text + audio embeddings

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/test_inputs_embeds.py'
"""

import sys
import os
import time
import traceback

sys.path.insert(0, r"C:\Apps\Qwen3-ASR")

import numpy as np

# Suppress warnings
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

MODEL_DIR = r"C:\Apps\Qwen3-ASR\models"
HF_MODEL_DIR = r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B"

# Special token IDs (from CLAUDE.md)
IM_START = 151644
IM_END = 151645
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676
ASR_TEXT = 151704
NEWLINE = 198


def separator(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


# ======================================================================
# Step 1: Check exported model inputs at the IR level
# ======================================================================
def step1_check_model_inputs():
    separator("Step 1: Check exported model inputs (IR-level)")

    import openvino as ov
    core = ov.Core()
    print(f"OpenVINO {ov.__version__}")
    print(f"Available devices: {core.available_devices}")
    print()

    models_to_check = [
        ("decoder_genai_int4", os.path.join(MODEL_DIR, "decoder_genai_int4", "openvino_model.xml")),
        ("decoder_stateful_ov", os.path.join(MODEL_DIR, "decoder_stateful_ov", "openvino_model.xml")),
        ("decoder_fp16 (no KV-cache)", os.path.join(MODEL_DIR, "decoder_fp16.xml")),
        ("encoder_fp16", os.path.join(MODEL_DIR, "encoder_fp16.xml")),
    ]

    results = {}
    for name, path in models_to_check:
        print(f"--- {name} ---")
        if not os.path.exists(path):
            print(f"  SKIPPED: {path} not found")
            print()
            continue

        try:
            m = core.read_model(path)
            has_inputs_embeds = False
            has_input_ids = False

            print(f"  Inputs ({len(m.inputs)}):")
            for inp in m.inputs:
                try:
                    inp_name = inp.any_name
                except RuntimeError:
                    inp_name = f"<unnamed:{inp.index}>"
                print(f"    {inp_name:30s} shape={inp.partial_shape}  type={inp.element_type}")
                if "inputs_embeds" in inp_name.lower() or "embed" in inp_name.lower():
                    has_inputs_embeds = True
                if "input_ids" in inp_name.lower():
                    has_input_ids = True

            print(f"  Outputs ({len(m.outputs)}):")
            for out in m.outputs:
                try:
                    out_name = out.any_name
                except RuntimeError:
                    out_name = f"<unnamed:{out.index}>"
                print(f"    {out_name:30s} shape={out.partial_shape}  type={out.element_type}")

            results[name] = {
                "has_inputs_embeds": has_inputs_embeds,
                "has_input_ids": has_input_ids,
                "num_inputs": len(m.inputs),
            }
            print(f"  >> has_input_ids: {has_input_ids}")
            print(f"  >> has_inputs_embeds: {has_inputs_embeds}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}
        print()

    return results


# ======================================================================
# Step 2: Test OVModelForCausalLM with inputs_embeds (forward pass)
# ======================================================================
def step2_test_forward_inputs_embeds():
    separator("Step 2: Test OVModelForCausalLM forward with inputs_embeds")

    import torch
    from optimum.intel import OVModelForCausalLM

    results = {}

    # Test both INT4 and stateful models
    models_to_test = [
        ("decoder_genai_int4", os.path.join(MODEL_DIR, "decoder_genai_int4")),
        ("decoder_stateful_ov", os.path.join(MODEL_DIR, "decoder_stateful_ov")),
    ]

    for name, path in models_to_test:
        print(f"--- Testing {name} ---")
        if not os.path.exists(path):
            print(f"  SKIPPED: not found")
            results[name] = {"status": "skipped"}
            print()
            continue

        try:
            print(f"  Loading model from {path}...")
            model = OVModelForCausalLM.from_pretrained(
                path,
                device="CPU",
                ov_config={"PERFORMANCE_HINT": "LATENCY"},
            )
            print(f"  Model loaded. Stateful: {getattr(model, 'stateful', 'N/A')}")

            # Check if model.request has inputs_embeds
            print(f"  Model input names: {[inp.any_name for inp in model.model.inputs]}")

            # Test 1: Forward with input_ids (baseline)
            print("  [Test 1] Forward with input_ids...")
            input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
            attn_mask = torch.ones(1, 5, dtype=torch.long)
            try:
                out1 = model(input_ids=input_ids, attention_mask=attn_mask)
                print(f"    OK - logits shape: {out1.logits.shape}")
            except Exception as e:
                print(f"    FAILED: {e}")

            # Test 2: Forward with inputs_embeds
            print("  [Test 2] Forward with inputs_embeds...")
            dummy_embeds = torch.randn(1, 10, 1024, dtype=torch.float32)
            attn_mask = torch.ones(1, 10, dtype=torch.long)
            try:
                out2 = model(inputs_embeds=dummy_embeds, attention_mask=attn_mask)
                print(f"    OK - logits shape: {out2.logits.shape}")
                results[name] = {"forward_inputs_embeds": True, "logits_shape": list(out2.logits.shape)}
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                results[name] = {"forward_inputs_embeds": False, "error": str(e)}

            # Test 3: Forward with inputs_embeds and position_ids
            print("  [Test 3] Forward with inputs_embeds + position_ids...")
            pos_ids = torch.arange(10, dtype=torch.long).unsqueeze(0)
            try:
                out3 = model(inputs_embeds=dummy_embeds, attention_mask=attn_mask, position_ids=pos_ids)
                print(f"    OK - logits shape: {out3.logits.shape}")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")

            del model
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = {"status": "load_failed", "error": str(e)}
        print()

    return results


# ======================================================================
# Step 3: Test generate() with inputs_embeds
# ======================================================================
def step3_test_generate_inputs_embeds():
    separator("Step 3: Test generate() with inputs_embeds")

    import torch
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    results = {}

    models_to_test = [
        ("decoder_genai_int4", os.path.join(MODEL_DIR, "decoder_genai_int4")),
        ("decoder_stateful_ov", os.path.join(MODEL_DIR, "decoder_stateful_ov")),
    ]

    for name, path in models_to_test:
        print(f"--- Testing {name} ---")
        if not os.path.exists(path):
            print(f"  SKIPPED: not found")
            results[name] = {"status": "skipped"}
            print()
            continue

        try:
            model = OVModelForCausalLM.from_pretrained(
                path,
                device="CPU",
                ov_config={"PERFORMANCE_HINT": "LATENCY"},
            )
            tokenizer = AutoTokenizer.from_pretrained(path)

            # Test generate with inputs_embeds
            print("  [Test] generate() with inputs_embeds...")
            dummy_embeds = torch.randn(1, 10, 1024, dtype=torch.float32)
            attn_mask = torch.ones(1, 10, dtype=torch.long)

            try:
                output = model.generate(
                    inputs_embeds=dummy_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=5,
                    do_sample=False,
                )
                gen_tokens = output[0].tolist()
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                print(f"    OK - generated {len(gen_tokens)} tokens: {gen_tokens}")
                print(f"    Decoded: '{gen_text[:100]}'")
                results[name] = {"generate_inputs_embeds": True, "tokens": gen_tokens}
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                results[name] = {"generate_inputs_embeds": False, "error": str(e)}

            del model
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            results[name] = {"status": "load_failed", "error": str(e)}
        print()

    return results


# ======================================================================
# Step 4: Benchmark inputs_embeds on CPU
# ======================================================================
def step4_benchmark():
    separator("Step 4: Benchmark inputs_embeds vs input_ids on CPU")

    import torch
    from optimum.intel import OVModelForCausalLM

    # Use the model that works (try INT4 first, fall back to stateful)
    for model_name, model_path in [
        ("decoder_genai_int4", os.path.join(MODEL_DIR, "decoder_genai_int4")),
        ("decoder_stateful_ov", os.path.join(MODEL_DIR, "decoder_stateful_ov")),
    ]:
        if not os.path.exists(model_path):
            continue

        print(f"Using model: {model_name}")
        model = OVModelForCausalLM.from_pretrained(
            model_path,
            device="CPU",
            ov_config={"PERFORMANCE_HINT": "LATENCY"},
        )

        N_TOKENS = 32
        PROMPT_LEN = 130  # ~typical ASR prompt length

        # ---- Benchmark 1: input_ids baseline ----
        print(f"\n  [Benchmark 1] input_ids, prompt_len={PROMPT_LEN}, gen={N_TOKENS} tokens")
        input_ids = torch.randint(0, 1000, (1, PROMPT_LEN), dtype=torch.long)
        attn_mask = torch.ones(1, PROMPT_LEN, dtype=torch.long)

        # Warmup
        try:
            model.generate(input_ids=input_ids, attention_mask=attn_mask,
                           max_new_tokens=5, do_sample=False)
        except Exception as e:
            print(f"    Warmup failed: {e}")

        times_ids = []
        for i in range(3):
            t0 = time.perf_counter()
            out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                 max_new_tokens=N_TOKENS, do_sample=False)
            elapsed = time.perf_counter() - t0
            actual_new = len(out[0]) - PROMPT_LEN
            times_ids.append(elapsed)
            print(f"    Run {i+1}: {elapsed*1000:.0f}ms ({actual_new} tokens, {elapsed/max(actual_new,1)*1000:.1f}ms/token)")

        avg_ids = sum(times_ids) / len(times_ids)
        print(f"    Average: {avg_ids*1000:.0f}ms total, {avg_ids/N_TOKENS*1000:.1f}ms/token")

        # ---- Benchmark 2: inputs_embeds ----
        print(f"\n  [Benchmark 2] inputs_embeds, prompt_len={PROMPT_LEN}, gen={N_TOKENS} tokens")
        embeds = torch.randn(1, PROMPT_LEN, 1024, dtype=torch.float32)
        attn_mask = torch.ones(1, PROMPT_LEN, dtype=torch.long)

        # Warmup
        try:
            model.generate(inputs_embeds=embeds, attention_mask=attn_mask,
                           max_new_tokens=5, do_sample=False)
        except Exception as e:
            print(f"    inputs_embeds generate() FAILED: {type(e).__name__}: {e}")
            print("    Cannot benchmark inputs_embeds - skipping")
            del model
            return {"input_ids_ms_per_token": avg_ids / N_TOKENS * 1000, "inputs_embeds": "FAILED"}

        times_embeds = []
        for i in range(3):
            t0 = time.perf_counter()
            out = model.generate(inputs_embeds=embeds, attention_mask=attn_mask,
                                 max_new_tokens=N_TOKENS, do_sample=False)
            elapsed = time.perf_counter() - t0
            actual_new = len(out[0])  # With embeds, output is only new tokens
            times_embeds.append(elapsed)
            print(f"    Run {i+1}: {elapsed*1000:.0f}ms ({actual_new} tokens, {elapsed/max(actual_new,1)*1000:.1f}ms/token)")

        avg_embeds = sum(times_embeds) / len(times_embeds)
        print(f"    Average: {avg_embeds*1000:.0f}ms total")

        # Summary
        print(f"\n  Summary for {model_name}:")
        print(f"    input_ids:     {avg_ids*1000:.0f}ms / {N_TOKENS} tokens = {avg_ids/N_TOKENS*1000:.1f}ms/token")
        print(f"    inputs_embeds: {avg_embeds*1000:.0f}ms / {N_TOKENS} tokens = {avg_embeds/N_TOKENS*1000:.1f}ms/token")
        ratio = avg_embeds / avg_ids if avg_ids > 0 else float("inf")
        print(f"    Ratio: {ratio:.2f}x")

        del model
        return {
            "model": model_name,
            "input_ids_ms_per_token": avg_ids / N_TOKENS * 1000,
            "inputs_embeds_ms_per_token": avg_embeds / N_TOKENS * 1000,
            "ratio": ratio,
        }

    return {"error": "No model found"}


# ======================================================================
# Step 5: Simulate real ASR prompt with mixed text + audio embeddings
# ======================================================================
def step5_simulate_asr_prompt():
    separator("Step 5: Simulate real ASR prompt (text + audio embeddings)")

    import torch
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    # Load model + tokenizer
    model_path = os.path.join(MODEL_DIR, "decoder_genai_int4")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "decoder_stateful_ov")

    print(f"Using model: {model_path}")
    model = OVModelForCausalLM.from_pretrained(
        model_path,
        device="CPU",
        ov_config={"PERFORMANCE_HINT": "LATENCY"},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load embedding table from the model itself
    # The model has embed_tokens inside -- we need to find it
    # For OVModelForCausalLM, we can get the embedding table from the IR
    print("  Loading embedding table...")
    embed_path = os.path.join(MODEL_DIR, "embed_tokens.npy")
    if os.path.exists(embed_path):
        embed_table = np.load(embed_path)
        print(f"  Loaded from npy: shape={embed_table.shape}, dtype={embed_table.dtype}")
    else:
        print(f"  embed_tokens.npy not found at {embed_path}")
        print("  Cannot simulate ASR prompt without embedding table")
        del model
        return None

    # Build ASR prompt tokens (same as test_e2e.py)
    AUDIO_PAD_COUNT = 104
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
    prompt_len = len(all_tokens)
    print(f"  Prompt length: {prompt_len} tokens")

    # Build inputs_embeds with simulated audio features
    input_ids = np.array(all_tokens, dtype=np.int64)
    inputs_embeds = embed_table[input_ids]  # [prompt_len, 1024]

    # Simulate audio encoder output (random for now -- just testing the pipeline)
    fake_audio_features = np.random.randn(AUDIO_PAD_COUNT, 1024).astype(np.float32) * 0.1

    # Replace audio_pad positions
    audio_pad_mask = (input_ids == AUDIO_PAD)
    audio_pad_positions = np.where(audio_pad_mask)[0]
    print(f"  Audio pad positions: {audio_pad_positions[0]}..{audio_pad_positions[-1]} ({len(audio_pad_positions)} tokens)")
    inputs_embeds[audio_pad_positions] = fake_audio_features

    # Convert to torch
    inputs_embeds_t = torch.from_numpy(inputs_embeds).unsqueeze(0).float()
    attn_mask = torch.ones(1, prompt_len, dtype=torch.long)

    print(f"  inputs_embeds shape: {inputs_embeds_t.shape}")
    print(f"  attention_mask shape: {attn_mask.shape}")

    # Test generate with the ASR prompt
    print("\n  Generating with mixed text+audio embeddings...")
    try:
        t0 = time.perf_counter()
        output = model.generate(
            inputs_embeds=inputs_embeds_t,
            attention_mask=attn_mask,
            max_new_tokens=20,
            do_sample=False,
        )
        elapsed = time.perf_counter() - t0

        gen_tokens = output[0].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        print(f"  OK! Generated {len(gen_tokens)} tokens in {elapsed*1000:.0f}ms")
        print(f"  Token IDs: {gen_tokens}")
        print(f"  Decoded: '{gen_text[:200]}'")

        # Now compare with input_ids approach
        print("\n  Comparing with input_ids approach (same token sequence)...")
        input_ids_t = torch.tensor([all_tokens], dtype=torch.long)
        t0 = time.perf_counter()
        output2 = model.generate(
            input_ids=input_ids_t,
            max_new_tokens=20,
            do_sample=False,
        )
        elapsed2 = time.perf_counter() - t0
        gen_tokens2 = output2[0][prompt_len:].tolist()
        gen_text2 = tokenizer.decode(gen_tokens2, skip_special_tokens=False)
        print(f"  input_ids result: {len(gen_tokens2)} tokens in {elapsed2*1000:.0f}ms")
        print(f"  Token IDs: {gen_tokens2}")
        print(f"  Decoded: '{gen_text2[:200]}'")

        result = {
            "inputs_embeds_works": True,
            "inputs_embeds_ms": elapsed * 1000,
            "input_ids_ms": elapsed2 * 1000,
        }
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        result = {"inputs_embeds_works": False, "error": str(e)}

    del model
    return result


# ======================================================================
# Step 6: Test with REAL encoder output
# ======================================================================
def step6_test_with_real_encoder():
    separator("Step 6: End-to-end with REAL encoder output + inputs_embeds")

    import torch
    import openvino as ov
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    core = ov.Core()

    # Load encoder
    encoder_path = os.path.join(MODEL_DIR, "encoder_fp16.xml")
    print("  Loading encoder on CPU...")
    encoder = core.compile_model(encoder_path, "CPU")
    encoder_input_name = encoder.inputs[0].any_name

    # Load decoder
    model_path = os.path.join(MODEL_DIR, "decoder_genai_int4")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "decoder_stateful_ov")
    print(f"  Loading decoder from {model_path}...")
    decoder = OVModelForCausalLM.from_pretrained(
        model_path,
        device="CPU",
        ov_config={"PERFORMANCE_HINT": "LATENCY"},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load embedding table
    embed_path = os.path.join(MODEL_DIR, "embed_tokens.npy")
    embed_table = np.load(embed_path)
    print(f"  Embedding table: {embed_table.shape}")

    # Generate silence audio (5s) and compute mel
    print("  Computing mel for 5s silence...")
    audio = np.zeros(int(16000 * 5.0), dtype=np.float32)
    feat_extractor = WhisperFeatureExtractor.from_pretrained(HF_MODEL_DIR)
    features = feat_extractor(audio, sampling_rate=16000, padding=True,
                              return_attention_mask=True, return_tensors="np")
    mel = features["input_features"].astype(np.float32)
    # Pad/trim to 800 frames
    T_FIXED = 800
    if mel.shape[2] < T_FIXED:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, T_FIXED - mel.shape[2])))
    elif mel.shape[2] > T_FIXED:
        mel = mel[:, :, :T_FIXED]
    print(f"  Mel shape: {mel.shape}")

    # Run encoder
    print("  Running encoder...")
    t0 = time.perf_counter()
    enc_result = encoder({encoder_input_name: mel})
    enc_time = time.perf_counter() - t0
    audio_features = list(enc_result.values())[0]  # [1, 104, 1024]
    print(f"  Encoder output: {audio_features.shape}, time={enc_time*1000:.0f}ms")

    # Build prompt with real encoder features
    AUDIO_PAD_COUNT = 104
    system_text = "system\nYou are a helpful assistant."
    system_tokens = [IM_START] + tokenizer.encode(system_text) + [IM_END, NEWLINE]
    user_text = "user\n"
    user_tokens = (
        [IM_START] + tokenizer.encode(user_text)
        + [AUDIO_START] + [AUDIO_PAD] * AUDIO_PAD_COUNT + [AUDIO_END, IM_END, NEWLINE]
    )
    assistant_text = "assistant\n"
    assistant_tokens = [IM_START] + tokenizer.encode(assistant_text)
    all_tokens = system_tokens + user_tokens + assistant_tokens
    prompt_len = len(all_tokens)

    # Build inputs_embeds
    input_ids = np.array(all_tokens, dtype=np.int64)
    inputs_embeds = embed_table[input_ids]  # [prompt_len, 1024]
    audio_pad_positions = np.where(input_ids == AUDIO_PAD)[0]
    inputs_embeds[audio_pad_positions] = np.array(audio_features[0])

    inputs_embeds_t = torch.from_numpy(inputs_embeds).unsqueeze(0).float()
    attn_mask = torch.ones(1, prompt_len, dtype=torch.long)

    # Generate
    print(f"  Generating (prompt_len={prompt_len}, max_new_tokens=30)...")
    t0 = time.perf_counter()
    try:
        output = decoder.generate(
            inputs_embeds=inputs_embeds_t,
            attention_mask=attn_mask,
            max_new_tokens=30,
            do_sample=False,
        )
        elapsed = time.perf_counter() - t0
        gen_tokens = output[0].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        print(f"  OK! Generated {len(gen_tokens)} tokens in {elapsed*1000:.0f}ms")
        print(f"  Token IDs: {gen_tokens}")
        print(f"  Decoded: '{gen_text}'")

        # Compare with test_e2e.py approach (input_ids, no embed injection)
        print("\n  Compare: input_ids without embed injection...")
        input_ids_t = torch.tensor([all_tokens], dtype=torch.long)
        t0 = time.perf_counter()
        output2 = decoder.generate(input_ids=input_ids_t, max_new_tokens=30, do_sample=False)
        elapsed2 = time.perf_counter() - t0
        gen_tokens2 = output2[0][prompt_len:].tolist()
        gen_text2 = tokenizer.decode(gen_tokens2, skip_special_tokens=False)
        print(f"  input_ids result: {len(gen_tokens2)} tokens in {elapsed2*1000:.0f}ms")
        print(f"  Decoded: '{gen_text2}'")

        return {
            "e2e_works": True,
            "inputs_embeds_text": gen_text,
            "input_ids_text": gen_text2,
        }
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"e2e_works": False, "error": str(e)}
    finally:
        del decoder


# ======================================================================
# Main
# ======================================================================
def main():
    print()
    print("#" * 70)
    print("#  Investigation: Can we inject audio encoder embeddings")
    print("#  into the text decoder via inputs_embeds?")
    print("#" * 70)
    print()

    # Step 1
    ir_results = step1_check_model_inputs()

    # Step 2
    forward_results = step2_test_forward_inputs_embeds()

    # Step 3
    generate_results = step3_test_generate_inputs_embeds()

    # Step 4
    benchmark_results = step4_benchmark()

    # Step 5
    asr_sim_results = step5_simulate_asr_prompt()

    # Step 6
    e2e_results = step6_test_with_real_encoder()

    # ================================================================
    # Final Summary
    # ================================================================
    separator("FINAL SUMMARY")

    print("1. Model IR-level inputs:")
    for name, info in ir_results.items():
        if "error" in info:
            print(f"   {name}: ERROR - {info['error']}")
        else:
            embeds = "YES" if info["has_inputs_embeds"] else "NO"
            ids = "YES" if info["has_input_ids"] else "NO"
            print(f"   {name}: input_ids={ids}, inputs_embeds={embeds} ({info['num_inputs']} total inputs)")

    print()
    print("2. OVModelForCausalLM.forward(inputs_embeds=...) support:")
    for name, info in forward_results.items():
        if info.get("forward_inputs_embeds"):
            print(f"   {name}: WORKS - logits {info.get('logits_shape')}")
        else:
            print(f"   {name}: FAILED - {info.get('error', info.get('status', '?'))}")

    print()
    print("3. OVModelForCausalLM.generate(inputs_embeds=...) support:")
    for name, info in generate_results.items():
        if info.get("generate_inputs_embeds"):
            print(f"   {name}: WORKS - generated {len(info.get('tokens', []))} tokens")
        else:
            print(f"   {name}: FAILED - {info.get('error', info.get('status', '?'))}")

    print()
    print("4. Benchmark (CPU, 32 tokens):")
    if isinstance(benchmark_results, dict) and "input_ids_ms_per_token" in benchmark_results:
        print(f"   Model: {benchmark_results.get('model', '?')}")
        print(f"   input_ids:     {benchmark_results['input_ids_ms_per_token']:.1f} ms/token")
        if "inputs_embeds_ms_per_token" in benchmark_results:
            print(f"   inputs_embeds: {benchmark_results['inputs_embeds_ms_per_token']:.1f} ms/token")
            print(f"   Ratio: {benchmark_results.get('ratio', '?'):.2f}x")
        else:
            print(f"   inputs_embeds: {benchmark_results.get('inputs_embeds', 'N/A')}")
    else:
        print(f"   {benchmark_results}")

    print()
    print("5. ASR prompt simulation:")
    if asr_sim_results and asr_sim_results.get("inputs_embeds_works"):
        print(f"   WORKS - inputs_embeds: {asr_sim_results.get('inputs_embeds_ms', '?'):.0f}ms, "
              f"input_ids: {asr_sim_results.get('input_ids_ms', '?'):.0f}ms")
    else:
        err = asr_sim_results.get("error", "N/A") if asr_sim_results else "N/A"
        print(f"   FAILED: {err}")

    print()
    print("6. End-to-end with real encoder:")
    if e2e_results and e2e_results.get("e2e_works"):
        print(f"   WORKS!")
        print(f"   inputs_embeds output: '{e2e_results.get('inputs_embeds_text', '')[:100]}'")
        print(f"   input_ids output:     '{e2e_results.get('input_ids_text', '')[:100]}'")
    else:
        err = e2e_results.get("error", "N/A") if e2e_results else "N/A"
        print(f"   FAILED: {err}")

    print()
    print("=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)

    # Determine recommendation based on results
    embeds_works = any(
        info.get("generate_inputs_embeds") for info in generate_results.values()
    )
    if embeds_works:
        print("  OVModelForCausalLM.generate(inputs_embeds=...) WORKS on CPU.")
        print("  Recommended approach for streaming ASR engine:")
        print("    1. Audio encoder (NPU) -> audio features [1, 104, 1024]")
        print("    2. Build inputs_embeds: embed text tokens via embed_tokens.npy,")
        print("       replace <|audio_pad|> positions with encoder features")
        print("    3. OVModelForCausalLM.generate(inputs_embeds=...) on CPU")
        print("    4. KV-cache stateful model gives O(n) decoding per token")
        if benchmark_results and isinstance(benchmark_results, dict):
            ms = benchmark_results.get("inputs_embeds_ms_per_token", "?")
            print(f"    5. Expected speed: ~{ms} ms/token on CPU")
    else:
        print("  inputs_embeds does NOT work with generate().")
        print("  Fallback: Use the no-KV-cache decoder_fp16.xml approach")
        print("  (as in test_e2e.py) with manual greedy decoding loop.")
    print()


if __name__ == "__main__":
    main()
