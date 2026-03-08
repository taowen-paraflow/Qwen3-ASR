"""
Benchmark: NPU no-cache vs CPU stateful decoder for Qwen3-ASR.

Compares two decoder approaches to determine the fastest practical
decoder for the desktop ASR app:

1. NPU no-cache: decoder_fp16.xml on NPU, full recompute each step (O(n^2))
2. CPU stateful: decoder_stateful_ov/ on CPU, with KV-cache (O(n))

Benchmark scenario:
- Prompt length: ~125 tokens (audio + text template)
- Generate: 50 new tokens (typical transcription length)
- Measure total time for all 50 decode steps
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Apps\Qwen3-ASR")

import openvino as ov

MODEL_DIR = r"C:\Apps\Qwen3-ASR\models"


# ===========================================================================
# Approach 1: NPU no-cache (decoder_fp16.xml)
# Each step recomputes the full sequence -- O(n^2) total work
# ===========================================================================
def benchmark_npu_no_cache(n_gen_tokens=50):
    print("=" * 60)
    print("Approach 1: NPU no-cache (full recompute per step)")
    print("=" * 60)

    core = ov.Core()

    print("  Compiling model on NPU (may take a while on first run)...")
    t_compile = time.time()
    decoder = core.compile_model(
        os.path.join(MODEL_DIR, "decoder_fp16.xml"),
        "NPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": os.path.join(MODEL_DIR, "cache"),
        },
    )
    print(f"  Compile time: {time.time() - t_compile:.1f}s")

    # Get input names from the compiled model
    input_names = [inp.any_name for inp in decoder.inputs]
    print(f"  Input names: {input_names}")

    SEQ_LEN = 256
    PROMPT_LEN = 125

    # Build dummy inputs matching the model's static shapes
    # Input "218": [1, 256, 1024] -- input embeddings
    # Input "170": [3, 1, 256]    -- 3D rope position ids
    # Input "attention_mask": [1, 256]
    embeds = np.random.randn(1, SEQ_LEN, 1024).astype(np.float32)
    pos_ids = np.broadcast_to(
        np.arange(SEQ_LEN, dtype=np.int64).reshape(1, 1, SEQ_LEN),
        (3, 1, SEQ_LEN),
    ).copy()
    mask = np.ones((1, SEQ_LEN), dtype=np.int64)

    inputs = {
        input_names[0]: embeds,
        input_names[1]: pos_ids,
        input_names[2]: mask,
    }

    # Warmup
    print("  Warming up (3 runs)...")
    for _ in range(3):
        decoder(inputs)

    # Simulate generation: each step recomputes the full sequence
    # The attention mask controls how many tokens are "visible"
    print(f"  Generating {n_gen_tokens} tokens (full recompute each)...")
    times = []
    for step in range(n_gen_tokens):
        cur_len = PROMPT_LEN + step
        mask = np.zeros((1, SEQ_LEN), dtype=np.int64)
        mask[0, :cur_len] = 1
        inputs[input_names[2]] = mask

        t0 = time.perf_counter()
        decoder(inputs)
        times.append(time.perf_counter() - t0)

    total_ms = sum(times) * 1000
    avg_ms = float(np.mean(times)) * 1000
    print(f"  Total: {total_ms:.0f}ms")
    print(f"  Avg per step: {avg_ms:.1f}ms")
    print(f"  First step: {times[0]*1000:.1f}ms")
    print(f"  Last step: {times[-1]*1000:.1f}ms")
    return total_ms


# ===========================================================================
# Approach 2: CPU stateful (decoder_stateful_ov)
# Uses KV-cache: prefill once, then 1 token per step -- O(n) total
# ===========================================================================
def benchmark_cpu_stateful(n_gen_tokens=50):
    print("\n" + "=" * 60)
    print("Approach 2: CPU stateful (KV-cache, O(n) per step)")
    print("=" * 60)

    import torch
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    STATEFUL_DIR = os.path.join(MODEL_DIR, "decoder_stateful_ov")

    print("  Loading model on CPU...")
    t0 = time.time()
    model = OVModelForCausalLM.from_pretrained(
        STATEFUL_DIR,
        device="CPU",
        ov_config={"PERFORMANCE_HINT": "LATENCY"},
    )
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(STATEFUL_DIR)

    # Build a realistic prompt (~125 tokens)
    # Mirrors what Qwen3-ASR uses: system + user + audio placeholders + assistant
    prompt_tokens = (
        [151644]  # <|im_start|>
        + list(range(100, 110))  # "system\nYou are..."
        + [151645, 198]  # <|im_end|>\n
        + [151644]  # <|im_start|>
        + list(range(200, 206))  # "user\n"
        + [151669]  # <|audio_start|>
        + [151676] * 104  # <|audio_pad|> * 104
        + [151670, 151645, 198]  # <|audio_end|><|im_end|>\n
        + [151644]  # <|im_start|>
        + list(range(300, 304))  # "assistant\n"
    )

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    print(f"  Prompt length: {input_ids.shape[1]} tokens")

    # Warmup
    print("  Warming up (1 generation)...")
    _ = model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)

    # Benchmark
    print(f"  Generating {n_gen_tokens} tokens...")
    t0 = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=n_gen_tokens,
        do_sample=False,
    )
    total_time = time.perf_counter() - t0

    generated = output[0][input_ids.shape[1] :]
    actual_tokens = len(generated)
    total_ms = total_time * 1000
    per_token_ms = total_ms / actual_tokens if actual_tokens > 0 else 0

    print(f"  Total: {total_ms:.0f}ms for {actual_tokens} tokens")
    print(f"  Avg per token: {per_token_ms:.1f}ms")
    print(f"  Generated token ids: {generated.tolist()[:10]}...")
    return total_ms


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    N_TOKENS = 50

    print("Qwen3-ASR Decoder Benchmark")
    print(f"  OpenVINO version: {ov.__version__}")
    print(f"  Available devices: {ov.Core().available_devices}")
    print(f"  Tokens to generate: {N_TOKENS}")
    print()

    # ------ Approach 1: NPU no-cache ------
    try:
        npu_time = benchmark_npu_no_cache(N_TOKENS)
    except Exception as e:
        print(f"  NPU benchmark FAILED: {e}")
        import traceback
        traceback.print_exc()
        npu_time = float("inf")

    # ------ Approach 2: CPU stateful ------
    try:
        cpu_time = benchmark_cpu_stateful(N_TOKENS)
    except Exception as e:
        print(f"  CPU benchmark FAILED: {e}")
        import traceback
        traceback.print_exc()
        cpu_time = float("inf")

    # ------ Summary ------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  NPU no-cache  (50 tokens): {npu_time:>8.0f} ms")
    print(f"  CPU stateful  (50 tokens): {cpu_time:>8.0f} ms")
    if npu_time < cpu_time:
        ratio = cpu_time / npu_time if npu_time > 0 else float("inf")
        print(f"  --> Winner: NPU no-cache ({ratio:.1f}x faster)")
    else:
        ratio = npu_time / cpu_time if cpu_time > 0 else float("inf")
        print(f"  --> Winner: CPU stateful ({ratio:.1f}x faster)")
    print("=" * 60)
