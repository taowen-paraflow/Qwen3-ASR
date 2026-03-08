"""Benchmark: NPU GenAI LLMPipeline vs CPU GenAI LLMPipeline vs CPU OVModelForCausalLM.

Compares three configurations for the Qwen3 0.6B decoder:
1. NPU + INT4 via openvino-genai LLMPipeline (with KV-cache)
2. CPU + INT4 via openvino-genai LLMPipeline (with KV-cache)
3. CPU + FP16 stateful via optimum-intel OVModelForCausalLM (existing baseline)
"""
import sys
import os
import time

os.environ["PYTHONIOENCODING"] = "utf-8"

NUM_TOKENS = 50
NUM_WARMUP = 1
NUM_RUNS = 3

PROMPT = "The quick brown fox jumps over the lazy dog. This is a test of the language model."

results = {}

# ============================================================
# 1. NPU GenAI LLMPipeline (INT4)
# ============================================================
print("=" * 60)
print("1. NPU GenAI LLMPipeline (INT4)")
print("=" * 60)

try:
    import openvino_genai as ov_genai

    MODEL_INT4 = r"C:\Apps\Qwen3-ASR\models\decoder_genai_int4"

    npu_config = {
        "MAX_PROMPT_LEN": 256,
        "MIN_RESPONSE_LEN": NUM_TOKENS,
        "NPUW_LLM_PREFILL_CHUNK_SIZE": 256,
        "GENERATE_HINT": "BEST_PERF",
        "CACHE_DIR": r"C:\Apps\Qwen3-ASR\models\cache",
    }

    print("Loading NPU pipeline...")
    t0 = time.time()
    npu_pipe = ov_genai.LLMPipeline(MODEL_INT4, "NPU", **npu_config)
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    # Warmup
    for _ in range(NUM_WARMUP):
        npu_pipe.generate(PROMPT, max_new_tokens=NUM_TOKENS)

    # Benchmark
    times = []
    for i in range(NUM_RUNS):
        t0 = time.time()
        result = npu_pipe.generate(PROMPT, max_new_tokens=NUM_TOKENS)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({elapsed/NUM_TOKENS*1000:.1f}ms/token)")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.3f}s ({avg/NUM_TOKENS*1000:.1f}ms/token)")
    results["NPU GenAI INT4"] = avg

    del npu_pipe
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# ============================================================
# 2. CPU GenAI LLMPipeline (INT4)
# ============================================================
print()
print("=" * 60)
print("2. CPU GenAI LLMPipeline (INT4)")
print("=" * 60)

try:
    import openvino_genai as ov_genai

    MODEL_INT4 = r"C:\Apps\Qwen3-ASR\models\decoder_genai_int4"

    print("Loading CPU GenAI pipeline...")
    t0 = time.time()
    cpu_genai_pipe = ov_genai.LLMPipeline(MODEL_INT4, "CPU")
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    # Warmup
    for _ in range(NUM_WARMUP):
        cpu_genai_pipe.generate(PROMPT, max_new_tokens=NUM_TOKENS)

    # Benchmark
    times = []
    for i in range(NUM_RUNS):
        t0 = time.time()
        result = cpu_genai_pipe.generate(PROMPT, max_new_tokens=NUM_TOKENS)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({elapsed/NUM_TOKENS*1000:.1f}ms/token)")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.3f}s ({avg/NUM_TOKENS*1000:.1f}ms/token)")
    results["CPU GenAI INT4"] = avg

    del cpu_genai_pipe
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# ============================================================
# 3. CPU OVModelForCausalLM (FP16 stateful) - baseline
# ============================================================
print()
print("=" * 60)
print("3. CPU OVModelForCausalLM (FP16 stateful) - baseline")
print("=" * 60)

try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    MODEL_STATEFUL = r"C:\Apps\Qwen3-ASR\models\decoder_stateful_ov"

    print("Loading CPU OVModel stateful...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_STATEFUL, trust_remote_code=True)
    model = OVModelForCausalLM.from_pretrained(MODEL_STATEFUL, device="CPU")
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    inputs = tokenizer(PROMPT, return_tensors="pt")

    # Warmup
    for _ in range(NUM_WARMUP):
        model.generate(**inputs, max_new_tokens=NUM_TOKENS, do_sample=False)

    # Benchmark
    times = []
    for i in range(NUM_RUNS):
        t0 = time.time()
        output = model.generate(**inputs, max_new_tokens=NUM_TOKENS, do_sample=False)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({elapsed/NUM_TOKENS*1000:.1f}ms/token)")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.3f}s ({avg/NUM_TOKENS*1000:.1f}ms/token)")
    results["CPU OVModel FP16 stateful"] = avg

    del model, tokenizer
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Config':<30} {'Total (s)':<12} {'Per token (ms)':<15} {'Speedup'}")
print("-" * 75)

baseline = results.get("CPU OVModel FP16 stateful", None)
for name, avg in sorted(results.items(), key=lambda x: x[1]):
    per_token = avg / NUM_TOKENS * 1000
    speedup = f"{baseline / avg:.2f}x" if baseline else "N/A"
    print(f"{name:<30} {avg:<12.3f} {per_token:<15.1f} {speedup}")

print()
print("Done!")
