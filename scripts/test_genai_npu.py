"""Test Qwen3 decoder on NPU via openvino-genai LLMPipeline."""
import sys
import os
import time

os.environ["PYTHONIOENCODING"] = "utf-8"

import openvino_genai as ov_genai

# Use whichever export succeeded
MODEL_PATH = r"C:\Apps\Qwen3-ASR\models\decoder_genai_int4"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r"C:\Apps\Qwen3-ASR\models\decoder_genai_fp16"
if not os.path.exists(MODEL_PATH):
    # Fall back to the existing stateful export
    MODEL_PATH = r"C:\Apps\Qwen3-ASR\models\decoder_stateful_ov"

print(f"Using model: {MODEL_PATH}")
print(f"Files in model dir: {os.listdir(MODEL_PATH)}")

# NPU-specific config (from colleague's advice)
pipeline_config = {
    "MAX_PROMPT_LEN": 256,
    "MIN_RESPONSE_LEN": 50,
    "NPUW_LLM_PREFILL_CHUNK_SIZE": 256,
    "GENERATE_HINT": "BEST_PERF",
    "CACHE_DIR": r"C:\Apps\Qwen3-ASR\models\cache",
}

print("Creating LLMPipeline on NPU...")
t0 = time.time()
try:
    pipe = ov_genai.LLMPipeline(MODEL_PATH, "NPU", pipeline_config)
    device_used = "NPU"
    print(f"NPU Pipeline created in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"NPU Pipeline failed: {type(e).__name__}: {e}")
    print("\nTrying CPU fallback...")
    t0 = time.time()
    try:
        pipe = ov_genai.LLMPipeline(MODEL_PATH, "CPU")
        device_used = "CPU"
        print(f"CPU Pipeline created in {time.time()-t0:.1f}s")
    except Exception as e2:
        print(f"CPU Pipeline also failed: {type(e2).__name__}: {e2}")
        sys.exit(1)

print(f"\nDevice: {device_used}")

# Test generation - 5 tokens
print("\n--- Test: generating 5 tokens ---")
t0 = time.time()
result = pipe.generate(
    "Hello world",
    max_new_tokens=5,
)
gen_time = time.time() - t0
print(f"Result: {result}")
print(f"Time: {gen_time:.2f}s")

# Benchmark: generate 50 tokens
print("\n--- Benchmark: generating 50 tokens ---")
t0 = time.time()
result = pipe.generate(
    "The quick brown fox jumps over the lazy dog. This is a test of the language model.",
    max_new_tokens=50,
)
gen_time = time.time() - t0
print(f"Result: {result}")
print(f"Time: {gen_time:.2f}s ({gen_time/50*1000:.1f}ms/token)")

# Benchmark: generate 50 tokens again (warm run)
print("\n--- Benchmark: generating 50 tokens (warm) ---")
t0 = time.time()
result = pipe.generate(
    "The quick brown fox jumps over the lazy dog. This is a test of the language model.",
    max_new_tokens=50,
)
gen_time = time.time() - t0
print(f"Result: {result}")
print(f"Time: {gen_time:.2f}s ({gen_time/50*1000:.1f}ms/token)")

print(f"\nDone! Device used: {device_used}")
