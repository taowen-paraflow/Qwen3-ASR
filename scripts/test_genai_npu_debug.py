"""Debug: verify NPU is actually being used, and test with longer generation."""
import sys
import os
import time

os.environ["PYTHONIOENCODING"] = "utf-8"

import openvino as ov
import openvino_genai as ov_genai

MODEL_INT4 = r"C:\Apps\Qwen3-ASR\models\decoder_genai_int4"

# Check available NPU properties
core = ov.Core()
print("Available devices:", core.available_devices)

# Try to read NPU-related properties
for prop_name in ["FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "OPTIMIZATION_CAPABILITIES"]:
    try:
        val = core.get_property("NPU", prop_name)
        print(f"NPU {prop_name}: {val}")
    except Exception as e:
        print(f"NPU {prop_name}: (unavailable) {e}")

# Test: NPU pipeline with verbose config
print("\n--- NPU Pipeline (with NPUW debug) ---")
npu_config = {
    "MAX_PROMPT_LEN": 256,
    "MIN_RESPONSE_LEN": 128,
    "NPUW_LLM_PREFILL_CHUNK_SIZE": 256,
    "GENERATE_HINT": "BEST_PERF",
    "CACHE_DIR": r"C:\Apps\Qwen3-ASR\models\cache",
}

t0 = time.time()
npu_pipe = ov_genai.LLMPipeline(MODEL_INT4, "NPU", **npu_config)
print(f"NPU pipeline loaded in {time.time()-t0:.1f}s")

# Generate 100 tokens
print("\nGenerating 100 tokens on NPU...")
t0 = time.time()
result = npu_pipe.generate(
    "Explain the theory of relativity in simple terms.",
    max_new_tokens=100,
)
elapsed = time.time() - t0
print(f"Time: {elapsed:.3f}s ({elapsed/100*1000:.1f}ms/token)")
print(f"Result: {result[:200]}...")

# Generate 200 tokens
print("\nGenerating 200 tokens on NPU...")
t0 = time.time()
result = npu_pipe.generate(
    "Write a short essay about artificial intelligence.",
    max_new_tokens=200,
)
elapsed = time.time() - t0
print(f"Time: {elapsed:.3f}s ({elapsed/200*1000:.1f}ms/token)")
print(f"Result: {result[:300]}...")

del npu_pipe

# CPU comparison at 200 tokens
print("\n--- CPU GenAI Pipeline ---")
t0 = time.time()
cpu_pipe = ov_genai.LLMPipeline(MODEL_INT4, "CPU")
print(f"CPU pipeline loaded in {time.time()-t0:.1f}s")

print("\nGenerating 200 tokens on CPU...")
t0 = time.time()
result = cpu_pipe.generate(
    "Write a short essay about artificial intelligence.",
    max_new_tokens=200,
)
elapsed = time.time() - t0
print(f"Time: {elapsed:.3f}s ({elapsed/200*1000:.1f}ms/token)")
print(f"Result: {result[:300]}...")

del cpu_pipe

print("\nDone!")
