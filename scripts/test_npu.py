"""Test encoder and decoder on NPU device."""
import openvino as ov
import numpy as np
import time

core = ov.Core()
print("Available devices:", core.available_devices)

# --- Encoder on NPU ---
print("\n" + "=" * 60)
print("Compiling encoder on NPU...")
t0 = time.time()
try:
    encoder = core.compile_model(
        r"C:\Apps\Qwen3-ASR\models\encoder_fp16.xml",
        "NPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": r"C:\Apps\Qwen3-ASR\models\cache",
        },
    )
    compile_time = time.time() - t0
    print(f"Encoder compiled on NPU in {compile_time:.1f}s")

    mel = np.zeros((1, 128, 800), dtype=np.float32)
    t0 = time.time()
    result = encoder({"mel": mel})
    out = list(result.values())[0]
    first_time = time.time() - t0
    print(f"Encoder NPU inference: {out.shape}, first run: {first_time:.3f}s")

    for i in range(3):
        t0 = time.time()
        encoder({"mel": mel})
        elapsed = time.time() - t0
        print(f"  Run {i + 1}: {elapsed:.3f}s")
except Exception as e:
    print(f"Encoder NPU FAILED: {type(e).__name__}: {e}")

# --- Decoder on NPU ---
print("\n" + "=" * 60)
print("Compiling decoder on NPU...")
t0 = time.time()
try:
    decoder = core.compile_model(
        r"C:\Apps\Qwen3-ASR\models\decoder_fp16.xml",
        "NPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": r"C:\Apps\Qwen3-ASR\models\cache",
        },
    )
    compile_time = time.time() - t0
    print(f"Decoder compiled on NPU in {compile_time:.1f}s")

    embeds = np.zeros((1, 256, 1024), dtype=np.float32)
    pos = np.zeros((3, 1, 256), dtype=np.int64)
    mask = np.ones((1, 256), dtype=np.int64)

    t0 = time.time()
    result = decoder({"218": embeds, "170": pos, "attention_mask": mask})
    out = list(result.values())[0]
    first_time = time.time() - t0
    print(f"Decoder NPU inference: {out.shape}, first run: {first_time:.3f}s")

    for i in range(3):
        t0 = time.time()
        decoder({"218": embeds, "170": pos, "attention_mask": mask})
        elapsed = time.time() - t0
        print(f"  Run {i + 1}: {elapsed:.3f}s")
except Exception as e:
    print(f"Decoder NPU FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Done.")
