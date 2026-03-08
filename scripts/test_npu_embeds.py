import openvino as ov
import numpy as np
import time

core = ov.Core()

# 1. Read the IR-surgery model
model = core.read_model(r"C:\Apps\Qwen3-ASR\models\decoder_stateful_embeds\openvino_model.xml")

print("=== Original inputs ===")
for inp in model.inputs:
    print(f"  {inp.any_name}: shape={inp.partial_shape}, type={inp.element_type}")

print("\n=== Original outputs ===")
for out in model.outputs:
    print(f"  {out.any_name}: shape={out.partial_shape}, type={out.element_type}")

print("\n=== State variables (KV-cache) ===")
for var in model.get_sinks():
    print(f"  sink: {var}")
for var in model.get_parameters():
    print(f"  param: {var.friendly_name} shape={var.partial_shape}")

# Count internal state ops
state_count = 0
for op in model.get_ordered_ops():
    type_name = op.get_type_name()
    if "ReadValue" in type_name or "Assign" in type_name:
        state_count += 1
print(f"  Total ReadValue/Assign ops: {state_count}")
print(f"  KV-cache layers: {state_count // 2} (K+V per layer)")


def try_npu_compile(label, reshape_dict, test_tensors):
    """Try to reshape and compile on NPU."""
    print(f"\n{'='*60}")
    print(f"=== {label} ===")
    print(f"{'='*60}")

    try:
        m = core.read_model(r"C:\Apps\Qwen3-ASR\models\decoder_stateful_embeds\openvino_model.xml")
        print(f"  Reshape dict: {reshape_dict}")
        m.reshape(reshape_dict)
        print("  Reshape succeeded!")

        # Verify all inputs are now static
        all_static = True
        for inp in m.inputs:
            is_static = inp.partial_shape.is_static
            print(f"    {inp.any_name}: {inp.partial_shape} (static={is_static})")
            if not is_static:
                all_static = False

        for out in m.outputs:
            is_static = out.partial_shape.is_static
            print(f"    output {out.any_name}: {out.partial_shape} (static={is_static})")
            if not is_static:
                all_static = False

        if not all_static:
            print("  WARNING: Not all shapes are static after reshape!")

        # Try NPU compilation
        print(f"\n  Attempting NPU compilation...")
        try:
            compiled = core.compile_model(m, "NPU", {
                "PERFORMANCE_HINT": "LATENCY",
                "CACHE_DIR": r"C:\Apps\Qwen3-ASR\models\cache",
            })
            print("  NPU compilation SUCCEEDED!")

            # Try inference
            request = compiled.create_infer_request()
            for name, tensor in test_tensors.items():
                if name in [inp.any_name for inp in compiled.inputs]:
                    request.set_tensor(name, ov.Tensor(tensor))

            request.infer()
            logits = request.get_output_tensor(0).data
            print(f"  Output logits shape: {logits.shape}")

            # Benchmark
            times = []
            for i in range(20):
                t0 = time.perf_counter()
                for name, tensor in test_tensors.items():
                    if name in [inp.any_name for inp in compiled.inputs]:
                        request.set_tensor(name, ov.Tensor(tensor))
                request.infer()
                times.append(time.perf_counter() - t0)

            avg = sum(times) / len(times)
            print(f"  Benchmark (20 iters): avg={avg*1000:.1f}ms, min={min(times)*1000:.1f}ms, max={max(times)*1000:.1f}ms")
            return True

        except Exception as e:
            print(f"  NPU compilation FAILED: {e}")

            # Try NPUW fallback
            print(f"\n  Retry with NPU_USE_NPUW=YES...")
            try:
                compiled = core.compile_model(m, "NPU", {
                    "PERFORMANCE_HINT": "LATENCY",
                    "CACHE_DIR": r"C:\Apps\Qwen3-ASR\models\cache",
                    "NPU_USE_NPUW": "YES",
                })
                print("  NPU+NPUW compilation SUCCEEDED!")

                request = compiled.create_infer_request()
                for name, tensor in test_tensors.items():
                    if name in [inp.any_name for inp in compiled.inputs]:
                        request.set_tensor(name, ov.Tensor(tensor))
                request.infer()
                logits = request.get_output_tensor(0).data
                print(f"  Output logits shape: {logits.shape}")

                times = []
                for i in range(20):
                    t0 = time.perf_counter()
                    for name, tensor in test_tensors.items():
                        if name in [inp.any_name for inp in compiled.inputs]:
                            request.set_tensor(name, ov.Tensor(tensor))
                    request.infer()
                    times.append(time.perf_counter() - t0)
                avg = sum(times) / len(times)
                print(f"  Benchmark (20 iters): avg={avg*1000:.1f}ms, min={min(times)*1000:.1f}ms, max={max(times)*1000:.1f}ms")
                return True

            except Exception as e2:
                print(f"  NPU+NPUW also FAILED: {e2}")
                return False

    except Exception as e:
        print(f"  Reshape FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Test 1: Decode (seq=1) — reshape ALL inputs including input_ids
# ============================================================
try_npu_compile(
    "Decode seq=1 (all inputs reshaped)",
    {
        "input_ids": [1, 1],
        "inputs_embeds": [1, 1, 1024],
        "attention_mask": [1, 1],
        "position_ids": [1, 1],
        "beam_idx": [1],
    },
    {
        "input_ids": np.array([[0]], dtype=np.int64),
        "inputs_embeds": np.random.randn(1, 1, 1024).astype(np.float32),
        "attention_mask": np.ones((1, 1), dtype=np.int64),
        "position_ids": np.zeros((1, 1), dtype=np.int64),
        "beam_idx": np.array([0], dtype=np.int32),
    }
)

# ============================================================
# Test 2: Prefill (seq=256) — reshape ALL inputs
# ============================================================
try_npu_compile(
    "Prefill seq=256 (all inputs reshaped)",
    {
        "input_ids": [1, 256],
        "inputs_embeds": [1, 256, 1024],
        "attention_mask": [1, 256],
        "position_ids": [1, 256],
        "beam_idx": [1],
    },
    {
        "input_ids": np.zeros((1, 256), dtype=np.int64),
        "inputs_embeds": np.random.randn(1, 256, 1024).astype(np.float32),
        "attention_mask": np.ones((1, 256), dtype=np.int64),
        "position_ids": np.arange(256, dtype=np.int64).reshape(1, -1),
        "beam_idx": np.array([0], dtype=np.int32),
    }
)

# ============================================================
# Test 3: Check if attention_mask needs to be longer (for KV-cache)
# In stateful models, attention_mask length = past_kv_length + current_seq
# Try decode with attention_mask=[1, 257] (256 past + 1 current)
# ============================================================
try_npu_compile(
    "Decode seq=1 with extended attention_mask=[1,257]",
    {
        "input_ids": [1, 1],
        "inputs_embeds": [1, 1, 1024],
        "attention_mask": [1, 257],  # past=256 + current=1
        "position_ids": [1, 1],
        "beam_idx": [1],
    },
    {
        "input_ids": np.array([[0]], dtype=np.int64),
        "inputs_embeds": np.random.randn(1, 1, 1024).astype(np.float32),
        "attention_mask": np.ones((1, 257), dtype=np.int64),
        "position_ids": np.array([[256]], dtype=np.int64),
        "beam_idx": np.array([0], dtype=np.int32),
    }
)

# ============================================================
# Test 4: Try CPU compilation as sanity check (decode seq=1, all reshaped)
# ============================================================
print(f"\n{'='*60}")
print("=== CPU sanity check: decode seq=1 (all inputs reshaped) ===")
print(f"{'='*60}")
try:
    m_cpu = core.read_model(r"C:\Apps\Qwen3-ASR\models\decoder_stateful_embeds\openvino_model.xml")
    m_cpu.reshape({
        "input_ids": [1, 1],
        "inputs_embeds": [1, 1, 1024],
        "attention_mask": [1, 1],
        "position_ids": [1, 1],
        "beam_idx": [1],
    })
    compiled_cpu = core.compile_model(m_cpu, "CPU")
    print("  CPU compilation succeeded!")

    request_cpu = compiled_cpu.create_infer_request()
    request_cpu.set_tensor("inputs_embeds", ov.Tensor(np.random.randn(1, 1, 1024).astype(np.float32)))
    request_cpu.set_tensor("input_ids", ov.Tensor(np.array([[0]], dtype=np.int64)))
    request_cpu.set_tensor("attention_mask", ov.Tensor(np.ones((1, 1), dtype=np.int64)))
    request_cpu.set_tensor("position_ids", ov.Tensor(np.zeros((1, 1), dtype=np.int64)))
    request_cpu.set_tensor("beam_idx", ov.Tensor(np.array([0], dtype=np.int32)))

    request_cpu.infer()
    logits_cpu = request_cpu.get_output_tensor(0).data
    print(f"  Output logits shape: {logits_cpu.shape}")

    times = []
    for i in range(20):
        t0 = time.perf_counter()
        request_cpu.set_tensor("inputs_embeds", ov.Tensor(np.random.randn(1, 1, 1024).astype(np.float32)))
        request_cpu.set_tensor("input_ids", ov.Tensor(np.array([[0]], dtype=np.int64)))
        request_cpu.set_tensor("attention_mask", ov.Tensor(np.ones((1, 1), dtype=np.int64)))
        request_cpu.set_tensor("position_ids", ov.Tensor(np.array([[i]], dtype=np.int64)))
        request_cpu.set_tensor("beam_idx", ov.Tensor(np.array([0], dtype=np.int32)))
        request_cpu.infer()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  CPU Benchmark (20 iters): avg={avg*1000:.1f}ms, min={min(times)*1000:.1f}ms, max={max(times)*1000:.1f}ms")

except Exception as e:
    print(f"  CPU compilation failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Test 5: Try GPU compilation (Intel Arc) as alternative
# ============================================================
print(f"\n{'='*60}")
print("=== GPU test: decode seq=1 (all inputs reshaped) ===")
print(f"{'='*60}")
available = core.available_devices
print(f"  Available devices: {available}")
if "GPU" in available:
    try:
        m_gpu = core.read_model(r"C:\Apps\Qwen3-ASR\models\decoder_stateful_embeds\openvino_model.xml")
        # GPU supports dynamic shapes, but let's try static first
        m_gpu.reshape({
            "input_ids": [1, 1],
            "inputs_embeds": [1, 1, 1024],
            "attention_mask": [1, 1],
            "position_ids": [1, 1],
            "beam_idx": [1],
        })
        compiled_gpu = core.compile_model(m_gpu, "GPU", {
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": r"C:\Apps\Qwen3-ASR\models\cache",
        })
        print("  GPU compilation succeeded!")

        request_gpu = compiled_gpu.create_infer_request()
        request_gpu.set_tensor("inputs_embeds", ov.Tensor(np.random.randn(1, 1, 1024).astype(np.float32)))
        request_gpu.set_tensor("input_ids", ov.Tensor(np.array([[0]], dtype=np.int64)))
        request_gpu.set_tensor("attention_mask", ov.Tensor(np.ones((1, 1), dtype=np.int64)))
        request_gpu.set_tensor("position_ids", ov.Tensor(np.zeros((1, 1), dtype=np.int64)))
        request_gpu.set_tensor("beam_idx", ov.Tensor(np.array([0], dtype=np.int32)))

        request_gpu.infer()
        logits_gpu = request_gpu.get_output_tensor(0).data
        print(f"  Output logits shape: {logits_gpu.shape}")

        # Warmup
        for _ in range(5):
            request_gpu.infer()

        times = []
        for i in range(20):
            t0 = time.perf_counter()
            request_gpu.set_tensor("inputs_embeds", ov.Tensor(np.random.randn(1, 1, 1024).astype(np.float32)))
            request_gpu.set_tensor("input_ids", ov.Tensor(np.array([[0]], dtype=np.int64)))
            request_gpu.set_tensor("attention_mask", ov.Tensor(np.ones((1, 1), dtype=np.int64)))
            request_gpu.set_tensor("position_ids", ov.Tensor(np.array([[i]], dtype=np.int64)))
            request_gpu.set_tensor("beam_idx", ov.Tensor(np.array([0], dtype=np.int32)))
            request_gpu.infer()
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        print(f"  GPU Benchmark (20 iters): avg={avg*1000:.1f}ms, min={min(times)*1000:.1f}ms, max={max(times)*1000:.1f}ms")
    except Exception as e:
        print(f"  GPU compilation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  GPU not available, skipping")
