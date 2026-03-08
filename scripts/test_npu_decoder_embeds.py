"""Test: Compile the IR-surgery stateful decoder on NPU with NPUW_LLM properties.

Attempts to run decoder_stateful_embeds on NPU using the NPUW LLM pipeline,
which is designed for stateful transformer models with KV-cache.

This tests whether the NPU can handle a decoder that uses inputs_embeds
(float tensor) instead of input_ids (token IDs).

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/test_npu_decoder_embeds.py'
"""

import sys
import os
import time
import traceback

import numpy as np
import openvino as ov

MODEL_DIR = r"C:\Apps\Qwen3-ASR\models"
DECODER_DIR = os.path.join(MODEL_DIR, "decoder_stateful_embeds")
DECODER_XML = os.path.join(DECODER_DIR, "openvino_model.xml")
EMBED_TABLE_NPY = os.path.join(MODEL_DIR, "embed_tokens.npy")

HIDDEN_SIZE = 1024


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    print()
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


def inspect_model(model):
    """Print detailed model input/output/state info."""
    print_subsection("Model Inputs")
    for i, inp in enumerate(model.inputs):
        try:
            name = inp.any_name
        except RuntimeError:
            name = f"(unnamed-{i})"
        print(f"  [{i}] name={name:20s}  shape={str(inp.partial_shape):25s}  type={inp.element_type}")

    print_subsection("Model Outputs")
    for i, out in enumerate(model.outputs):
        try:
            name = out.any_name
        except RuntimeError:
            name = f"(unnamed-{i})"
        print(f"  [{i}] name={name:20s}  shape={str(out.partial_shape):25s}  type={out.element_type}")

    # State variables (KV-cache)
    sinks = model.get_sinks()
    print_subsection(f"State Variables (Sinks: {len(sinks)})")
    for i, sink in enumerate(sinks):
        try:
            name = sink.get_friendly_name()
        except Exception:
            name = f"sink-{i}"
        print(f"  [{i}] {name}")
        if i >= 9:
            print(f"  ... and {len(sinks) - 10} more")
            break


def try_compile_npu(core, model):
    """Try to compile on NPU with NPUW_LLM config."""
    config = {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 2,
        "NPUW_LLM_MAX_PROMPT_LEN": 256,
        "NPUW_LLM_MIN_RESPONSE_LEN": 64,
    }

    print_subsection("Compiling on NPU with NPUW_LLM")
    print("  Config:")
    for k, v in config.items():
        print(f"    {k}: {v}")
    print()

    t0 = time.perf_counter()
    compiled = core.compile_model(model, "NPU", config)
    compile_time = time.perf_counter() - t0
    print(f"  Compilation succeeded in {compile_time*1000:.0f}ms")
    return compiled


def run_prefill_test(compiled):
    """Run a simple prefill inference test with dummy data."""
    print_subsection("Prefill Test (dummy inputs_embeds [1, 10, 1024])")

    request = compiled.create_infer_request()
    request.reset_state()

    seq_len = 10
    inputs_embeds = np.random.randn(1, seq_len, HIDDEN_SIZE).astype(np.float32) * 0.01
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    beam_idx = np.array([0], dtype=np.int32)
    input_ids = np.zeros((1, seq_len), dtype=np.int64)

    feed = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "beam_idx": beam_idx,
        "input_ids": input_ids,
    }

    print("  Running prefill inference...")
    t0 = time.perf_counter()
    request.infer(feed)
    infer_time = time.perf_counter() - t0

    logits = request.get_output_tensor(0).data
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")
    print(f"  Prefill time: {infer_time*1000:.1f}ms")

    # NPUW_LLM may return only the last token's logits: shape [1, 1, vocab]
    # Standard CPU returns full sequence logits: shape [1, seq_len, vocab]
    if logits.shape[1] == 1:
        print("  NOTE: NPUW_LLM returned only last-position logits (optimized)")
        predicted = int(np.argmax(logits[0, 0, :]))
    else:
        predicted = int(np.argmax(logits[0, seq_len - 1, :]))
    print(f"  Predicted token (last position): {predicted}")

    return request, seq_len


def run_decode_benchmark(request, past_len, embed_table):
    """Benchmark 20 single-token decode steps."""
    print_subsection("Decode Benchmark (20 single-token steps)")

    NUM_STEPS = 20
    # Start with a random token
    current_token_id = 198  # newline token as starting point
    step_times = []

    for step in range(NUM_STEPS):
        token_embed = embed_table[current_token_id][np.newaxis, np.newaxis, :]  # [1, 1, 1024]

        feed = {
            "inputs_embeds": token_embed.astype(np.float32),
            "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
            "position_ids": np.array([[past_len]], dtype=np.int64),
            "beam_idx": np.array([0], dtype=np.int32),
            "input_ids": np.array([[0]], dtype=np.int64),
        }

        t0 = time.perf_counter()
        request.infer(feed)
        step_time = time.perf_counter() - t0
        step_times.append(step_time)

        logits = request.get_output_tensor(0).data
        current_token_id = int(np.argmax(logits[0, -1, :]))
        past_len += 1

        print(f"  [Step {step:2d}] token={current_token_id:6d}  time={step_time*1000:.1f}ms")

    avg_ms = np.mean(step_times) * 1000
    min_ms = np.min(step_times) * 1000
    max_ms = np.max(step_times) * 1000
    total_ms = np.sum(step_times) * 1000

    print()
    print(f"  Decode stats ({NUM_STEPS} steps):")
    print(f"    Total:   {total_ms:.0f}ms")
    print(f"    Average: {avg_ms:.1f}ms/token")
    print(f"    Min:     {min_ms:.1f}ms/token")
    print(f"    Max:     {max_ms:.1f}ms/token")
    print(f"    Throughput: {1000.0 / avg_ms:.1f} tokens/sec")


def main():
    print_section("NPU Decoder Test: NPUW_LLM with inputs_embeds")

    core = ov.Core()
    print(f"  OpenVINO version: {ov.__version__}")
    print(f"  Available devices: {core.available_devices}")

    if "NPU" not in core.available_devices:
        print()
        print("  ERROR: NPU device not available. Cannot run this test.")
        sys.exit(1)

    # Print NPU device info
    try:
        npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
        print(f"  NPU device: {npu_name}")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 1. Read model
    # ------------------------------------------------------------------
    print_subsection("Loading Model")
    if not os.path.exists(DECODER_XML):
        print(f"  ERROR: Model not found: {DECODER_XML}")
        sys.exit(1)

    print(f"  Reading: {DECODER_XML}")
    t0 = time.perf_counter()
    model = core.read_model(DECODER_XML)
    read_time = time.perf_counter() - t0
    print(f"  Model read in {read_time*1000:.0f}ms")

    # ------------------------------------------------------------------
    # 2. Inspect model
    # ------------------------------------------------------------------
    inspect_model(model)

    # ------------------------------------------------------------------
    # 3. Try NPU compilation
    # ------------------------------------------------------------------
    try:
        compiled = try_compile_npu(core, model)
    except Exception as e:
        print()
        print("  *** NPU COMPILATION FAILED ***")
        print()
        print("  Full error:")
        traceback.print_exc()
        print()
        print("  This likely means NPUW_LLM does not support the inputs_embeds")
        print("  parameter layout in this IR-surgery model.")
        print()

        # Try without NPUW_LLM as a fallback diagnostic
        print_subsection("Fallback: Trying NPU without NPUW_LLM")
        try:
            config_basic = {
                "NPU_USE_NPUW": "YES",
            }
            print("  Config: NPU_USE_NPUW=YES only")
            t0 = time.perf_counter()
            compiled_basic = core.compile_model(model, "NPU", config_basic)
            compile_time = time.perf_counter() - t0
            print(f"  Basic NPU compilation succeeded in {compile_time*1000:.0f}ms")
        except Exception as e2:
            print(f"  Basic NPU compilation also failed: {e2}")
            traceback.print_exc()

        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Simple inference test
    # ------------------------------------------------------------------
    try:
        request, past_len = run_prefill_test(compiled)
    except Exception as e:
        print()
        print("  *** PREFILL INFERENCE FAILED ***")
        print()
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Decode benchmark
    # ------------------------------------------------------------------
    try:
        print()
        print("  Loading embedding table for decode benchmark...")
        embed_table = np.load(EMBED_TABLE_NPY)
        print(f"  Embedding table shape: {embed_table.shape}")
        run_decode_benchmark(request, past_len, embed_table)
    except Exception as e:
        print()
        print("  *** DECODE BENCHMARK FAILED ***")
        print()
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_section("Summary")
    print("  NPU compilation with NPUW_LLM: SUCCESS")
    print("  This means the IR-surgery decoder CAN run on NPU via NPUW.")
    print("  Compare ms/token with CPU baseline (~28ms/token) to evaluate")
    print("  whether NPU is worthwhile for this model.")


if __name__ == "__main__":
    main()
