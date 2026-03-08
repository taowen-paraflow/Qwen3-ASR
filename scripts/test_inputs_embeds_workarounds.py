"""Investigation Part 2: Workarounds for inputs_embeds injection.

Part 1 (test_inputs_embeds.py) showed that OVModelForCausalLM does NOT support
inputs_embeds -- neither at the IR level nor the API level.

This script investigates two workarounds:

1. IR Surgery: Modify the stateful OpenVINO model to replace the embedding
   lookup (Gather) with a direct inputs_embeds Parameter. This preserves
   the KV-cache stateful behavior for O(n) decoding.

2. Fallback: Benchmark decoder_fp16.xml (no KV-cache) on CPU with inputs_embeds
   via manual greedy decoding. O(n^2) but works today.

3. Re-export: Export Qwen3ForCausalLM with inputs_embeds input via
   ov.convert_model() with custom tracing.

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/test_inputs_embeds_workarounds.py'
"""

import sys
import os
import time
import traceback

sys.path.insert(0, r"C:\Apps\Qwen3-ASR")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np

MODEL_DIR = r"C:\Apps\Qwen3-ASR\models"
HF_MODEL_DIR = r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B"


def separator(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


# ======================================================================
# Approach 1: IR Surgery on the stateful model
# ======================================================================
def approach1_ir_surgery():
    separator("Approach 1: IR Surgery -- replace embedding with inputs_embeds Parameter")

    import openvino as ov
    from openvino import opset13 as opset

    core = ov.Core()

    stateful_path = os.path.join(MODEL_DIR, "decoder_stateful_ov", "openvino_model.xml")
    print(f"Loading stateful model: {stateful_path}")
    model = core.read_model(stateful_path)

    # Step 1: Inspect the graph to find the embedding layer
    print("\nStep 1: Inspecting graph for embedding operations...")
    print(f"Total operations: {len(model.get_ordered_ops())}")

    # Find operations related to input_ids and embedding
    gather_ops = []
    parameter_ops = []
    const_ops_large = []

    for op in model.get_ordered_ops():
        op_type = op.type_info.name
        op_name = op.get_friendly_name()

        if op_type == "Parameter":
            parameter_ops.append(op)
            try:
                name = op.output(0).any_name
            except RuntimeError:
                name = "<unnamed>"
            print(f"  Parameter: {op_name} -> '{name}' shape={op.output(0).partial_shape} type={op.output(0).element_type}")

        if op_type == "Gather":
            gather_ops.append(op)

        # Large constants might be the embedding table
        if op_type == "Constant":
            shape = op.output(0).partial_shape
            if shape.rank.get_length() == 2:
                dims = [shape[i].get_length() if shape[i].is_static else -1 for i in range(2)]
                if dims[0] > 100000:  # vocab-sized
                    const_ops_large.append(op)
                    print(f"  Large Constant: {op_name} shape={shape} type={op.output(0).element_type}")

    print(f"\nFound {len(gather_ops)} Gather ops")
    print(f"Found {len(const_ops_large)} large constants (vocab-sized)")

    # Step 2: Find the first Gather that uses input_ids
    print("\nStep 2: Finding embedding Gather...")
    input_ids_param = None
    for p in parameter_ops:
        try:
            if p.output(0).any_name == "input_ids":
                input_ids_param = p
                break
        except RuntimeError:
            pass

    if input_ids_param is None:
        print("  ERROR: Could not find input_ids Parameter!")
        return None

    print(f"  Found input_ids Parameter: {input_ids_param.get_friendly_name()}")

    # Trace consumers of input_ids to find the Gather
    embedding_gather = None
    for target_input in input_ids_param.output(0).get_target_inputs():
        node = target_input.get_node()
        print(f"  input_ids consumer: {node.type_info.name} '{node.get_friendly_name()}'")
        if node.type_info.name == "Gather":
            embedding_gather = node
        # Sometimes there's a Convert between Parameter and Gather
        elif node.type_info.name == "Convert":
            for target_input2 in node.output(0).get_target_inputs():
                node2 = target_input2.get_node()
                print(f"    Convert consumer: {node2.type_info.name} '{node2.get_friendly_name()}'")
                if node2.type_info.name == "Gather":
                    embedding_gather = node2

    if embedding_gather is None:
        print("  ERROR: Could not find embedding Gather!")
        # Let's check what operations consume input_ids
        print("\n  Tracing input_ids through the graph (first 5 levels)...")
        _trace_consumers(input_ids_param.output(0), depth=0, max_depth=5)
        return None

    print(f"\n  Found embedding Gather: {embedding_gather.get_friendly_name()}")
    gather_output = embedding_gather.output(0)
    gather_shape = gather_output.partial_shape
    print(f"  Gather output shape: {gather_shape}")
    print(f"  Gather output type: {gather_output.element_type}")
    print(f"  Gather consumers: {len(gather_output.get_target_inputs())}")
    for ti in gather_output.get_target_inputs():
        node = ti.get_node()
        print(f"    -> {node.type_info.name} '{node.get_friendly_name()}'")

    # Step 3: Create inputs_embeds Parameter with same shape
    print("\nStep 3: Attempting IR surgery...")
    print("  Creating inputs_embeds Parameter to replace Gather output...")

    # The Gather output shape is [batch, seq_len, hidden_size]
    # For the stateful model: [?,?, 1024] (dynamic batch and seq)
    try:
        from openvino import PartialShape, Dimension, Type as OVType

        # Create new parameter with same shape as Gather output
        inputs_embeds_shape = PartialShape([Dimension(-1), Dimension(-1), Dimension(1024)])
        inputs_embeds_param = opset.parameter(
            inputs_embeds_shape,
            dtype=np.float32,
            name="inputs_embeds"
        )
        inputs_embeds_param.output(0).set_names({"inputs_embeds"})
        print(f"  Created Parameter: inputs_embeds shape={inputs_embeds_shape}")

        # Replace Gather output with inputs_embeds
        # We need to redirect all consumers of the Gather output to the new Parameter
        consumers = list(gather_output.get_target_inputs())
        print(f"  Redirecting {len(consumers)} consumer(s)...")

        for target_input in consumers:
            target_input.replace_source_output(inputs_embeds_param.output(0))
            node = target_input.get_node()
            print(f"    Redirected: {node.type_info.name} '{node.get_friendly_name()}'")

        # Add the new parameter to the model
        model.add_parameters([inputs_embeds_param])

        # Remove input_ids parameter (it's no longer needed)
        # Actually, let's keep it for now and just not use it
        # The model will have both input_ids and inputs_embeds

        # Validate the model
        print("\n  Validating modified model...")
        model.validate_nodes_and_infer_types()
        print("  Validation passed!")

        # Check new inputs
        print("\n  Modified model inputs:")
        for inp in model.inputs:
            try:
                name = inp.any_name
            except RuntimeError:
                name = "<unnamed>"
            print(f"    {name:30s} shape={inp.partial_shape} type={inp.element_type}")

        print("\n  Modified model outputs:")
        for out in model.outputs:
            try:
                name = out.any_name
            except RuntimeError:
                name = "<unnamed>"
            print(f"    {name:30s} shape={out.partial_shape} type={out.element_type}")

        # Step 4: Try to compile and run the modified model
        print("\nStep 4: Compiling modified model on CPU...")
        compiled = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})

        # Test inference with inputs_embeds
        print("  Running test inference...")
        test_embeds = np.random.randn(1, 10, 1024).astype(np.float32)
        test_attn = np.ones((1, 10), dtype=np.int64)
        test_pos = np.arange(10, dtype=np.int64).reshape(1, -1)
        test_beam = np.array([0], dtype=np.int32)

        # Build inputs dict
        input_dict = {}
        for inp in compiled.inputs:
            try:
                name = inp.any_name
            except RuntimeError:
                continue
            if name == "inputs_embeds":
                input_dict[name] = test_embeds
            elif name == "attention_mask":
                input_dict[name] = test_attn
            elif name == "position_ids":
                input_dict[name] = test_pos
            elif name == "beam_idx":
                input_dict[name] = test_beam
            elif name == "input_ids":
                # Still present but not connected -- feed dummy
                input_dict[name] = np.array([[0] * 10], dtype=np.int64)

        result = compiled(input_dict)
        logits = list(result.values())[0]
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}")

        # Step 5: Save the modified model
        save_path = os.path.join(MODEL_DIR, "decoder_stateful_embeds")
        os.makedirs(save_path, exist_ok=True)
        xml_path = os.path.join(save_path, "openvino_model.xml")
        print(f"\n  Saving modified model to: {xml_path}")
        ov.save_model(model, xml_path)
        print("  Saved!")

        # Copy config files
        import shutil
        src_dir = os.path.join(MODEL_DIR, "decoder_stateful_ov")
        for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "vocab.json", "merges.txt",
                   "added_tokens.json", "generation_config.json"]:
            src = os.path.join(src_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(save_path, f))
        print("  Copied config files")

        return {
            "status": "success",
            "logits_shape": list(logits.shape),
            "save_path": save_path,
        }

    except Exception as e:
        print(f"\n  IR Surgery FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def _trace_consumers(output, depth, max_depth):
    """Recursively trace consumers of an output tensor."""
    if depth >= max_depth:
        return
    indent = "    " * (depth + 1)
    for target_input in output.get_target_inputs():
        node = target_input.get_node()
        print(f"{indent}{node.type_info.name} '{node.get_friendly_name()}' "
              f"(output shape: {node.output(0).partial_shape})")
        for i in range(node.get_output_size()):
            _trace_consumers(node.output(i), depth + 1, max_depth)


# ======================================================================
# Approach 2: Benchmark decoder_fp16.xml on CPU (no KV-cache fallback)
# ======================================================================
def approach2_benchmark_no_kvcache_cpu():
    separator("Approach 2: Benchmark decoder_fp16.xml on CPU (no KV-cache)")

    import openvino as ov
    core = ov.Core()

    decoder_path = os.path.join(MODEL_DIR, "decoder_fp16.xml")
    print(f"Loading model: {decoder_path}")
    decoder = core.compile_model(decoder_path, "CPU", {"PERFORMANCE_HINT": "LATENCY"})

    input_names = [inp.any_name for inp in decoder.inputs]
    print(f"Inputs: {input_names}")

    SEQ_LEN = 256
    PROMPT_LEN = 125
    N_TOKENS = 32

    # Build dummy inputs
    embeds = np.random.randn(1, SEQ_LEN, 1024).astype(np.float32)
    pos_ids = np.broadcast_to(
        np.arange(SEQ_LEN, dtype=np.int64).reshape(1, 1, SEQ_LEN),
        (3, 1, SEQ_LEN),
    ).copy()
    mask = np.zeros((1, SEQ_LEN), dtype=np.int64)
    mask[0, :PROMPT_LEN] = 1

    inputs = {
        input_names[0]: embeds,
        input_names[1]: pos_ids,
        input_names[2]: mask,
    }

    # Warmup
    print("Warming up (3 runs)...")
    for _ in range(3):
        decoder(inputs)

    # Benchmark: simulate greedy generation (O(n^2))
    print(f"Generating {N_TOKENS} tokens (full recompute each step)...")
    times = []
    for step in range(N_TOKENS):
        cur_len = PROMPT_LEN + step
        mask = np.zeros((1, SEQ_LEN), dtype=np.int64)
        mask[0, :cur_len] = 1
        inputs[input_names[2]] = mask

        t0 = time.perf_counter()
        decoder(inputs)
        times.append(time.perf_counter() - t0)

    total_ms = sum(times) * 1000
    avg_ms = float(np.mean(times)) * 1000
    print(f"\nResults (decoder_fp16.xml on CPU, no KV-cache):")
    print(f"  Total for {N_TOKENS} tokens: {total_ms:.0f}ms")
    print(f"  Average per step: {avg_ms:.1f}ms")
    print(f"  First step: {times[0]*1000:.1f}ms")
    print(f"  Last step: {times[-1]*1000:.1f}ms")

    return {
        "total_ms": total_ms,
        "avg_ms_per_step": avg_ms,
        "first_step_ms": times[0] * 1000,
        "last_step_ms": times[-1] * 1000,
    }


# ======================================================================
# Approach 3: Re-export with inputs_embeds via ov.convert_model
# ======================================================================
def approach3_reexport():
    separator("Approach 3: Re-export Qwen3ForCausalLM with inputs_embeds input")

    import torch
    import openvino as ov

    standalone_dir = os.path.join(MODEL_DIR, "qwen3_decoder_standalone")
    if not os.path.exists(standalone_dir):
        print(f"Standalone model not found: {standalone_dir}")
        print("Cannot re-export. Run export_decoder_stateful.py first.")
        return None

    print(f"Loading Qwen3ForCausalLM from: {standalone_dir}")
    from transformers import Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(standalone_dir, torch_dtype=torch.float32)
    model.eval()
    print(f"  Model loaded, num_params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Test forward with inputs_embeds (PyTorch level)
    print("\nTest PyTorch forward with inputs_embeds...")
    test_embeds = torch.randn(1, 10, 1024, dtype=torch.float32)
    test_mask = torch.ones(1, 10, dtype=torch.long)
    test_pos = torch.arange(10, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model(inputs_embeds=test_embeds, attention_mask=test_mask, position_ids=test_pos)
    print(f"  PyTorch output: {output.logits.shape}")
    print("  PyTorch inputs_embeds works!")

    # Now try ov.convert_model with inputs_embeds
    print("\nConverting with ov.convert_model (inputs_embeds input)...")
    try:
        example_input = {
            "inputs_embeds": test_embeds,
            "attention_mask": test_mask,
            "position_ids": test_pos,
        }

        ov_model = ov.convert_model(
            model,
            example_input=example_input,
        )

        print("  Conversion succeeded!")
        print("  Model inputs:")
        for inp in ov_model.inputs:
            try:
                name = inp.any_name
            except RuntimeError:
                name = "<unnamed>"
            print(f"    {name:30s} shape={inp.partial_shape} type={inp.element_type}")

        print("  Model outputs:")
        for out in ov_model.outputs:
            try:
                name = out.any_name
            except RuntimeError:
                name = "<unnamed>"
            print(f"    {name:30s} shape={out.partial_shape} type={out.element_type}")

        # Save the model
        save_path = os.path.join(MODEL_DIR, "decoder_embeds_ov")
        os.makedirs(save_path, exist_ok=True)
        xml_path = os.path.join(save_path, "openvino_model.xml")
        print(f"\n  Saving to: {xml_path}")
        ov.save_model(ov_model, xml_path, compress_to_fp16=True)
        print("  Saved!")

        # Test inference
        core = ov.Core()
        print("\n  Compiling on CPU...")
        compiled = core.compile_model(xml_path, "CPU", {"PERFORMANCE_HINT": "LATENCY"})

        test_inputs = {}
        for inp in compiled.inputs:
            try:
                name = inp.any_name
            except RuntimeError:
                continue
            if "embeds" in name.lower() or "embed" in name.lower():
                test_inputs[name] = test_embeds.numpy()
            elif "mask" in name.lower():
                test_inputs[name] = test_mask.numpy()
            elif "position" in name.lower():
                test_inputs[name] = test_pos.numpy()

        print(f"  Test inputs: {list(test_inputs.keys())}")
        result = compiled(test_inputs)
        logits = list(result.values())[0]
        print(f"  Logits shape: {logits.shape}")
        print("  CPU inference OK!")

        return {
            "status": "success",
            "save_path": save_path,
            "has_kv_cache": False,  # Basic conversion, no stateful KV-cache
            "note": "No KV-cache -- need optimum-intel stateful export for KV-cache"
        }

    except Exception as e:
        print(f"\n  Re-export FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ======================================================================
# Approach 4: Benchmark the IR-surgery model (if Approach 1 succeeded)
# ======================================================================
def approach4_benchmark_surgery_model():
    separator("Approach 4: Benchmark IR-surgery stateful model with inputs_embeds")

    import openvino as ov
    core = ov.Core()

    surgery_path = os.path.join(MODEL_DIR, "decoder_stateful_embeds", "openvino_model.xml")
    if not os.path.exists(surgery_path):
        print("IR-surgery model not found. Skipping benchmark.")
        return None

    print(f"Loading model: {surgery_path}")
    compiled = core.compile_model(surgery_path, "CPU", {"PERFORMANCE_HINT": "LATENCY"})

    # Print inputs
    input_map = {}
    for inp in compiled.inputs:
        try:
            name = inp.any_name
        except RuntimeError:
            continue
        input_map[name] = inp
        print(f"  Input: {name} shape={inp.partial_shape} type={inp.element_type}")

    PROMPT_LEN = 125
    N_TOKENS = 32

    # Test: prefill with full prompt
    print(f"\nPrefill test (prompt_len={PROMPT_LEN})...")
    embeds = np.random.randn(1, PROMPT_LEN, 1024).astype(np.float32)
    attn = np.ones((1, PROMPT_LEN), dtype=np.int64)
    pos = np.arange(PROMPT_LEN, dtype=np.int64).reshape(1, -1)
    beam = np.array([0], dtype=np.int32)

    inputs = {}
    if "inputs_embeds" in input_map:
        inputs["inputs_embeds"] = embeds
    if "attention_mask" in input_map:
        inputs["attention_mask"] = attn
    if "position_ids" in input_map:
        inputs["position_ids"] = pos
    if "beam_idx" in input_map:
        inputs["beam_idx"] = beam
    if "input_ids" in input_map:
        inputs["input_ids"] = np.zeros((1, PROMPT_LEN), dtype=np.int64)

    # Reset state (important for stateful model)
    request = compiled.create_infer_request()
    request.reset_state()

    t0 = time.perf_counter()
    request.infer(inputs)
    prefill_time = time.perf_counter() - t0
    logits = request.get_output_tensor(0).data
    print(f"  Prefill time: {prefill_time*1000:.1f}ms")
    print(f"  Logits shape: {logits.shape}")

    # Decode steps (single token at a time)
    print(f"\nDecode test ({N_TOKENS} tokens, single-token steps)...")
    times = []
    past_len = PROMPT_LEN
    for step in range(N_TOKENS):
        single_embed = np.random.randn(1, 1, 1024).astype(np.float32)
        single_attn = np.ones((1, past_len + 1), dtype=np.int64)
        single_pos = np.array([[past_len]], dtype=np.int64)

        step_inputs = {}
        if "inputs_embeds" in input_map:
            step_inputs["inputs_embeds"] = single_embed
        if "attention_mask" in input_map:
            step_inputs["attention_mask"] = single_attn
        if "position_ids" in input_map:
            step_inputs["position_ids"] = single_pos
        if "beam_idx" in input_map:
            step_inputs["beam_idx"] = beam
        if "input_ids" in input_map:
            step_inputs["input_ids"] = np.array([[0]], dtype=np.int64)

        t0 = time.perf_counter()
        request.infer(step_inputs)
        times.append(time.perf_counter() - t0)
        past_len += 1

    avg_ms = float(np.mean(times)) * 1000
    total_ms = sum(times) * 1000
    print(f"\nResults (stateful with inputs_embeds, KV-cache):")
    print(f"  Prefill: {prefill_time*1000:.1f}ms")
    print(f"  Total decode ({N_TOKENS} tokens): {total_ms:.0f}ms")
    print(f"  Average per decode step: {avg_ms:.1f}ms")
    print(f"  First decode step: {times[0]*1000:.1f}ms")
    print(f"  Last decode step: {times[-1]*1000:.1f}ms")

    return {
        "prefill_ms": prefill_time * 1000,
        "total_decode_ms": total_ms,
        "avg_decode_ms": avg_ms,
    }


# ======================================================================
# Main
# ======================================================================
def main():
    print()
    print("#" * 70)
    print("#  Investigation Part 2: Workarounds for inputs_embeds")
    print("#" * 70)
    print()

    # Approach 1: IR Surgery
    surgery_result = approach1_ir_surgery()

    # Approach 2: Benchmark no-KV-cache on CPU (fallback)
    no_kv_result = approach2_benchmark_no_kvcache_cpu()

    # Approach 3: Re-export with inputs_embeds
    reexport_result = approach3_reexport()

    # Approach 4: Benchmark IR-surgery model (if available)
    surgery_bench = approach4_benchmark_surgery_model()

    # ================================================================
    # Final Summary
    # ================================================================
    separator("FINAL SUMMARY -- Workarounds for inputs_embeds")

    print("Approach 1: IR Surgery (stateful + inputs_embeds)")
    if surgery_result and surgery_result.get("status") == "success":
        print(f"  STATUS: SUCCESS")
        print(f"  Saved to: {surgery_result.get('save_path')}")
        if surgery_bench:
            print(f"  Prefill: {surgery_bench['prefill_ms']:.1f}ms")
            print(f"  Decode: {surgery_bench['avg_decode_ms']:.1f}ms/token (KV-cache, O(n))")
            print(f"  Total {32} tokens: {surgery_bench['total_decode_ms']:.0f}ms + {surgery_bench['prefill_ms']:.1f}ms prefill")
    else:
        err = surgery_result.get("error", "unknown") if surgery_result else "not attempted"
        print(f"  STATUS: FAILED -- {err}")

    print()
    print("Approach 2: decoder_fp16.xml on CPU (no KV-cache, fallback)")
    if no_kv_result:
        print(f"  Decode: {no_kv_result['avg_ms_per_step']:.1f}ms/step (full recompute, O(n^2))")
        print(f"  Total {32} tokens: {no_kv_result['total_ms']:.0f}ms")
        print(f"  First step: {no_kv_result['first_step_ms']:.1f}ms, Last step: {no_kv_result['last_step_ms']:.1f}ms")
    else:
        print(f"  STATUS: FAILED")

    print()
    print("Approach 3: Re-export with inputs_embeds")
    if reexport_result and reexport_result.get("status") == "success":
        print(f"  STATUS: SUCCESS (but no KV-cache)")
        print(f"  Saved to: {reexport_result.get('save_path')}")
        print(f"  Note: {reexport_result.get('note')}")
    else:
        err = reexport_result.get("error", "unknown") if reexport_result else "not attempted"
        print(f"  STATUS: FAILED -- {err}")

    print()
    print("=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    if surgery_result and surgery_result.get("status") == "success" and surgery_bench:
        print("  IR Surgery WORKS! Use the stateful model with inputs_embeds.")
        print(f"  Speed: {surgery_bench['avg_decode_ms']:.1f}ms/token (with KV-cache)")
        print("  This is the optimal approach for streaming ASR:")
        print("    1. Encoder (NPU): mel -> audio features [1, 104, 1024]")
        print("    2. Build inputs_embeds: embed_tokens[text_ids] + audio features")
        print("    3. Prefill: feed full inputs_embeds to stateful model")
        print("    4. Decode: feed single-token embeds, KV-cache handles context")
    elif no_kv_result:
        kv_ratio = 1.0
        if surgery_bench:
            kv_ratio = no_kv_result["total_ms"] / (surgery_bench["total_decode_ms"] + surgery_bench["prefill_ms"])
        print("  Fallback to decoder_fp16.xml on CPU (no KV-cache)")
        print(f"  Speed: {no_kv_result['avg_ms_per_step']:.1f}ms/step")
        if surgery_bench:
            print(f"  This is {kv_ratio:.1f}x slower than the stateful approach")
    print()


if __name__ == "__main__":
    main()
