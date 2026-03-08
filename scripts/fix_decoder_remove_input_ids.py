"""Fix NPU decoder: remove input_ids from IR-surgery model.

Root cause: NPUW_LLM plugin checks for input_ids first (llm_infer_request.cpp:124).
If found, it routes through input_ids instead of inputs_embeds.
Our model has both, so NPUW uses input_ids (zeros) -> garbage output.

Fix: Re-do IR surgery from decoder_stateful_ov, but this time:
1. Replace embedding Gather consumers with inputs_embeds Parameter
2. REMOVE input_ids Parameter entirely (don't keep it as dangling input)
3. Save as decoder_stateful_embeds (overwrite)

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/fix_decoder_remove_input_ids.py'
"""

import os
import sys
import shutil
import time

sys.path.insert(0, r"C:\Apps\Qwen3-ASR")

import numpy as np
import openvino as ov
from openvino import opset13 as opset
from openvino import PartialShape, Dimension

MODEL_DIR = r"C:\Apps\Qwen3-ASR\models"


def main():
    print("=" * 70)
    print("  Fix: Remove input_ids from decoder IR")
    print("=" * 70)

    core = ov.Core()

    # Load the ORIGINAL stateful model (before any surgery)
    src_path = os.path.join(MODEL_DIR, "decoder_stateful_ov", "openvino_model.xml")
    print(f"\nLoading original stateful model: {src_path}")
    model = core.read_model(src_path)

    # Show original inputs
    print("\nOriginal model inputs:")
    for inp in model.inputs:
        try:
            name = inp.any_name
        except RuntimeError:
            name = "<unnamed>"
        print(f"  {name:30s} shape={inp.partial_shape} type={inp.element_type}")

    # Step 1: Find input_ids Parameter and embedding Gather
    print("\nStep 1: Finding input_ids and embedding Gather...")
    input_ids_param = None
    for op in model.get_ordered_ops():
        if op.type_info.name == "Parameter":
            try:
                if op.output(0).any_name == "input_ids":
                    input_ids_param = op
            except RuntimeError:
                pass

    if input_ids_param is None:
        print("  ERROR: input_ids not found!")
        sys.exit(1)

    print(f"  Found input_ids: {input_ids_param.get_friendly_name()}")

    # Trace input_ids consumers to find the embedding Gather
    embedding_gather = None
    for target_input in input_ids_param.output(0).get_target_inputs():
        node = target_input.get_node()
        if node.type_info.name == "Gather":
            embedding_gather = node
        elif node.type_info.name == "Convert":
            for ti2 in node.output(0).get_target_inputs():
                n2 = ti2.get_node()
                if n2.type_info.name == "Gather":
                    embedding_gather = n2

    if embedding_gather is None:
        print("  ERROR: embedding Gather not found!")
        sys.exit(1)

    gather_output = embedding_gather.output(0)
    print(f"  Found Gather: {embedding_gather.get_friendly_name()}")
    print(f"  Gather output shape: {gather_output.partial_shape}")
    print(f"  Gather consumers: {len(gather_output.get_target_inputs())}")

    # Step 2: Create inputs_embeds Parameter
    print("\nStep 2: Creating inputs_embeds Parameter...")
    inputs_embeds_shape = PartialShape([Dimension(-1), Dimension(-1), Dimension(1024)])
    inputs_embeds_param = opset.parameter(
        inputs_embeds_shape,
        dtype=np.float32,
        name="inputs_embeds"
    )
    inputs_embeds_param.output(0).set_names({"inputs_embeds"})
    print(f"  Created: inputs_embeds shape={inputs_embeds_shape}")

    # Step 3: Redirect Gather consumers to inputs_embeds
    print("\nStep 3: Redirecting Gather consumers...")
    consumers = list(gather_output.get_target_inputs())
    for target_input in consumers:
        target_input.replace_source_output(inputs_embeds_param.output(0))
        node = target_input.get_node()
        print(f"  Redirected: {node.type_info.name} '{node.get_friendly_name()}'")

    # Step 4: Disconnect input_ids from ALL its consumers
    # The Gather still consumes input_ids (via Convert), making it reachable.
    # Replace all input_ids consumers with dummy constants to orphan it.
    print("\nStep 4: Disconnecting input_ids from all consumers...")
    input_ids_consumers = list(input_ids_param.output(0).get_target_inputs())
    print(f"  input_ids has {len(input_ids_consumers)} consumer(s)")
    for target_input in input_ids_consumers:
        node = target_input.get_node()
        print(f"  Disconnecting: {node.type_info.name} '{node.get_friendly_name()}'")
        dummy = opset.constant(np.array([[0]], dtype=np.int64))
        target_input.replace_source_output(dummy.output(0))
    print(f"  input_ids now has {len(list(input_ids_param.output(0).get_target_inputs()))} consumers")

    # Step 5: Add inputs_embeds and create new Model without input_ids
    model.add_parameters([inputs_embeds_param])

    print("\nStep 5: Creating new model without input_ids...")
    new_params = [p for p in model.get_parameters()
                  if p.output(0).any_name != "input_ids"]
    print(f"  New parameters: {[p.output(0).any_name for p in new_params]}")

    new_model = ov.Model(
        results=model.get_results(),
        sinks=list(model.get_sinks()),
        parameters=new_params,
        name="decoder_stateful_embeds"
    )
    model = new_model

    # Validate
    print("\nStep 6: Validating modified model...")
    model.validate_nodes_and_infer_types()
    print("  Validation passed!")

    # Show final inputs
    print("\nFinal model inputs:")
    for inp in model.inputs:
        try:
            name = inp.any_name
        except RuntimeError:
            name = "<unnamed>"
        print(f"  {name:30s} shape={inp.partial_shape} type={inp.element_type}")

    print(f"\nFinal model outputs:")
    for out in model.outputs:
        try:
            name = out.any_name
        except RuntimeError:
            name = "<unnamed>"
        print(f"  {name:30s} shape={out.partial_shape} type={out.element_type}")

    # Step 6: Save model
    save_dir = os.path.join(MODEL_DIR, "decoder_stateful_embeds")
    os.makedirs(save_dir, exist_ok=True)
    xml_path = os.path.join(save_dir, "openvino_model.xml")
    print(f"\nStep 6: Saving to {xml_path}...")
    ov.save_model(model, xml_path)
    print("  Saved!")

    # Copy config files from original
    src_dir = os.path.join(MODEL_DIR, "decoder_stateful_ov")
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt",
              "added_tokens.json", "generation_config.json"]:
        src = os.path.join(src_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(save_dir, f))
    print("  Config files copied")

    # Step 7: Verify on CPU
    print("\n" + "=" * 70)
    print("  Verification: CPU")
    print("=" * 70)

    compiled_cpu = core.compile_model(xml_path, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    request_cpu = compiled_cpu.create_infer_request()
    request_cpu.reset_state()

    seq_len = 10
    test_embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
    test_inputs = {
        "inputs_embeds": test_embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
        "beam_idx": np.array([0], dtype=np.int32),
    }

    request_cpu.infer(test_inputs)
    logits_cpu = request_cpu.get_output_tensor(0).data.copy()
    print(f"  CPU logits shape: {logits_cpu.shape}")
    print(f"  CPU logits[0,-1,:5]: {logits_cpu[0, -1, :5]}")

    # Step 8: Verify on NPU
    print("\n" + "=" * 70)
    print("  Verification: NPU (NPUW_LLM)")
    print("=" * 70)

    npu_config = {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 2,
        "NPUW_LLM_MAX_PROMPT_LEN": 256,
        "NPUW_LLM_MIN_RESPONSE_LEN": 64,
    }

    t0 = time.perf_counter()
    compiled_npu = core.compile_model(xml_path, "NPU", npu_config)
    compile_time = time.perf_counter() - t0
    print(f"  NPU compilation: {compile_time*1000:.0f}ms")

    request_npu = compiled_npu.create_infer_request()

    # Same test inputs (no input_ids needed!)
    request_npu.infer(test_inputs)
    logits_npu = request_npu.get_output_tensor(0).data.copy()
    print(f"  NPU logits shape: {logits_npu.shape}")
    print(f"  NPU logits[0,-1,:5]: {logits_npu[0, -1, :5]}")

    # Compare
    cpu_top = int(np.argmax(logits_cpu[0, -1, :]))
    npu_top = int(np.argmax(logits_npu[0, -1, :]))
    print(f"\n  CPU argmax: {cpu_top}")
    print(f"  NPU argmax: {npu_top}")
    print(f"  Match: {cpu_top == npu_top}")

    # Step 9: Real ASR test
    print("\n" + "=" * 70)
    print("  Real ASR test: encoder + decoder on NPU")
    print("=" * 70)

    embed_table = np.load(os.path.join(MODEL_DIR, "embed_tokens.npy"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B", trust_remote_code=True
    )

    # Build a simple prompt
    IM_START, IM_END, NEWLINE = 151644, 151645, 198
    AUDIO_START, AUDIO_END, AUDIO_PAD = 151669, 151670, 151676
    ASR_TEXT = 151704

    system_tokens = [IM_START] + tokenizer.encode("system\nYou are a helpful assistant.") + [IM_END, NEWLINE]
    user_tokens = [IM_START] + tokenizer.encode("user\n") + [AUDIO_START] + [AUDIO_PAD] * 104 + [AUDIO_END, IM_END, NEWLINE]
    assistant_tokens = [IM_START] + tokenizer.encode("assistant\nlanguage Chinese") + [ASR_TEXT]
    all_tokens = system_tokens + user_tokens + assistant_tokens
    prompt_len = len(all_tokens)

    # Build inputs_embeds with silence (zeros for audio)
    ids = np.array(all_tokens, dtype=np.int64)
    embeds = embed_table[ids]
    # Audio positions get zeros (silence)
    audio_positions = np.where(ids == AUDIO_PAD)[0]
    embeds[audio_positions] = 0.0  # silence

    prefill_inputs = {
        "inputs_embeds": embeds[np.newaxis, :, :].astype(np.float32),
        "attention_mask": np.ones((1, prompt_len), dtype=np.int64),
        "position_ids": np.arange(prompt_len, dtype=np.int64).reshape(1, -1),
        "beam_idx": np.array([0], dtype=np.int32),
    }

    # Reset and run on NPU
    request_npu2 = compiled_npu.create_infer_request()
    request_npu2.infer(prefill_inputs)
    logits = request_npu2.get_output_tensor(0).data.copy()
    first_token = int(np.argmax(logits[0, -1, :]))
    print(f"  First token (NPU, silence): {first_token} = '{tokenizer.decode([first_token])}'")

    # Also test on CPU for comparison
    request_cpu2 = compiled_cpu.create_infer_request()
    request_cpu2.reset_state()
    # CPU needs same inputs
    request_cpu2.infer(prefill_inputs)
    logits_cpu2 = request_cpu2.get_output_tensor(0).data.copy()
    first_token_cpu = int(np.argmax(logits_cpu2[0, -1, :]))
    print(f"  First token (CPU, silence): {first_token_cpu} = '{tokenizer.decode([first_token_cpu])}'")

    # Generate a few tokens on NPU
    print("\n  Generating 10 tokens on NPU...")
    generated = []
    past_len = prompt_len
    current_token = first_token
    for step in range(10):
        if current_token == IM_END:
            break
        generated.append(current_token)
        token_embed = embed_table[current_token][np.newaxis, np.newaxis, :].astype(np.float32)
        step_inputs = {
            "inputs_embeds": token_embed,
            "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
            "position_ids": np.array([[past_len]], dtype=np.int64),
            "beam_idx": np.array([0], dtype=np.int32),
        }
        request_npu2.infer(step_inputs)
        logits = request_npu2.get_output_tensor(0).data.copy()
        current_token = int(np.argmax(logits[0, -1, :]))
        past_len += 1

    text = tokenizer.decode(generated, skip_special_tokens=False)
    print(f"  NPU generated: {generated}")
    print(f"  NPU text: '{text}'")
    is_garbage = all(t == generated[0] for t in generated) if generated else True
    print(f"  Is garbage (all same token): {is_garbage}")

    print("\n" + "=" * 70)
    if not is_garbage:
        print("  SUCCESS: NPU decoder produces non-garbage output!")
    else:
        print("  STILL BROKEN: NPU decoder still produces garbage")
    print("=" * 70)


if __name__ == "__main__":
    main()
