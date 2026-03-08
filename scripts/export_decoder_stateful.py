"""Export Qwen3-ASR Text Decoder as a stateful OpenVINO model (KV-cache).

Strategy:
  1. Load full Qwen3ASR model
  2. Extract the thinker's state_dict (excluding audio_tower keys)
  3. Create a matching Qwen3Config
  4. Instantiate Qwen3ForCausalLM and load the weights
  5. Save as a standalone HF model
  6. Export via optimum-intel OVModelForCausalLM with stateful=True

The thinker's weight names (model.layers.X.self_attn.q_proj.weight,
lm_head.weight, model.embed_tokens.weight) are identical to standard
Qwen3ForCausalLM, so this repackaging works directly.

Key detail: tie_word_embeddings=True means lm_head.weight is shared with
model.embed_tokens.weight (no separate lm_head.weight in the state_dict).

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/export_decoder_stateful.py'
"""

import os
import sys
import traceback

sys.path.insert(0, r"C:\Apps\Qwen3-ASR")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch


# ============================================================================
# Step 1: Load full Qwen3-ASR model
# ============================================================================
def step1_load_model():
    print("=" * 60)
    print("Step 1: Loading Qwen3-ASR model")
    print("=" * 60)

    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRConfig,
    )
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)

    model_path = r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B"
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print(f"  Model loaded from: {model_path}")
    print()
    return model


# ============================================================================
# Step 2: Extract thinker state_dict (decoder only, no audio_tower)
# ============================================================================
def step2_extract_decoder_weights(model):
    print("=" * 60)
    print("Step 2: Extracting thinker decoder weights")
    print("=" * 60)

    thinker = model.thinker
    thinker_sd = thinker.state_dict()
    print(f"  Total thinker params: {len(thinker_sd)}")

    # Show first few keys to verify naming
    for k in list(thinker_sd.keys())[:5]:
        print(f"    {k}: {thinker_sd[k].shape}")

    # Remove audio_tower keys -- they belong to the encoder, not the decoder
    decoder_sd = {
        k: v for k, v in thinker_sd.items() if not k.startswith("audio_tower.")
    }
    print(f"  Decoder params (no audio_tower): {len(decoder_sd)}")

    # Show which key prefixes remain
    prefixes = set()
    for k in decoder_sd:
        prefixes.add(k.split(".")[0])
    print(f"  Key prefixes: {sorted(prefixes)}")

    # Check for lm_head.weight
    has_lm_head = "lm_head.weight" in decoder_sd
    print(f"  lm_head.weight present: {has_lm_head}")
    if has_lm_head:
        print(f"    lm_head.weight shape: {decoder_sd['lm_head.weight'].shape}")
    print()
    return decoder_sd


# ============================================================================
# Step 3: Create Qwen3Config matching the text_config
# ============================================================================
def step3_create_config():
    print("=" * 60)
    print("Step 3: Creating Qwen3Config")
    print("=" * 60)

    from transformers import Qwen3Config

    # Values from Qwen3-ASR-0.6B/config.json -> thinker_config -> text_config
    qwen3_config = Qwen3Config(
        hidden_size=1024,
        intermediate_size=3072,
        max_position_embeddings=65536,
        num_attention_heads=16,
        num_hidden_layers=28,
        num_key_value_heads=8,
        rms_norm_eps=1e-06,
        rope_scaling={
            "interleaved": True,
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
            "type": "default",
        },
        rope_theta=1000000.0,
        sliding_window=None,
        tie_word_embeddings=True,
        vocab_size=151936,
        head_dim=128,
        attention_dropout=0.0,
        attention_bias=False,
        hidden_act="silu",
        initializer_range=0.02,
        use_cache=True,
    )
    print(f"  model_type: {qwen3_config.model_type}")
    print(f"  hidden_size: {qwen3_config.hidden_size}")
    print(f"  num_hidden_layers: {qwen3_config.num_hidden_layers}")
    print(f"  num_attention_heads: {qwen3_config.num_attention_heads}")
    print(f"  num_key_value_heads: {qwen3_config.num_key_value_heads}")
    print(f"  tie_word_embeddings: {qwen3_config.tie_word_embeddings}")
    print(f"  rope_scaling: {qwen3_config.rope_scaling}")
    print()
    return qwen3_config


# ============================================================================
# Step 4: Create standalone Qwen3ForCausalLM and load weights
# ============================================================================
def step4_create_standalone(qwen3_config, decoder_sd):
    print("=" * 60)
    print("Step 4: Creating Qwen3ForCausalLM and loading weights")
    print("=" * 60)

    from transformers import Qwen3ForCausalLM

    standalone = Qwen3ForCausalLM(qwen3_config)
    standalone.eval()

    # Load weights (strict=False because tie_word_embeddings may cause
    # lm_head.weight to be absent or to appear as unexpected)
    missing, unexpected = standalone.load_state_dict(decoder_sd, strict=False)
    print(f"  Missing keys ({len(missing)}):")
    for k in missing:
        print(f"    {k}")
    print(f"  Unexpected keys ({len(unexpected)}):")
    for k in unexpected:
        print(f"    {k}")

    # Verify with a quick forward pass
    print("  Verifying forward pass...")
    with torch.no_grad():
        dummy_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        output = standalone(input_ids=dummy_ids)
        print(f"  Output logits shape: {output.logits.shape}")
        assert output.logits.shape == (1, 5, 151936), (
            f"Expected [1, 5, 151936], got {list(output.logits.shape)}"
        )
    print("  Forward pass OK")
    print()
    return standalone


# ============================================================================
# Step 5: Save standalone model + tokenizer
# ============================================================================
def step5_save_standalone(standalone, qwen3_config):
    print("=" * 60)
    print("Step 5: Saving standalone Qwen3ForCausalLM")
    print("=" * 60)

    save_dir = r"C:\Apps\Qwen3-ASR\models\qwen3_decoder_standalone"
    os.makedirs(save_dir, exist_ok=True)

    standalone.save_pretrained(save_dir)
    print(f"  Model saved to: {save_dir}")

    # Copy the tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B",
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(save_dir)
    print(f"  Tokenizer saved to: {save_dir}")
    print()
    return save_dir


# ============================================================================
# Step 6: Export via optimum-intel with stateful=True
# ============================================================================
def step6_export_openvino(save_dir):
    print("=" * 60)
    print("Step 6: Exporting with optimum-intel (stateful=True)")
    print("=" * 60)

    from optimum.intel import OVModelForCausalLM

    export_dir = r"C:\Apps\Qwen3-ASR\models\decoder_stateful_ov"

    print(f"  Source: {save_dir}")
    print(f"  Target: {export_dir}")
    print("  This may take several minutes...")
    print()

    ov_model = OVModelForCausalLM.from_pretrained(
        save_dir,
        export=True,
        stateful=True,
        trust_remote_code=False,  # Standard Qwen3, no custom code
    )
    ov_model.save_pretrained(export_dir)
    print(f"  Exported to: {export_dir}")

    # Copy tokenizer files to export dir (optimum-intel may not copy vocab)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    tokenizer.save_pretrained(export_dir)
    print("  Tokenizer copied to export dir")

    # List exported files
    for f in sorted(os.listdir(export_dir)):
        fpath = os.path.join(export_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    {f}: {size_mb:.1f} MB")
    print()
    return export_dir


# ============================================================================
# Step 7: Verify exported model on CPU
# ============================================================================
def step7_verify_cpu(export_dir):
    print("=" * 60)
    print("Step 7: Verifying exported model on CPU")
    print("=" * 60)

    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    ov_model = OVModelForCausalLM.from_pretrained(
        export_dir,
        device="CPU",
        ov_config={"PERFORMANCE_HINT": "LATENCY"},
    )
    tokenizer = AutoTokenizer.from_pretrained(export_dir)

    test_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    output = ov_model.generate(
        input_ids=test_ids,
        max_new_tokens=5,
        do_sample=False,
    )
    print(f"  Input IDs:    {test_ids.tolist()}")
    print(f"  Output IDs:   {output.tolist()}")
    print(f"  Decoded text: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    print("  CPU verification OK")
    print()


# ============================================================================
# Step 8: Try NPU compilation
# ============================================================================
def step8_try_npu(export_dir):
    print("=" * 60)
    print("Step 8: Trying NPU compilation")
    print("=" * 60)

    from optimum.intel import OVModelForCausalLM

    cache_dir = r"C:\Apps\Qwen3-ASR\models\cache"
    os.makedirs(cache_dir, exist_ok=True)

    try:
        ov_model_npu = OVModelForCausalLM.from_pretrained(
            export_dir,
            device="NPU",
            ov_config={
                "PERFORMANCE_HINT": "LATENCY",
                "CACHE_DIR": cache_dir,
            },
        )
        test_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        output = ov_model_npu.generate(
            input_ids=test_ids,
            max_new_tokens=5,
            do_sample=False,
        )
        print(f"  NPU output IDs: {output.tolist()}")
        print("  NPU WORKS!")
    except Exception as e:
        print(f"  NPU failed: {type(e).__name__}: {e}")
        print("  Will need manual tuning for NPU compatibility")
    print()


# ============================================================================
# Main
# ============================================================================
def main():
    print()
    print("=" * 60)
    print("  Qwen3-ASR Stateful Decoder Export")
    print("  (Qwen3ForCausalLM + KV-cache via optimum-intel)")
    print("=" * 60)
    print()

    # Step 1: Load model
    model = step1_load_model()

    # Step 2: Extract decoder weights
    decoder_sd = step2_extract_decoder_weights(model)

    # Step 3: Create config
    qwen3_config = step3_create_config()

    # Step 4: Create standalone model
    standalone = step4_create_standalone(qwen3_config, decoder_sd)

    # Free original model memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 5: Save standalone
    save_dir = step5_save_standalone(standalone, qwen3_config)

    # Free standalone model memory before export
    del standalone
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 6: Export to OpenVINO
    try:
        export_dir = step6_export_openvino(save_dir)
    except Exception as e:
        print(f"  Export FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print()
        print("Stopping here -- fix the export error before continuing.")
        return

    # Step 7: Verify on CPU
    try:
        step7_verify_cpu(export_dir)
    except Exception as e:
        print(f"  CPU verification FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print()

    # Step 8: Try NPU
    try:
        step8_try_npu(export_dir)
    except Exception as e:
        print(f"  NPU step FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print()

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
