"""Export Qwen3-ASR Text Decoder to OpenVINO IR (static shape, no KV-cache).

Wraps the text decoder (Qwen3 LLM backbone + lm_head) in a static-shape
module suitable for NPU export.  KV-cache is disabled for this first version;
the model processes the full sequence in a single prefill pass.

Fixed input:  inputs_embeds  [1, 256, 1024]
              position_ids   [3, 1, 256]
              attention_mask  [1, 256]
Fixed output: logits         [1, 256, 151936]

Architecture (per the Qwen3ASRThinkerTextModel):
  - 28 transformer decoder layers (GQA: 16 heads, 8 KV heads, head_dim=128)
  - mRoPE with sections [24, 20, 20] (sum=64 = head_dim/2)
  - RMS norm (eps=1e-6)
  - SiLU-gated MLP (hidden=1024, intermediate=3072)
  - LM head: Linear(1024, 151936)

Usage:
    powershell.exe -Command '$env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/export_decoder.py'
"""

import os
import sys

# Add project root so transformers can find the custom model code
sys.path.insert(0, r"C:\Apps\Qwen3-ASR")

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Static wrapper
# ---------------------------------------------------------------------------

class StaticDecoder(nn.Module):
    """Static-shape wrapper around the Qwen3-ASR text decoder for OpenVINO export.

    Manually implements the forward pass to bypass non-traceable decorators
    (@check_model_inputs, @dynamic_rope_update) and the create_causal_mask
    utility.  The causal attention mask is built inline using standard torch
    operations that are friendly to torch.fx / OpenVINO tracing.

    Submodules are NOT copied -- they are referenced from the original model
    so that weights are shared (no extra memory).
    """

    def __init__(self, text_model, lm_head):
        super().__init__()
        self.layers = text_model.layers          # 28 x Qwen3ASRThinkerTextDecoderLayer
        self.norm = text_model.norm              # Qwen3ASRTextRMSNorm
        self.rotary_emb = text_model.rotary_emb  # Qwen3ASRThinkerTextRotaryEmbedding (mRoPE)
        self.lm_head = lm_head                   # nn.Linear(1024, 151936)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds:  [1, seq_len, 1024] -- merged audio + text embeddings
            position_ids:   [3, 1, seq_len]    -- mRoPE indices (temporal, height, width)
            attention_mask:  [1, seq_len]       -- 1=attend, 0=pad

        Returns:
            logits: [1, seq_len, 151936]
        """
        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype

        # ----- Build 4D causal attention mask [1, 1, seq_len, seq_len] -----
        # Upper triangle = large negative (future tokens masked),
        # lower triangle + diagonal = 0 (attend).
        mask_value = torch.finfo(dtype).min
        causal_mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                mask_value,
                dtype=dtype,
                device=inputs_embeds.device,
            ),
            diagonal=1,
        )
        # Apply padding mask: where attention_mask==0, mask those key columns
        padding_mask = (
            (1.0 - attention_mask.to(dtype))[:, None, None, :] * mask_value
        )
        causal_mask = causal_mask[None, None, :, :] + padding_mask
        # Final shape: [1, 1, seq_len, seq_len]

        # ----- Rotary position embeddings (mRoPE) -----
        # text_position_ids [1, seq_len] is used by the decoder layers
        # for logging / position tracking; the actual rotation is done via
        # position_embeddings (cos, sin) computed by the rotary_emb module.
        text_position_ids = position_ids[0]  # [1, seq_len]
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # ----- 28 decoder layers -----
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                cache_position=None,
                position_embeddings=position_embeddings,
            )

        # ----- Final RMS norm -----
        hidden_states = self.norm(hidden_states)

        # ----- LM head (project to vocabulary) -----
        logits = self.lm_head(hidden_states)

        return logits


# ---------------------------------------------------------------------------
# Main export script
# ---------------------------------------------------------------------------

def main():
    model_path = r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B"
    output_dir = r"C:\Apps\Qwen3-ASR\models"
    os.makedirs(output_dir, exist_ok=True)

    SEQ_LEN = 256

    # ------------------------------------------------------------------
    # 1. Load the full Qwen3-ASR model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading Qwen3-ASR model from:", model_path)
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

    model = AutoModel.from_pretrained(model_path)
    model.eval()
    print("Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # 2. Extract decoder components and force eager attention
    # ------------------------------------------------------------------
    text_model = model.thinker.model   # Qwen3ASRThinkerTextModel
    lm_head = model.thinker.lm_head   # nn.Linear(1024, 151936)

    print(f"Text decoder: {len(text_model.layers)} transformer layers")
    print(f"  Hidden size:    {text_model.config.hidden_size}")
    print(f"  Num heads:      {text_model.config.num_attention_heads}")
    print(f"  Num KV heads:   {text_model.config.num_key_value_heads}")
    print(f"  Head dim:       {getattr(text_model.config, 'head_dim', 128)}")
    print(f"  LM head:        Linear({lm_head.in_features}, {lm_head.out_features})")

    # Force eager attention -- SDPA/flash_attention_2 are not traceable
    text_model.config._attn_implementation = "eager"
    for layer in text_model.layers:
        layer.self_attn.config._attn_implementation = "eager"

    print(f"  Attention forced to: eager\n")

    # ------------------------------------------------------------------
    # 3. Create static wrapper
    # ------------------------------------------------------------------
    print("Creating StaticDecoder wrapper...")
    wrapper = StaticDecoder(text_model, lm_head)
    wrapper.eval()
    print("  Wrapper created.\n")

    # ------------------------------------------------------------------
    # 4. Verify with PyTorch forward pass
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PyTorch verification (dummy input)")
    print("=" * 60)

    # All 3 mRoPE channels use the same sequential positions for plain text
    dummy_embeds = torch.randn(1, SEQ_LEN, 1024)
    dummy_pos_ids = (
        torch.arange(SEQ_LEN)
        .view(1, 1, SEQ_LEN)
        .expand(3, 1, SEQ_LEN)
        .contiguous()
    )
    dummy_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

    print(f"  inputs_embeds:  {dummy_embeds.shape}")
    print(f"  position_ids:   {dummy_pos_ids.shape}")
    print(f"  attention_mask:  {dummy_mask.shape}")
    print()

    with torch.no_grad():
        logits = wrapper(dummy_embeds, dummy_pos_ids, dummy_mask)

    print(f"  Output logits:  {logits.shape}")
    assert logits.shape == (1, SEQ_LEN, 151936), (
        f"Shape mismatch! Expected [1, {SEQ_LEN}, 151936], got {list(logits.shape)}"
    )
    print("  PyTorch forward pass: OK\n")

    # ------------------------------------------------------------------
    # 5. Export to OpenVINO IR
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Exporting to OpenVINO IR")
    print("=" * 60)

    import openvino as ov

    print(f"  OpenVINO version: {ov.__version__}")
    print(f"  SEQ_LEN = {SEQ_LEN}")
    print("  Converting model (this may take several minutes)...")
    print()

    ov_model = ov.convert_model(
        wrapper,
        example_input=(dummy_embeds, dummy_pos_ids, dummy_mask),
        input=[
            ov.PartialShape([1, SEQ_LEN, 1024]),   # inputs_embeds
            ov.PartialShape([3, 1, SEQ_LEN]),       # position_ids
            ov.PartialShape([1, SEQ_LEN]),           # attention_mask
        ],
    )

    output_path = os.path.join(output_dir, "decoder_fp16.xml")
    ov.save_model(ov_model, output_path, compress_to_fp16=True)

    print("Decoder exported successfully!")
    print(f"  Path:   {output_path}")
    print(f"  Input:  inputs_embeds  [1, {SEQ_LEN}, 1024]")
    print(f"  Input:  position_ids   [3, 1, {SEQ_LEN}]")
    print(f"  Input:  attention_mask  [1, {SEQ_LEN}]")
    print(f"  Output: logits         [1, {SEQ_LEN}, 151936]")

    # ------------------------------------------------------------------
    # 6. Save embedding table for token -> embedding lookup
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Saving embedding table")
    print("=" * 60)

    import numpy as np

    embed_weights = text_model.embed_tokens.weight.detach().float().numpy()
    embed_path = os.path.join(output_dir, "embed_tokens.npy")
    np.save(embed_path, embed_weights)

    print(f"  Shape: {embed_weights.shape}")       # [151936, 1024]
    print(f"  Path:  {embed_path}")
    print(f"  Size:  {embed_weights.nbytes / 1024 / 1024:.1f} MB")

    print()
    print("=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
