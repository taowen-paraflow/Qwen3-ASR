"""Export Qwen3-ASR Audio Encoder to OpenVINO IR (static shape).

Wraps the audio_tower in a static-shape module suitable for NPU export.
Fixed input:  mel [1, 128, 800]
Fixed output: encoder_out [1, 104, 1024]

Shape trace (T_fixed=800, n_window=50):
  mel [1, 128, 800]
    -> squeeze + reshape  -> chunks [8, 1, 128, 100]
    -> conv2d1 (gelu)     -> [8, 480, 64, 50]
    -> conv2d2 (gelu)     -> [8, 480, 32, 25]
    -> conv2d3 (gelu)     -> [8, 480, 16, 13]
    -> permute + flatten  -> [8, 13, 7680]
    -> conv_out           -> [8, 13, 896]
    -> + pos_embed[:13]   -> [8, 13, 896]
    -> reshape            -> [104, 896]   (2D, NO batch dim for transformer)
    -> 18x transformer    -> [104, 896]   (cu_seqlens = [0, 104])
    -> ln_post            -> [104, 896]
    -> proj1 -> act -> proj2 -> [104, 1024]
    -> unsqueeze(0)       -> [1, 104, 1024]

Usage:
    powershell.exe -Command 'cd C:\\Apps\\Qwen3-ASR; uv run python scripts/export_encoder.py'
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root so transformers can find the custom model code
sys.path.insert(0, r"C:\Apps\Qwen3-ASR")


# ---------------------------------------------------------------------------
# Static wrapper
# ---------------------------------------------------------------------------

class StaticAudioEncoder(nn.Module):
    """Static-shape wrapper around Qwen3ASRAudioEncoder for OpenVINO export.

    All dynamic operations (chunking by feature_lens, padding, cu_seqlens
    construction) are replaced with fixed constants for T_fixed=800.

    Submodules are NOT copied -- they are referenced from the original
    audio_tower so that weights are shared (no extra memory).
    """

    # Constants for T_fixed = 800, n_window = 50
    N_CHUNKS = 8          # 800 / (n_window * 2) = 800 / 100
    CHUNK_LEN = 100       # n_window * 2
    T_AFTER_CNN = 13      # per-chunk time after 3x stride-2 conv: 100 -> 50 -> 25 -> 13
    FREQ_AFTER_CNN = 16   # freq dim after 3x stride-2 conv:     128 -> 64 -> 32 -> 16
    SEQ_LEN = 104         # N_CHUNKS * T_AFTER_CNN = 8 * 13
    D_MODEL = 896
    DOWNSAMPLE_HIDDEN = 480

    def __init__(self, audio_tower: nn.Module):
        super().__init__()

        # --- CNN layers ---
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out

        # --- Transformer layers (18 layers) ---
        self.layers = audio_tower.layers

        # --- Post-processing ---
        self.ln_post = audio_tower.ln_post
        self.proj1 = audio_tower.proj1
        self.act = audio_tower.act
        self.proj2 = audio_tower.proj2

        # --- Fixed position embedding: only the first T_AFTER_CNN=13 rows ---
        pos_embed = audio_tower.positional_embedding.positional_embedding[
            : self.T_AFTER_CNN, :
        ].clone()
        self.register_buffer("pos_embed", pos_embed)  # [13, 896]

        # --- Fixed cu_seqlens for a single attention window of 104 tokens ---
        cu_seqlens = torch.tensor([0, self.SEQ_LEN], dtype=torch.int32)
        self.register_buffer("cu_seqlens", cu_seqlens)  # [0, 104]

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel spectrogram, shape [1, 128, 800].

        Returns:
            Encoder output, shape [1, 104, 1024].
        """
        # Reshape mel into 8 fixed chunks of 100 frames
        # Use [0] indexing instead of squeeze to avoid dynamic Squeeze op
        x = mel[0].T.reshape(self.N_CHUNKS, self.CHUNK_LEN, 128)
        x = x.transpose(1, 2).unsqueeze(1)                         # [8, 1, 128, 100]

        # 3-layer CNN with GELU (8x downsampling)
        x = F.gelu(self.conv2d1(x))                                # [8, 480, 64, 50]
        x = F.gelu(self.conv2d2(x))                                # [8, 480, 32, 25]
        x = F.gelu(self.conv2d3(x))                                # [8, 480, 16, 13]

        # Flatten freq dim and project to d_model
        # Use literal constants instead of x.size() to avoid dynamic shapes
        x = x.permute(0, 3, 1, 2).contiguous().view(
            self.N_CHUNKS, self.T_AFTER_CNN, self.DOWNSAMPLE_HIDDEN * self.FREQ_AFTER_CNN
        )                                                           # [8, 13, 7680]
        x = self.conv_out(x)                                       # [8, 13, 896]

        # Sinusoidal position embedding
        x = x + self.pos_embed.unsqueeze(0)                        # [8, 13, 896]

        # Flatten to single sequence (2D for transformer layers)
        x = x.reshape(self.SEQ_LEN, self.D_MODEL)                 # [104, 896]

        # 18 transformer encoder layers
        for layer in self.layers:
            x = layer(x, self.cu_seqlens)[0]

        # Post-processing
        x = self.ln_post(x)
        x = self.proj2(self.act(self.proj1(x)))                    # [104, 1024]

        return x.unsqueeze(0)                                      # [1, 104, 1024]


# ---------------------------------------------------------------------------
# Main export script
# ---------------------------------------------------------------------------

def main():
    model_path = r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B"
    output_dir = r"C:\Apps\Qwen3-ASR\models"
    output_path = os.path.join(output_dir, "encoder_fp16.xml")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load the full Qwen3-ASR model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading Qwen3-ASR model from:", model_path)
    print("=" * 60)

    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)

    model = AutoModel.from_pretrained(model_path)
    model.eval()
    print("Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # 2. Extract audio encoder and force eager attention
    # ------------------------------------------------------------------
    audio_tower = model.thinker.audio_tower

    # Eager attention is required for torch tracing (no flash_attention_2)
    audio_tower.config._attn_implementation = "eager"
    for layer in audio_tower.layers:
        layer.self_attn.config._attn_implementation = "eager"

    print(f"Audio encoder: {len(audio_tower.layers)} transformer layers")
    print(f"Attention implementation forced to: eager\n")

    # ------------------------------------------------------------------
    # 3. Create static wrapper
    # ------------------------------------------------------------------
    print("Creating StaticAudioEncoder wrapper...")
    wrapper = StaticAudioEncoder(audio_tower)
    wrapper.eval()
    print(f"  pos_embed buffer:  {wrapper.pos_embed.shape}")
    print(f"  cu_seqlens buffer: {wrapper.cu_seqlens}")
    print()

    # ------------------------------------------------------------------
    # 4. Verify with PyTorch forward pass
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PyTorch verification (dummy input)")
    print("=" * 60)

    dummy = torch.randn(1, 128, 800)
    print(f"  Input shape:        {dummy.shape}\n")

    with torch.no_grad():
        out = wrapper(dummy)

    print(f"\nOutput shape: {out.shape}")
    assert out.shape == (1, 104, 1024), (
        f"Shape mismatch! Expected [1, 104, 1024], got {list(out.shape)}"
    )
    print("PyTorch forward pass: OK\n")

    # ------------------------------------------------------------------
    # 5. Export to OpenVINO IR
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Exporting to OpenVINO IR")
    print("=" * 60)

    import openvino as ov

    print("OpenVINO version:", ov.__version__)
    print("Converting model (this may take a minute)...")

    ov_model = ov.convert_model(
        wrapper,
        example_input=dummy,
        input=ov.PartialShape([1, 128, 800]),
    )
    ov.save_model(ov_model, output_path, compress_to_fp16=True)

    print()
    print("Encoder exported successfully!")
    print(f"  Path:   {output_path}")
    print(f"  Input:  mel [1, 128, 800]")
    print(f"  Output: encoder_out [1, 104, 1024]")


if __name__ == "__main__":
    main()
