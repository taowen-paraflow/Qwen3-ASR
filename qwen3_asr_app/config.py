"""Configuration constants for Qwen3-ASR desktop app."""

import os

# Paths
PROJECT_ROOT = r"C:\Apps\Qwen3-ASR"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
HF_MODEL_DIR = os.path.join(PROJECT_ROOT, "Qwen3-ASR-0.6B")

# Audio
SAMPLE_RATE = 16000
MEL_T_FIXED = 800  # Fixed mel frames (5 seconds of audio)
AUDIO_PAD_COUNT = 104  # Encoder output token count

# Special token IDs (Qwen3-ASR ChatML)
IM_START = 151644
IM_END = 151645
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676
ASR_TEXT = 151704
NEWLINE = 198

# Streaming defaults
CHUNK_SIZE_SEC = 2.0
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
MAX_NEW_TOKENS = 32

# Model file paths
ENCODER_XML = os.path.join(MODEL_DIR, "encoder_fp16.xml")
DECODER_DIR = os.path.join(MODEL_DIR, "decoder_stateful_embeds")
DECODER_XML = os.path.join(DECODER_DIR, "openvino_model.xml")
EMBED_TABLE_NPY = os.path.join(MODEL_DIR, "embed_tokens.npy")

# NPU decoder compilation config (NPUW_LLM)
NPU_DECODER_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 256,
    "NPUW_LLM_MIN_RESPONSE_LEN": 64,
}
