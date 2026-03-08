"""Debug test for the ASR engine - check token generation on silence."""
import sys
import os
import time
sys.path.insert(0, r"C:\Apps\Qwen3-ASR")

import numpy as np
from qwen3_asr_app.inference.engine import ASREngine

# Test 1: Without language forcing
print("=== Test 1: No language forcing (silence) ===")
engine = ASREngine(encoder_device="NPU", language=None)
state = engine.new_session()

# Show prompt tokens
tokens = engine._build_prompt_tokens("")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(r"C:\Apps\Qwen3-ASR\Qwen3-ASR-0.6B", trust_remote_code=True)
print(f"Prompt tokens ({len(tokens)}):")
print(f"  Last 10 tokens: {tokens[-10:]}")
print(f"  Last 10 decoded: {[tok.decode([t]) for t in tokens[-10:]]}")

chunk = np.zeros(16000 * 2, dtype=np.float32)
engine.feed(chunk, state)
print(f"  _raw_decoded: '{state._raw_decoded}'")
print(f"  text: '{state.text}'")
print(f"  lang: '{state.language}'")

# Test 2: With Chinese language forcing
print("\n=== Test 2: language='Chinese' (silence) ===")
engine2 = ASREngine(encoder_device="NPU", language="Chinese")
state2 = engine2.new_session()

tokens2 = engine2._build_prompt_tokens("")
print(f"Prompt tokens ({len(tokens2)}):")
print(f"  Last 10 tokens: {tokens2[-10:]}")
print(f"  Last 10 decoded: {[tok.decode([t]) for t in tokens2[-10:]]}")

# Also check what assistant tokens look like
print(f"  Assistant tokens: {engine2._assistant_tokens}")
print(f"  Assistant decoded: {tok.decode(engine2._assistant_tokens)}")

chunk2 = np.zeros(16000 * 2, dtype=np.float32)
engine2.feed(chunk2, state2)
print(f"  _raw_decoded: '{state2._raw_decoded}'")
print(f"  text: '{state2.text}'")
print(f"  lang: '{state2.language}'")

# Test 3: Check what _raw_decoded contains token by token
print("\n=== Test 3: Detailed token generation (no language, silence) ===")
engine3 = ASREngine(encoder_device="NPU", language=None)

# Manually run the pipeline steps
mel = engine3._mel(np.zeros(16000 * 2, dtype=np.float32))
audio_feats = engine3._encoder(mel)
tokens3 = engine3._build_prompt_tokens("")
embeds = engine3._build_inputs_embeds(tokens3, audio_feats)
print(f"inputs_embeds shape: {embeds.shape}")

# Generate
generated = engine3._decoder.generate(embeds, max_new_tokens=20)
print(f"Generated IDs: {generated}")
print(f"Generated text: '{tok.decode(generated, skip_special_tokens=False)}'")
print(f"Generated per-token: {[(tid, tok.decode([tid])) for tid in generated]}")
