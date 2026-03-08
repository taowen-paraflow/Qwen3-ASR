"""Quick import and smoke test for the qwen3_asr_app package."""
import sys
import os
import time

sys.path.insert(0, r"C:\Apps\Qwen3-ASR")

print("=== Import Tests ===")

print("1. Config...")
from qwen3_asr_app.config import ENCODER_XML, DECODER_XML, EMBED_TABLE_NPY
print(f"   ENCODER_XML exists: {os.path.exists(ENCODER_XML)}")
print(f"   DECODER_XML exists: {os.path.exists(DECODER_XML)}")
print(f"   EMBED_TABLE exists: {os.path.exists(EMBED_TABLE_NPY)}")

print("2. MelProcessor...")
from qwen3_asr_app.audio.processor import MelProcessor
import numpy as np
mel_proc = MelProcessor()
silence = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds
mel = mel_proc(silence)
print(f"   Mel shape: {mel.shape} (expect [1, 128, 800])")

print("3. OVEncoder...")
from qwen3_asr_app.inference.ov_encoder import OVEncoder
t0 = time.perf_counter()
encoder = OVEncoder(device="NPU")
print(f"   Encoder loaded in {(time.perf_counter()-t0)*1000:.0f}ms")
audio_feats = encoder(mel)
print(f"   Audio features: {audio_feats.shape} (expect [1, 104, 1024])")

print("4. OVDecoder...")
from qwen3_asr_app.inference.ov_decoder import OVDecoder
t0 = time.perf_counter()
decoder = OVDecoder()
print(f"   Decoder loaded in {(time.perf_counter()-t0)*1000:.0f}ms")
print(f"   Embed table: {decoder.embed_table.shape}")

print("5. ASREngine (full pipeline with silence)...")
from qwen3_asr_app.inference.engine import ASREngine
engine = ASREngine(encoder_device="NPU", language="Chinese")
state = engine.new_session()

# Simulate streaming: feed 3x 2-second chunks of silence
for i in range(3):
    chunk = np.zeros(16000 * 2, dtype=np.float32)
    t0 = time.perf_counter()
    engine.feed(chunk, state)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"   Chunk {i}: text='{state.text}', lang='{state.language}', time={elapsed:.0f}ms")

engine.finish(state)
print(f"   Final: text='{state.text}', lang='{state.language}'")

print("\n=== All tests passed! ===")
