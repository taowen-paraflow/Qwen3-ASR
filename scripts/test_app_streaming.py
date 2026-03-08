"""Test the streaming ASR engine with synthetic audio (no UI needed).

Creates a synthetic audio signal and feeds it through the streaming engine
chunk by chunk to validate the full pipeline.

Usage:
    uv run python scripts/test_app_streaming.py [path/to/audio.wav]
"""
import sys
import os
import time
sys.path.insert(0, r"C:\Apps\Qwen3-ASR")

import numpy as np
from qwen3_asr_app.inference.engine import ASREngine


def generate_test_audio(duration_sec=5.0, sr=16000):
    """Generate a synthetic audio signal (sine wave + noise)."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    # Mix of frequencies to simulate speech-like signal
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t)).astype(np.float32)
    return audio


def load_audio(path, sr=16000):
    """Load a real audio file."""
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  Streaming ASR Engine Test")
    print("=" * 60)

    # Load or generate audio
    if audio_path:
        print(f"Loading audio: {audio_path}")
        audio = load_audio(audio_path)
    else:
        print("No audio file provided, using 5s synthetic audio")
        audio = generate_test_audio(5.0)

    duration = len(audio) / 16000
    print(f"Audio: {duration:.1f}s, {len(audio)} samples")
    print()

    # Initialize engine
    print("Initializing ASR engine...")
    t0 = time.perf_counter()
    engine = ASREngine(encoder_device="NPU", language=None)
    init_time = time.perf_counter() - t0
    print(f"  Engine initialized in {init_time:.1f}s")
    print()

    # Simulate streaming: feed 2-second chunks
    print("-" * 60)
    print("  Streaming simulation (2s chunks)")
    print("-" * 60)

    state = engine.new_session()
    chunk_size = 16000 * 2  # 2 seconds
    offset = 0
    total_infer_time = 0

    while offset < len(audio):
        end = min(offset + chunk_size, len(audio))
        chunk = audio[offset:end]
        chunk_dur = len(chunk) / 16000

        t0 = time.perf_counter()

        if end == len(audio):
            # Last chunk - use finish to flush
            # First feed what we have, then finish
            engine.feed(chunk, state)
            engine.finish(state)
        else:
            engine.feed(chunk, state)

        elapsed = (time.perf_counter() - t0) * 1000
        total_infer_time += elapsed

        print(f"  [{offset/16000:.1f}s - {end/16000:.1f}s] "
              f"chunk={chunk_dur:.1f}s, infer={elapsed:.0f}ms, "
              f"lang='{state.language}', text='{state.text}'")

        offset = end

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Audio duration: {duration:.1f}s")
    print(f"  Total inference: {total_infer_time:.0f}ms")
    print(f"  Real-time factor: {total_infer_time / (duration * 1000):.2f}x")
    print(f"  Language: '{state.language}'")
    print(f"  Text: '{state.text}'")
    print(f"  Raw decoded: '{state._raw_decoded}'")
    print()

    if total_infer_time < duration * 1000:
        print("  PASS: Inference is faster than real-time!")
    else:
        print(f"  WARNING: Inference ({total_infer_time:.0f}ms) slower than audio ({duration*1000:.0f}ms)")


if __name__ == "__main__":
    main()
