"""End-to-end ASR test with real speech WAV files.

Tests both CPU and NPU decoder paths with the official Qwen3-ASR test audio.
Expected outputs:
  - asr_zh.wav: "甚至出现交易几乎停滞的情况。"
  - asr_en.wav: Long English speech (~67s)

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python scripts/test_e2e_wav.py'
"""

import os
import sys
import time
import numpy as np
import librosa

# Project root
PROJECT_ROOT = r"C:\Apps\Qwen3-ASR"
sys.path.insert(0, PROJECT_ROOT)

ZH_WAV = os.path.join(PROJECT_ROOT, "test_zh.wav")
EN_WAV = os.path.join(PROJECT_ROOT, "test_en.wav")
EXPECTED_ZH = "甚至出现交易几乎停滞的情况"


def load_wav(path: str, sr: int = 16000) -> np.ndarray:
    """Load WAV file and resample to 16kHz mono float32."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def test_file(wav_path: str, decoder_device: str, language: str | None = None, label: str = ""):
    """Run streaming ASR on a WAV file and return results."""
    from qwen3_asr_app.inference.engine import ASREngine

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  File: {os.path.basename(wav_path)}")
    print(f"  Decoder: {decoder_device}")
    print(f"{'='*60}")

    # Load audio
    audio = load_wav(wav_path)
    duration = len(audio) / 16000
    print(f"  Audio duration: {duration:.1f}s ({len(audio)} samples)")

    # Init engine
    t0 = time.perf_counter()
    engine = ASREngine(
        encoder_device="NPU",
        decoder_device=decoder_device,
        language=language,
    )
    init_time = time.perf_counter() - t0
    print(f"  Engine init: {init_time*1000:.0f}ms")

    # Run streaming: feed audio in 2s chunks
    state = engine.new_session()
    chunk_size = int(2.0 * 16000)  # 2 seconds
    chunk_times = []

    t_total = time.perf_counter()
    pos = 0
    while pos < len(audio):
        chunk = audio[pos:pos + chunk_size]
        pos += chunk_size

        t0 = time.perf_counter()
        engine.feed(chunk, state)
        elapsed = time.perf_counter() - t0
        chunk_times.append(elapsed)

        if state.text:
            print(f"  [Chunk {len(chunk_times)}] {elapsed*1000:.0f}ms | text: {state.text[:60]}...")

    # Flush remaining
    t0 = time.perf_counter()
    engine.finish(state)
    flush_time = time.perf_counter() - t0
    total_time = time.perf_counter() - t_total

    print(f"\n  --- Results ---")
    print(f"  Language: {state.language}")
    print(f"  Text: {state.text}")
    print(f"  Raw decoded: {state._raw_decoded}")
    print(f"  Total inference: {total_time*1000:.0f}ms")
    print(f"  RTF: {total_time/duration:.3f}x")
    print(f"  Chunks: {len(chunk_times)}, flush: {flush_time*1000:.0f}ms")

    return state.text, state.language, total_time, duration


def main():
    print("=" * 60)
    print("  Qwen3-ASR End-to-End WAV Test")
    print("=" * 60)

    # Check files exist
    for f in [ZH_WAV, EN_WAV]:
        if not os.path.exists(f):
            print(f"  ERROR: {f} not found. Download first.")
            sys.exit(1)

    results = {}

    # Test 1: Chinese WAV with CPU decoder
    text, lang, t, dur = test_file(ZH_WAV, "CPU", language="Chinese", label="Chinese (CPU decoder)")
    results["zh_cpu"] = {"text": text, "lang": lang, "time": t, "duration": dur}

    # Test 2: Chinese WAV with NPU decoder
    text, lang, t, dur = test_file(ZH_WAV, "NPU", language="Chinese", label="Chinese (NPU decoder)")
    results["zh_npu"] = {"text": text, "lang": lang, "time": t, "duration": dur}

    # Test 3: Chinese WAV without language hint (auto-detect)
    text, lang, t, dur = test_file(ZH_WAV, "CPU", language=None, label="Chinese auto-detect (CPU)")
    results["zh_auto"] = {"text": text, "lang": lang, "time": t, "duration": dur}

    # Test 4: English WAV with CPU decoder (first 10s only to save time)
    # We'll test the full file separately if needed
    text, lang, t, dur = test_file(EN_WAV, "CPU", language="English", label="English (CPU decoder)")
    results["en_cpu"] = {"text": text, "lang": lang, "time": t, "duration": dur}

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for key, r in results.items():
        rtf = r["time"] / r["duration"]
        match = ""
        if "zh" in key and EXPECTED_ZH in r["text"]:
            match = " [MATCH]"
        elif "zh" in key:
            match = " [MISMATCH]"
        print(f"  {key:15s} | RTF={rtf:.3f}x | lang={r['lang']:8s} | {r['text'][:50]}{match}")

    # Explicit accuracy check
    print()
    for key in ["zh_cpu", "zh_npu", "zh_auto"]:
        r = results[key]
        if EXPECTED_ZH in r["text"]:
            print(f"  {key}: PASS - contains expected text")
        else:
            print(f"  {key}: FAIL - expected '{EXPECTED_ZH}' not found in '{r['text']}'")


if __name__ == "__main__":
    main()
