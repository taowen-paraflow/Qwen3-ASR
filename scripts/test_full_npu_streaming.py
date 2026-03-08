"""Test full-NPU streaming ASR pipeline (encoder AND decoder on NPU).

Runs two configurations:
  1. Full NPU:  encoder=NPU, decoder=NPU  (NPUW_LLM ~21ms/token)
  2. NPU+CPU:   encoder=NPU, decoder=CPU  (IR-surgery ~28ms/token)

Tests:
  - Silence test: 4 seconds of silence (2 chunks of 2s), expect empty/minimal output
  - Performance benchmark: per-chunk latency, total time, real-time factor

Usage:
    uv run python scripts/test_full_npu_streaming.py
"""

import sys
import time

sys.path.insert(0, r"C:\Apps\Qwen3-ASR")

import numpy as np
from qwen3_asr_app.inference.engine import ASREngine


SAMPLE_RATE = 16000
CHUNK_SEC = 2.0
NUM_SILENCE_CHUNKS = 2  # 4 seconds total


def make_silence(num_chunks: int, chunk_sec: float = CHUNK_SEC) -> np.ndarray:
    """Generate silent audio (zeros) for the requested number of chunks."""
    total_samples = int(num_chunks * chunk_sec * SAMPLE_RATE)
    return np.zeros(total_samples, dtype=np.float32)


def run_streaming_test(
    engine: ASREngine,
    audio: np.ndarray,
    label: str,
) -> dict:
    """Feed audio through the engine chunk-by-chunk, collecting timing info.

    Returns a dict with per-chunk latencies, total time, and final state.
    """
    state = engine.new_session()
    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    offset = 0
    chunk_latencies: list[float] = []

    print(f"\n  Streaming chunks ({label}):")
    while offset < len(audio):
        end = min(offset + chunk_samples, len(audio))
        chunk = audio[offset:end]
        chunk_dur = len(chunk) / SAMPLE_RATE

        t0 = time.perf_counter()
        if end >= len(audio):
            engine.feed(chunk, state)
            engine.finish(state)
        else:
            engine.feed(chunk, state)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        chunk_latencies.append(elapsed_ms)
        print(
            f"    chunk {len(chunk_latencies):2d}  "
            f"[{offset/SAMPLE_RATE:.1f}s-{end/SAMPLE_RATE:.1f}s]  "
            f"audio={chunk_dur:.1f}s  latency={elapsed_ms:7.1f}ms  "
            f"text='{state.text}'"
        )
        offset = end

    total_ms = sum(chunk_latencies)
    audio_dur = len(audio) / SAMPLE_RATE
    rtf = total_ms / (audio_dur * 1000) if audio_dur > 0 else float("inf")

    return {
        "label": label,
        "chunk_latencies": chunk_latencies,
        "total_ms": total_ms,
        "audio_dur": audio_dur,
        "rtf": rtf,
        "language": state.language,
        "text": state.text,
        "raw": state._raw_decoded,
    }


def print_separator(char: str = "=", width: int = 70):
    print(char * width)


def print_header(title: str, char: str = "=", width: int = 70):
    print()
    print_separator(char, width)
    print(f"  {title}")
    print_separator(char, width)


def main():
    print_header("Full-NPU Streaming ASR Pipeline Test")

    silence = make_silence(NUM_SILENCE_CHUNKS)
    print(f"\n  Test audio: {len(silence)/SAMPLE_RATE:.1f}s of silence "
          f"({NUM_SILENCE_CHUNKS} chunks x {CHUNK_SEC}s)")

    # ------------------------------------------------------------------
    # 1. Full NPU (encoder=NPU, decoder=NPU)
    # ------------------------------------------------------------------
    print_header("Initializing Full-NPU engine (encoder=NPU, decoder=NPU)", "-")
    t0 = time.perf_counter()
    try:
        engine_npu = ASREngine(
            encoder_device="NPU",
            decoder_device="NPU",
            language="Chinese",
        )
        init_npu_ms = (time.perf_counter() - t0) * 1000
        print(f"  Initialized in {init_npu_ms:.0f}ms")
        npu_ok = True
    except Exception as exc:
        init_npu_ms = (time.perf_counter() - t0) * 1000
        print(f"  FAILED after {init_npu_ms:.0f}ms: {exc}")
        npu_ok = False
        engine_npu = None

    npu_silence_result = None
    npu_bench_result = None

    if npu_ok:
        # Test 1: Silence
        print_header("Test 1 - Silence (Full NPU)", "-")
        npu_silence_result = run_streaming_test(engine_npu, silence, "NPU silence")

        text = npu_silence_result["text"].strip()
        if len(text) == 0:
            print("\n  PASS: Silence produced empty text (as expected)")
        elif len(text) < 5:
            print(f"\n  PASS: Silence produced minimal text: '{text}'")
        else:
            print(f"\n  NOTE: Silence produced text: '{text}' (len={len(text)})")

        # Test 2: Performance benchmark (re-run silence for consistent comparison)
        print_header("Test 2 - Performance Benchmark (Full NPU)", "-")
        npu_bench_result = run_streaming_test(engine_npu, silence, "NPU bench")

        # Release NPU decoder resources before loading CPU variant
        del engine_npu

    # ------------------------------------------------------------------
    # 2. CPU decoder baseline (encoder=NPU, decoder=CPU)
    # ------------------------------------------------------------------
    print_header("Initializing CPU-decoder engine (encoder=NPU, decoder=CPU)", "-")
    t0 = time.perf_counter()
    try:
        engine_cpu = ASREngine(
            encoder_device="NPU",
            decoder_device="CPU",
            language="Chinese",
        )
        init_cpu_ms = (time.perf_counter() - t0) * 1000
        print(f"  Initialized in {init_cpu_ms:.0f}ms")
        cpu_ok = True
    except Exception as exc:
        init_cpu_ms = (time.perf_counter() - t0) * 1000
        print(f"  FAILED after {init_cpu_ms:.0f}ms: {exc}")
        cpu_ok = False
        engine_cpu = None

    cpu_silence_result = None
    cpu_bench_result = None

    if cpu_ok:
        # Test 1: Silence
        print_header("Test 1 - Silence (CPU decoder)", "-")
        cpu_silence_result = run_streaming_test(engine_cpu, silence, "CPU silence")

        text = cpu_silence_result["text"].strip()
        if len(text) == 0:
            print("\n  PASS: Silence produced empty text (as expected)")
        elif len(text) < 5:
            print(f"\n  PASS: Silence produced minimal text: '{text}'")
        else:
            print(f"\n  NOTE: Silence produced text: '{text}' (len={len(text)})")

        # Test 2: Performance benchmark
        print_header("Test 2 - Performance Benchmark (CPU decoder)", "-")
        cpu_bench_result = run_streaming_test(engine_cpu, silence, "CPU bench")

        del engine_cpu

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_header("SUMMARY: NPU vs CPU Decoder Comparison")

    row_fmt = "  {:<28s} {:>12s} {:>12s} {:>12s}"
    print(row_fmt.format("Metric", "Full NPU", "NPU+CPU", "Speedup"))
    print("  " + "-" * 66)

    def fmt_ms(v):
        return f"{v:.0f}ms" if v is not None else "N/A"

    def fmt_rtf(v):
        return f"{v:.3f}x" if v is not None else "N/A"

    def speedup(cpu_val, npu_val):
        if cpu_val is not None and npu_val is not None and npu_val > 0:
            return f"{cpu_val / npu_val:.2f}x"
        return "N/A"

    npu_total = npu_bench_result["total_ms"] if npu_bench_result else None
    cpu_total = cpu_bench_result["total_ms"] if cpu_bench_result else None
    npu_rtf = npu_bench_result["rtf"] if npu_bench_result else None
    cpu_rtf = cpu_bench_result["rtf"] if cpu_bench_result else None

    print(row_fmt.format(
        "Init time",
        fmt_ms(init_npu_ms if npu_ok else None),
        fmt_ms(init_cpu_ms if cpu_ok else None),
        "",
    ))
    print(row_fmt.format(
        "Total inference time",
        fmt_ms(npu_total),
        fmt_ms(cpu_total),
        speedup(cpu_total, npu_total),
    ))
    print(row_fmt.format(
        "Real-time factor (RTF)",
        fmt_rtf(npu_rtf),
        fmt_rtf(cpu_rtf),
        speedup(cpu_rtf, npu_rtf),
    ))

    # Per-chunk comparison
    if npu_bench_result and cpu_bench_result:
        n_chunks = min(
            len(npu_bench_result["chunk_latencies"]),
            len(cpu_bench_result["chunk_latencies"]),
        )
        print()
        chunk_fmt = "  {:<28s} {:>12s} {:>12s} {:>12s}"
        for i in range(n_chunks):
            npu_lat = npu_bench_result["chunk_latencies"][i]
            cpu_lat = cpu_bench_result["chunk_latencies"][i]
            label = f"Chunk {i+1} latency"
            print(chunk_fmt.format(
                label,
                fmt_ms(npu_lat),
                fmt_ms(cpu_lat),
                speedup(cpu_lat, npu_lat),
            ))

    # Silence test results
    print()
    print("  Silence test output:")
    if npu_silence_result:
        print(f"    Full NPU : lang='{npu_silence_result['language']}', "
              f"text='{npu_silence_result['text']}'")
    else:
        print("    Full NPU : (not available)")
    if cpu_silence_result:
        print(f"    NPU+CPU  : lang='{cpu_silence_result['language']}', "
              f"text='{cpu_silence_result['text']}'")
    else:
        print("    NPU+CPU  : (not available)")

    # Final verdict
    print()
    if npu_total is not None and cpu_total is not None:
        if npu_total < cpu_total:
            pct = (1 - npu_total / cpu_total) * 100
            print(f"  RESULT: Full-NPU is {pct:.0f}% faster than CPU decoder "
                  f"({fmt_ms(npu_total)} vs {fmt_ms(cpu_total)})")
        elif npu_total > cpu_total:
            pct = (npu_total / cpu_total - 1) * 100
            print(f"  RESULT: Full-NPU is {pct:.0f}% SLOWER than CPU decoder "
                  f"({fmt_ms(npu_total)} vs {fmt_ms(cpu_total)})")
        else:
            print(f"  RESULT: NPU and CPU decoder have identical performance")
    elif npu_total is None:
        print("  RESULT: Full-NPU engine failed to initialize; only CPU results available")
    elif cpu_total is None:
        print("  RESULT: CPU engine failed to initialize; only NPU results available")

    audio_dur = len(silence) / SAMPLE_RATE * 1000
    if npu_total is not None and npu_total < audio_dur:
        print(f"  Full-NPU is real-time capable (RTF={npu_rtf:.3f}x < 1.0)")
    elif npu_total is not None:
        print(f"  Full-NPU is NOT real-time (RTF={npu_rtf:.3f}x > 1.0)")

    print_separator()
    print()


if __name__ == "__main__":
    main()
