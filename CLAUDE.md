# Project: Qwen3-ASR Desktop — 流式语音识别 Windows 桌面应用

## 项目目标

基于 Qwen3-ASR-0.6B 模型，构建一个轻量级 Windows 桌面应用，实现：
- 实时麦克风流式语音识别（Streaming ASR）
- OpenVINO 加速推理（Encoder→NPU, Decoder→NPU 全 NPU 部署）
- 低延迟、低功耗的本地离线识别

## Development Environment

- **OS**: Windows 11 (accessed via WSL2)
- **CPU**: Intel Core Ultra 7 258V (4P + 4E, Lunar Lake)
- **GPU**: Intel Arc 140V (16GB, Xe2, ~67 TOPS INT8)
- **NPU**: Intel AI Boost NPU 4 (~47 TOPS INT8)
- **Memory**: On-package LPDDR5X-8533, 共享统一内存架构 (UMA)
- **Python**: 3.12, managed by `uv`
- **OpenVINO**: 2026.0.0
- **NNCF**: 3.0.0

## 技术栈

### 推理引擎：OpenVINO NPU + CPU 混合部署

Qwen3-ASR **无官方 OpenVINO 支持**（`Qwen3ASRForConditionalGeneration` 未注册），
已手动拆分 audio encoder / text decoder 分别导出。

**最终架构（全 NPU）**：
- **Audio Encoder → NPU**：`ov.compile_model()` 直接推理，静态形状 `[1, 128, 800]`，~50ms
- **Text Decoder → NPU**：IR 图手术 stateful 模型 + NPUW_LLM，接受 `inputs_embeds` + KV-cache，**RTF 0.17x**

**IR 图手术方案**：对 optimum-intel 导出的 stateful decoder IR 进行手术：
1. 将 embedding Gather 节点的消费者重定向到新的 `inputs_embeds` Parameter 输入
2. **必须移除 `input_ids` Parameter**（见下方"NPUW_LLM 关键约束"）
3. 断开 `input_ids` 的所有消费者（ShapeOf、Convert→Gather 等），使其成为孤立节点
4. 创建新 Model 时排除 `input_ids`，让图验证通过

> 手术脚本：`scripts/fix_decoder_remove_input_ids.py`

**NPUW_LLM 关键约束（input_ids vs inputs_embeds）**：

NPUW_LLM 插件在 `llm_infer_request.cpp:124-133` 中用 **名称优先** 策略选择主输入：
```cpp
auto input_ids_port = find_port_by_name(inputs(), "input_ids");
if (input_ids_port.has_value()) {
    m_input_ids_name = "input_ids";     // 找到 input_ids → 用它
} else {
    m_input_ids_name = "inputs_embeds"; // 没找到 → 回退到 inputs_embeds
}
```

如果模型同时有 `input_ids` 和 `inputs_embeds` 两个输入，NPUW 会选择 `input_ids`（int64 零值），
导致 decoder 输出垃圾（全 `!` 号）。**必须从模型 IR 中移除 `input_ids` 参数**，
让 NPUW 回退到 `inputs_embeds` 路径。

> 源码参考：`/mnt/c/Apps/openvino/src/plugins/intel_npu/src/plugin/npuw/llm_infer_request.cpp`

**NPU 编译配置 (NPUW_LLM)**：
```python
NPU_DECODER_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 256,
    "NPUW_LLM_MIN_RESPONSE_LEN": 64,
}
```

### 模型转换路线

```
Audio Encoder: PyTorch → ov.convert_model() → encoder_fp16.xml → NPU (ov.compile_model)
Text Decoder:  PyTorch → Qwen3ForCausalLM → optimum-cli(FP16 stateful) → IR图手术(inputs_embeds, 移除input_ids) → decoder_stateful_embeds/ → NPU (NPUW_LLM)
Text Decoder (text-only ref): PyTorch → Qwen3ForCausalLM → optimum-cli(INT4) → decoder_genai_int4/ → NPU (LLMPipeline, 仅纯文本可用)
```

### Benchmark 结果

| 方案 | RTF / ms/token | ASR 可用 | 备注 |
|------|----------------|----------|------|
| **NPU IR-surgery NPUW_LLM (inputs_embeds)** | **RTF 0.17x** | **YES** | **当前方案**，全 NPU，KV-cache + inputs_embeds |
| CPU IR-surgery stateful (inputs_embeds) | RTF 0.62x, ~28ms/tok | YES | CPU 备选方案 |
| NPU GenAI INT4 (LLMPipeline) | 10.0ms/tok | NO | 只接受 token IDs，无法注入音频 embedding |
| CPU GenAI INT4 | 11.1-17.4ms/tok | NO | 同上，LLMPipeline API 限制 |
| CPU OVModel FP16 stateful | 58.8ms/tok | NO | optimum-intel 原始导出，无 inputs_embeds |
| NPU 无 KV-cache | 132.6ms/tok | YES (慢) | 静态形状直接推理，已弃用 |

**E2E WAV 测试（test_zh.wav, 4.2s 中文语音）**：
- NPU decoder: RTF 0.171x, 总计 717ms (chunk1=297ms, chunk2=234ms), 转写正确
- CPU decoder: RTF 0.623x, 总计 2618ms (chunk1=1021ms, chunk2=788ms), 转写正确
- **NPU 比 CPU 快 3.6x**

**NPU decoder 注意事项**：
- 初始化慢（~7.6s NPU 编译 vs ~2s CPU），适合长时间运行的桌面应用
- NPUW_LLM prefill 返回 [1,1,vocab] 形状（仅最后位置 logits），CPU 返回 [1,seq,vocab]
- `logits[0, -1, :]` 两者都适用

### 已导出模型文件

```
models/
├── encoder_fp16.xml/.bin        # Audio encoder, 固定 [1, 128, 800] → [1, 104, 1024], NPU
├── decoder_stateful_embeds/     # Text decoder (IR图手术, inputs_embeds + KV-cache stateful, 无input_ids), NPU ★当前 ASR 方案
│   └── openvino_model.xml/.bin + tokenizer
├── decoder_genai_int4/          # Text decoder (INT4, GenAI LLMPipeline 用), NPU (仅纯文本, ASR 不可用)
│   └── openvino_model.xml/.bin + tokenizer
├── decoder_fp16.xml/.bin        # Text decoder (无 KV-cache), 固定 [1, 256, 1024], 已弃用
├── decoder_stateful_ov/         # Text decoder (KV-cache stateful), optimum-intel 原始导出 (IR手术前的基础)
├── qwen3_decoder_standalone/    # Qwen3ForCausalLM 格式的 thinker 权重 (中间产物)
├── embed_tokens.npy             # Embedding table [151936, 1024] (构建 inputs_embeds 用)
└── cache/                       # OpenVINO 编译缓存
```

---

### 流式推理策略（Streaming ASR）

Qwen3-ASR 原生支持流式推理。上游实现：
- 核心逻辑：`qwen_asr/inference/qwen3_asr.py`（`streaming_transcribe`）
- vLLM 示例：`examples/example_qwen3_asr_vllm_streaming.py`
- Web Demo：`qwen_asr/cli/demo_streaming.py`

**注意**：上游仅支持 vLLM 后端做流式。我们的 OpenVINO 方案需要自己实现等效逻辑。

**核心机制：累积重编码 + 文本前缀回退**

不是独立处理每个音频块，而是：
1. **累积所有音频**：新音频追加到 `audio_accum` buffer，每次推理重新编码从开头到当前的全部音频
2. **前缀提示**：把上一轮的转写结果（去掉末尾 N 个 token）作为 decoder 的前缀，引导续写
3. **回退修正**：`unfixed_token_num=5`，每次回退 5 个 token 重新生成，允许模型修正前面的错误
4. **Unicode 安全**：回退 token 时如果切到多字节字符中间（产生 `\ufffd`），自动多退几个 token 直到得到合法 UTF-8

**参数**（不同场景有不同默认值）：

| 参数 | example (vllm) | 核心库默认 | web demo |
|------|---------------|-----------|----------|
| `chunk_size_sec` | 2.0 | 2.0 | **1.0** |
| `unfixed_chunk_num` | 2 | 2 | **4** |
| `unfixed_token_num` | 5 | 5 | 5 |
| `max_new_tokens` | **32** | 512 | **32** |

桌面应用推荐：`chunk_size_sec=2.0`, `unfixed_chunk_num=2`, `unfixed_token_num=5`, `max_new_tokens=32`

**语言强制**：如果指定了语言，会在 assistant prompt 里预填 `language {Language}<asr_text>` 前缀，强制模型输出该语言。

**对 OpenVINO 部署的影响**：
- Audio Encoder 需要处理累积增长的音频（当前固定 800 帧 = 5s，需要扩展或分窗口）
- Decoder 使用 NPU NPUW_LLM stateful 模型（IR-surgery），通过 `inputs_embeds` 注入音频特征
- 每个 chunk 重置 KV-cache 并重新 prefill（因为累积音频变化导致 encoder 输出变化，embedding 序列整体改变）
- 性能预算（NPU）：单 chunk ~250-300ms，RTF 0.17x，远低于 2s chunk 预算
- 流式输出格式：`language {Language}<asr_text>{Transcription}`

**Prompt 格式 (ChatML)**：
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n
<|im_start|>assistant\n[language Chinese<asr_text>前缀文本...]
```

特殊 Token ID：
- `<|im_start|>` = 151644, `<|im_end|>` = 151645
- `<|audio_start|>` = 151669, `<|audio_end|>` = 151670, `<|audio_pad|>` = 151676
- `<asr_text>` = 151704

---

### 桌面 UI：PySide6 (Qt for Python)

选择理由：
- 与推理代码同语言（Python），无 IPC 开销
- `QAudioSource` 直接采集麦克风 PCM 音频
- `QThread` 后台推理不阻塞 UI
- 原生 Windows 外观，体积可控
- PyInstaller 打包为单 EXE

核心组件：
- **AudioCaptureThread**: QThread，通过 QAudioSource 采集 16kHz/16bit PCM
- **InferenceThread**: QThread，运行 OpenVINO 推理（固定分块 → mel → encoder → decoder）
- **MainWindow**: 显示实时转写文本、音量指示

### 音频处理：WhisperFeatureExtractor + librosa

- 16kHz 单声道 PCM（QAudioSource 采集）
- 128-bin Log-Mel 频谱（WhisperFeatureExtractor，与原模型 preprocessor_config.json 一致）
- 流式分块：每 2 秒触发推理，累积全部音频重编码
- Encoder 固定输入 800 帧 ≈ 5s，超长音频需分窗 padding

## Development Conventions

### Use PowerShell to control Windows

所有涉及 Windows Python/uv 的命令必须通过 PowerShell 执行：

```bash
# Pattern: run Windows commands from WSL
powershell.exe -Command '<powershell commands here>'

# Run Python scripts (always set UTF-8):
powershell.exe -Command '$env:Path = "C:\Users\taowen\.local\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\Apps\Qwen3-ASR; uv run python <script>'
```

- **Always set `$env:PYTHONIOENCODING = "utf-8"`**，否则中文输出和 rich 进度条会因 GBK 编码崩溃
- 用 `uv run python <script>` 运行脚本，确保正确虚拟环境
- 不要在 WSL 中直接运行项目代码，始终通过 PowerShell 使用 Windows 原生 Python

### uv 配置

pyproject.toml 中 XPU 源必须设为 `explicit = true`：

```toml
[tool.uv]
environments = ["sys_platform == 'win32'"]

[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-xpu" }]
torchaudio = [{ index = "pytorch-xpu" }]
```

### Project Commands

```powershell
# 验证 NPU 可用
uv run python -c "import openvino as ov; print(ov.Core().available_devices)"

# 运行桌面应用
uv run python -m qwen3_asr_app.main

# 运行原始 demo（Transformers 后端）
uv run python -m qwen_asr.cli.demo --model Qwen3-ASR-0.6B --audio test.wav

# 模型转换 benchmark
uv run python scripts/export_openvino.py
uv run python scripts/benchmark_npu.py
```

## 项目结构（规划）

```
Qwen3-ASR/
├── CLAUDE.md
├── pyproject.toml
├── qwen_asr/                  # 上游 Qwen3-ASR 库（保持不动）
│   ├── core/                  # transformers / vllm 后端
│   ├── inference/             # 推理封装
│   └── cli/                   # 命令行 demo
├── qwen3_asr_app/             # 新增：桌面应用
│   ├── main.py                # 入口，PySide6 应用
│   ├── ui/
│   │   └── main_window.py     # 主窗口（实时转写文本 + 音量指示）
│   ├── audio/
│   │   ├── capture.py         # QAudioSource 麦克风采集 (16kHz/16bit PCM)
│   │   └── processor.py       # Mel 特征提取、流式分块
│   ├── inference/
│   │   ├── engine.py          # 流式推理引擎（累积重编码 + 前缀回退）
│   │   ├── ov_encoder.py      # Audio Encoder (NPU, encoder_fp16.xml)
│   │   └── ov_decoder.py      # Text Decoder (NPU/CPU, decoder_stateful_embeds/, IR surgery + NPUW_LLM)
│   └── config.py              # 应用配置
├── scripts/                   # 已有脚本
│   ├── export_encoder.py      # Audio encoder 导出 (已完成)
│   ├── export_decoder.py      # Text decoder 无 KV-cache 导出 (已完成)
│   ├── export_decoder_stateful.py  # Text decoder stateful 导出 (已完成)
│   ├── test_e2e.py            # 端到端验证 (已完成)
│   ├── test_npu.py            # NPU 推理测试 (已完成)
│   ├── benchmark_decoder.py   # CPU vs NPU decoder benchmark (已完成)
│   ├── benchmark_genai_npu.py # GenAI LLMPipeline NPU benchmark (已完成)
│   ├── test_genai_npu.py      # GenAI NPU 验证 (已完成)
│   └── quantize_int8.py       # NNCF INT8 量化 (待写)
├── Qwen3-ASR-0.6B/            # 本地模型权重
└── tests/
```

## 模型信息

- **模型**: Qwen3-ASR-0.6B（0.6B 参数，1.87GB safetensors）
- **架构**: `Qwen3ASRForConditionalGeneration`
  - Audio Encoder: 3 层 2D Conv + 18 层 Transformer（d_model=896, 16 heads）
  - Text Decoder: 28 层 Qwen3 LLM（hidden=1024, 16 heads, 8 KV heads, GQA）
  - Mel bins: 128, conv_chunksize=500, n_window_infer=800
- **支持语言**: 52 种语言和方言
- **流式**: 2 秒分块 + 累积重编码 + 前缀回退 5 tokens（详见"流式推理策略"章节）

## 开发 Checklist

- [x] Audio encoder 导出 OpenVINO IR（`encoder_fp16.xml`，静态 `[1, 128, 800]`，NPU 验证通过）
- [x] Text decoder 导出 OpenVINO IR — 多版本：无 KV-cache (`decoder_fp16.xml`), CPU stateful (`decoder_stateful_ov/`), NPU GenAI INT4 (`decoder_genai_int4/`)
- [x] 端到端推理验证（`test_e2e.py`：静音 → `language None<asr_text>` 正确）
- [x] Benchmark：NPU GenAI INT4 (10ms/token, 纯文本) vs CPU stateful (59ms) vs NPU no-cache (133ms)
- [x] NPU LLMPipeline 不可用于 ASR：只接受 token IDs，无法注入 inputs_embeds（音频特征）
- [x] IR 图手术 decoder（`decoder_stateful_embeds/`）：替换 Gather 为 inputs_embeds Parameter，移除 input_ids，KV-cache stateful
- [x] 全 NPU decoder 推理（NPUW_LLM + inputs_embeds）：RTF 0.17x，比 CPU 快 3.6x
  - 根因：NPUW_LLM 按名称优先选 `input_ids`，必须从 IR 中移除 `input_ids` Parameter
  - 修复脚本：`scripts/fix_decoder_remove_input_ids.py`
  - 源码参考：`openvino/src/plugins/intel_npu/src/plugin/npuw/llm_infer_request.cpp:124`
- [x] 流式推理引擎（`qwen3_asr_app/inference/engine.py`：累积重编码 + 前缀回退 + greedy decode）
- [x] PySide6 桌面应用（`qwen3_asr_app/`：麦克风采集 + 实时转写 + 音量指示 + Start/Stop/Clear）
- [x] 真实音频端到端测试（`scripts/test_e2e_wav.py`：中文 PASS，英文 PASS，NPU RTF 0.17x）
- [ ] PyInstaller 打包为 Windows EXE