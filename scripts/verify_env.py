"""Verify development environment setup."""
import sys

print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")

import openvino as ov
print(f"OpenVINO: {ov.__version__}")
print(f"Devices: {ov.Core().available_devices}")

import nncf
print(f"NNCF: {nncf.__version__}")

import PySide6
print(f"PySide6: {PySide6.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")

import librosa
print(f"librosa: {librosa.__version__}")

print("\nAll dependencies OK!")
