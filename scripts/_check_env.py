import openvino as ov
print("OpenVINO:", ov.__version__)
core = ov.Core()
print("Devices:", core.available_devices)
try:
    print("NPU driver:", core.get_property("NPU", "DRIVER_VERSION"))
except Exception as e:
    print("NPU driver query failed:", e)

try:
    import openvino_genai
    print("GenAI version:", openvino_genai.__version__)
except ImportError:
    print("openvino-genai: NOT INSTALLED")
