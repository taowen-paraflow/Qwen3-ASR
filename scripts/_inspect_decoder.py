import openvino as ov

core = ov.Core()
m = core.read_model("models/decoder_fp16.xml")
print("decoder_fp16.xml inputs:")
for i in m.inputs:
    print(f"  {i.any_name}: {i.partial_shape}")
print("decoder_fp16.xml outputs:")
for o in m.outputs:
    print(f"  {o.any_name}: {o.partial_shape}")
