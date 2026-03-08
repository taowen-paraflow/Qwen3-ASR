"""Quick dependency check."""
import sys

# Check Qwen3ForCausalLM
try:
    from transformers import Qwen3ForCausalLM
    print(f"Qwen3ForCausalLM: YES - {Qwen3ForCausalLM}")
except ImportError as e:
    print(f"Qwen3ForCausalLM: NO - {e}")

# Check optimum-intel
try:
    from optimum.intel import OVModelForCausalLM
    print(f"OVModelForCausalLM: YES")
except ImportError as e:
    print(f"OVModelForCausalLM: NO - {e}")

# Check make_stateful API
try:
    import openvino as ov
    # Check for make_stateful in various locations
    if hasattr(ov, 'make_stateful'):
        print(f"ov.make_stateful: YES")
    else:
        print(f"ov.make_stateful: NO")

    from openvino.runtime.passes import Manager
    print(f"ov.runtime.passes.Manager: YES")

    # Check for MakeStateful pass
    try:
        from openvino.runtime.passes import MakeStateful
        print(f"MakeStateful pass: YES")
    except ImportError:
        print(f"MakeStateful pass: NO")
except Exception as e:
    print(f"OpenVINO passes check error: {e}")

# Check Qwen3Config
try:
    from transformers import Qwen3Config
    print(f"Qwen3Config: YES")
except ImportError as e:
    print(f"Qwen3Config: NO - {e}")
