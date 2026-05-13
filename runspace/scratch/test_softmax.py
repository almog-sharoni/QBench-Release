import torch
import torch.nn as nn
from runspace.src.ops.quant_softmax import QuantSoftmax

def test_quant_softmax():
    print("Testing QuantSoftmax...")
    
    # Initialize
    qs = QuantSoftmax(dim=-1, q_type="fp8_e4m3")
    
    # Test input
    x = torch.randn(2, 10).cuda()
    
    # 1. Test Bypass Path
    print("Testing bypass path...")
    qs.input_quantization = False
    qs.capture_activations = True
    y_bypass = qs(x)
    
    # Check if capture attributes are initialized
    assert hasattr(qs, 'last_quant_input'), "last_quant_input missing"
    assert hasattr(qs, 'last_quant_inputs_unscaled'), "last_quant_inputs_unscaled missing"
    assert hasattr(qs, 'last_quant_input_max'), "last_quant_input_max missing"
    print("Bypass path capture check passed.")
    
    # 2. Test Main Path
    print("Testing main path...")
    qs.input_quantization = True
    qs.capture_activations = True
    
    # We need to mock quantize_input/quantize_output if they fail on CUDA requirements or similar
    # But since we are on H100 (from the hostname in metadata), CUDA should work.
    
    y_quant = qs(x)
    print(f"Quantized output shape: {y_quant.shape}")
    
    # Check multiple unscaled inputs capture
    assert len(qs.last_quant_inputs_unscaled) == 2, "Should capture 2 unscaled inputs"
    assert qs.last_quant_input_formats == ["fp8_e4m3", "fp8_e4m3"], "Formats mismatch"
    
    print("Main path capture check passed.")
    
    # 3. Test Unsigned Input Sources
    print("Testing unsigned input sources...")
    qs_u = QuantSoftmax(dim=-1, q_type="fp8_e4m3", unsigned_input_sources=["softmax"])
    qs_u.capture_activations = True
    y_u = qs_u(x)
    assert qs_u.last_quant_input_formats[1].startswith("u"), f"Should use unsigned format, got {qs_u.last_quant_input_formats[1]}"
    print("Unsigned input sources check passed.")
    
    print("QuantSoftmax test passed!")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test (QuantSoftmax requires CUDA for codec).")
    else:
        try:
            test_quant_softmax()
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
