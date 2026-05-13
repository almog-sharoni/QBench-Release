import torch
import torch.nn as nn
from runspace.src.ops.quant_activations import QuantHardsigmoid

def test_quant_hardsigmoid():
    print("Testing QuantHardsigmoid...")
    
    # Initialize
    qhs = QuantHardsigmoid(A=4.0)
    
    # Test input
    x = torch.tensor([-5.0, -4.0, -3.0, 0.0, 3.0, 4.0, 5.0])
    
    # Forward pass (without quantization enabled, should bypass)
    qhs.input_quantization = False
    y_bypass = qhs(x)
    y_expected = nn.functional.hardsigmoid(x)
    print(f"Bypass output: {y_bypass}")
    assert torch.allclose(y_bypass, y_expected), "Bypass path failed"
    
    # Forward pass (with quantization enabled, should use LUT)
    qhs.input_quantization = True
    
    # Monkeypatch quantize_output to return the input for testing
    qhs.quantize_output = lambda x: x
    
    y_lut = qhs(x)
    print(f"LUT output:    {y_lut}")
    
    # Verify boundaries
    assert y_lut[0] == 0.0  # -5.0 <= -4.0
    assert y_lut[1] == 0.0  # -4.0 <= -4.0
    assert y_lut[5] == 1.0  # 4.0 >= 4.0
    assert y_lut[6] == 1.0  # 5.0 >= 4.0
    
    # Verify state_dict for persistent=False
    sd = qhs.state_dict()
    assert 'piecewise_lut' not in sd, "piecewise_lut should not be in state_dict"
    print("state_dict check passed (piecewise_lut excluded)")
    
    print("QuantHardsigmoid test passed!")

if __name__ == "__main__":
    try:
        test_quant_hardsigmoid()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
