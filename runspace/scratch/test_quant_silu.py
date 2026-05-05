import torch
import torch.nn as nn
import sys
import os

# Add runspace/src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from runspace.src.ops.quant_activations import QuantSiLU

def test_quant_silu():
    print("Testing QuantSiLU...")
    A = 4.0
    q_silu = QuantSiLU(A=A)
    silu = nn.SiLU()

    # Test values
    test_inputs = torch.tensor([-10.0, -5.0, -4.0, -2.0, 0.0, 2.0, 4.0, 5.0, 10.0])
    
    with torch.no_grad():
        output_quant = q_silu(test_inputs)
        output_ref = silu(test_inputs)

    print(f"{'Input':>10} | {'Quant':>10} | {'Ref':>10} | {'Diff':>10}")
    print("-" * 50)
    for i, val in enumerate(test_inputs):
        q_val = output_quant[i].item()
        r_val = output_ref[i].item()
        diff = abs(q_val - r_val)
        print(f"{val.item():10.4f} | {q_val:10.4f} | {r_val:10.4f} | {diff:10.4f}")

    # Check piecewise logic
    # x <= -A should be 0
    assert output_quant[0] == 0.0
    assert output_quant[1] == 0.0
    
    # x >= A should be x
    assert output_quant[-1] == 10.0
    assert output_quant[-2] == 5.0

    print("\nPiecewise logic verified (x <= -A -> 0, x >= A -> x)")
    
    # Check LUT region error
    # For A=4, L=256, max error should be small
    max_diff = (output_quant[2:-2] - output_ref[2:-2]).abs().max().item()
    print(f"\nMax difference in LUT region [-4, 4]: {max_diff:.6f}")
    assert max_diff < 0.1 # SiLU has a dip at -1.28, LUT should capture it reasonably well

    print("\nQuantSiLU test passed!")

if __name__ == "__main__":
    test_quant_silu()
