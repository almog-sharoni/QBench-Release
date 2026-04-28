import sys
import os

# Add the project root to sys.path
project_root = "/data/almog/Projects/QBench-Release"
sys.path.append(project_root)

from runspace.src.ops.quant_softmax import qtype_to_unsigned_qtype

test_cases = [
    ("fp8_e4m3", "mant"),
    ("fp8_e4m3", "exp"),
    ("fp8_e5m2", "mant"),
    ("fp8_e5m2", "exp"),
]

for tc, add_to in test_cases:
    print(f"{tc} (add_to={add_to}) -> {qtype_to_unsigned_qtype(tc, add_to=add_to)}")
