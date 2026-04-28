import sys
import os

# Add the project root to sys.path
project_root = "/data/almog/Projects/QBench-Release"
sys.path.append(project_root)

from runspace.src.ops.quant_softmax import qtype_to_unsigned_qtype

test_cases = [
    "fp32",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8",
    "ufp8_e4m4"
]

for tc in test_cases:
    print(f"{tc} -> {qtype_to_unsigned_qtype(tc)}")
