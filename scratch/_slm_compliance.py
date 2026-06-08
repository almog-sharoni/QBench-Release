import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "runspace"))

import torch
from core.runner import Runner
from src.adapters.adapter_factory import create_adapter
from src.eval.comparator import LayerComparator

config = {
    "model": {"name": "facebook/opt-125m", "source": "huggingface"},
    "adapter": {
        "type": "slm",
        "quantize_first_layer": False,
        "quantized_ops": ["all"],
        "input_quantization": True,
        "weight_quantization": True,
        "quantization_type": "fp8_e4m3",
    },
    "quantization": {"format": "fp8_e4m3", "input_format": "fp8_e4m3", "bias": 7,
                     "mode": "chunk", "chunk_size": 128,
                     "weight_mode": "chunk", "weight_chunk_size": 128,
                     "act_mode": "chunk", "act_chunk_size": 128},
    "dataset": {"name": "wikitext2_lm", "batch_size": 2, "seq_len": 128,
                "num_workers": 0, "max_blocks": 4},
    "evaluation": {"mode": "compare", "compare_batches": 1},
    "output_name": "slm_compliance",
}

runner = Runner()
data_loader = runner.setup_data_loader(config)

quant_adapter = create_adapter(config)
quant_model = quant_adapter.model.to(runner.device).eval()
ref_model = quant_adapter.build_reference_model().to(runner.device).eval()

cmp = LayerComparator(
    ref_model=ref_model,
    quant_model=quant_model,
    model_name="facebook/opt-125m",
    quant_type="fp8_e4m3",
    adapter=quant_adapter,
    device=runner.device,
    compare_mode="propagated",
)
cmp.compare(data_loader, num_batches=1)

print("\n\n##### COVERAGE SUMMARY #####")
for line in cmp.coverage_report_lines:
    print(line)
print(f"\nSupported-but-unquantized ops: {cmp.unquantized_supported_count}")
