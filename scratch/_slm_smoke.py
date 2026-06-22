import os, sys, copy, collections
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.join(REPO_ROOT, "runspace")
sys.path.insert(0, PROJECT_ROOT)

import torch
from core.runner import Runner
from src.adapters.adapter_factory import create_adapter

MAX_BLOCKS = int(os.environ.get("MAX_BLOCKS", "40"))

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
    "quantization": {"format": "fp8_e4m3", "bias": 7,
                     "mode": "chunk", "chunk_size": 128,
                     "weight_mode": "chunk", "weight_chunk_size": 128,
                     "act_mode": "chunk", "act_chunk_size": 128},
    "dataset": {"name": "wikitext2_lm", "batch_size": 4, "seq_len": 512,
                "num_workers": 0, "max_blocks": MAX_BLOCKS},
    "evaluation": {"mode": "compare", "compare_batches": -1},
    "output_name": "slm_smoke",
}

runner = Runner()
data_loader = runner.setup_data_loader(config)

def count_quant(model):
    c = collections.Counter()
    for m in model.modules():
        n = type(m).__name__
        if n.startswith("Quant"):
            c[n] += 1
    return dict(c)

# --- Quantized model ---
q_adapter = create_adapter(config)
q_model = q_adapter.model.to(runner.device).eval()
print("QUANT layers:", count_quant(q_model))
q_res = runner.evaluate_model(q_model, data_loader, q_adapter, max_batches=-1, desc="FP8")

# --- FP reference ---
ref_cfg = copy.deepcopy(config)
ref_cfg["adapter"].update(weight_quantization=False, input_quantization=False,
                          quantized_ops=[], build_quantized=False)
ref_cfg["quantization"]["format"] = "fp32"
ref_adapter = create_adapter(ref_cfg)
ref_model = ref_adapter.model.to(runner.device).eval()
print("REF Quant layers (should be empty):", count_quant(ref_model))
ref_res = runner.evaluate_model(ref_model, data_loader, ref_adapter, max_batches=-1, desc="FP32")

print("\n==================== RESULTS ====================")
print(f"blocks evaluated : {MAX_BLOCKS} x seq_len 512")
print(f"FP32  perplexity : {ref_res['ppl']:.4f}   acc1(next-tok): {ref_res['acc1']:.2f}%")
print(f"FP8   perplexity : {q_res['ppl']:.4f}   acc1(next-tok): {q_res['acc1']:.2f}%")
print(f"ppl delta (FP8-FP32): {q_res['ppl']-ref_res['ppl']:+.4f}")
print("=================================================")
