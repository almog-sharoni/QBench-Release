#!/usr/bin/env python3
"""
Generate the SLM model support matrix.

The default mode is static and does not download models. It records what the
current QBench op set can cover and where an architecture-specific Quant*
wrapper is still needed. Use --run-build-probes when the local environment has
the HuggingFace checkpoints/dependencies cached and you want a live build check.
"""

import argparse
import os
import sys
from datetime import datetime, timezone


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RUNSPACE_ROOT = os.path.join(REPO_ROOT, "runspace")
DEFAULT_REPORT = os.path.join(RUNSPACE_ROOT, "src/adapters/SLM_MODEL_SUPPORT.md")


MODEL_SUPPORT = [
    {
        "model": "facebook/opt-125m",
        "family": "OPT",
        "status": "supported",
        "wrappers": ["QuantOPTAttention"],
        "missing": [],
        "notes": "First verified target. Embedding tables are intentionally kept FP/gather-side.",
    },
    {
        "model": "facebook/opt-350m",
        "family": "OPT",
        "status": "supported_if_resources_allow",
        "wrappers": ["QuantOPTAttention"],
        "missing": [],
        "notes": "Same block structure as OPT-125M; runtime may be limited by local memory/cache.",
    },
    {
        "model": "facebook/opt-1.3b",
        "family": "OPT",
        "status": "supported_if_resources_allow",
        "wrappers": ["QuantOPTAttention"],
        "missing": [],
        "notes": "Same wrapper path as smaller OPT models; expected to be resource-bound locally.",
    },
    {
        "model": "Qwen/Qwen2.5-0.5B",
        "family": "Qwen2",
        "status": "unsupported_wrapper_missing",
        "wrappers": [],
        "missing": [
            "QuantQwen2Attention wrapper for RoPE, GQA repeat_kv, qk matmul, softmax, and attn-value matmul",
            "QuantQwen2MLP wrapper for SiLU(gate_proj) * up_proj gated MLP",
            "QuantRMSNorm for Qwen2RMSNorm",
        ],
        "notes": "Most primitives already exist (QuantLinear, QuantMatMul/QuantBMM, QuantSoftmax, QuantSiLU, QuantMul, QuantAdd/QuantCat); the missing work is mainly wrapping Qwen2 blocks so those primitives are actually used.",
    },
    {
        "model": "Qwen/Qwen2.5-1.5B",
        "family": "Qwen2",
        "status": "unsupported_wrapper_missing",
        "wrappers": [],
        "missing": [
            "Same QuantQwen2Attention, QuantQwen2MLP, and QuantRMSNorm gaps as Qwen2.5-0.5B",
        ],
        "notes": "Bigger model in the same family; probe after 0.5B wrapper support is validated.",
    },
    {
        "model": "EleutherAI/pythia-70m-deduped",
        "family": "GPT-NeoX/Pythia",
        "status": "unsupported_wrapper_missing",
        "wrappers": [],
        "missing": [
            "Quant GPT-NeoX/Pythia attention wrapper for rotary q/k and qk/av matmuls",
            "RMSNorm support if the checkpoint uses GPTNeoXLayerNorm-style RMS normalization",
        ],
        "notes": "Linears and GELU-like activation primitives are covered; functional attention/rotary paths need a wrapper.",
    },
    {
        "model": "EleutherAI/pythia-160m-deduped",
        "family": "GPT-NeoX/Pythia",
        "status": "unsupported_wrapper_missing",
        "wrappers": [],
        "missing": [
            "Same GPT-NeoX/Pythia attention and norm wrapper gaps as pythia-70m",
        ],
        "notes": "Use after the smaller Pythia model validates the wrapper path.",
    },
    {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "family": "Llama",
        "status": "unsupported_wrapper_missing",
        "wrappers": [],
        "missing": [
            "Quant Llama attention wrapper for RoPE, grouped-query attention, qk softmax, and av matmul",
            "Quant Llama gated MLP wrapper",
            "QuantRMSNorm",
        ],
        "notes": "Primitives mostly exist, but no Llama-family Quant* block wrappers are registered yet.",
    },
    {
        "model": "gpt2",
        "family": "GPT-2",
        "status": "unsupported_real_op_missing",
        "wrappers": [],
        "missing": [
            "HF Conv1D projection support or a GPT-2 block wrapper that treats Conv1D projections as quantized linears",
        ],
        "notes": "Current recursive replacement only covers nn.Linear projections, so GPT-2 projections remain FP.",
    },
]


def _status_label(status: str) -> str:
    return {
        "supported": "Supported",
        "supported_if_resources_allow": "Supported architecture; resource-gated",
        "unsupported_wrapper_missing": "Unsupported: Quant* wrapper missing",
        "unsupported_real_op_missing": "Unsupported: low-level op missing",
        "probe_failed": "Probe failed",
    }.get(status, status)


def _probe_build(model_name: str) -> str:
    """Try a local quantized model build. This may download unless HF is cached."""
    if RUNSPACE_ROOT not in sys.path:
        sys.path.insert(0, RUNSPACE_ROOT)
    from src.adapters.adapter_factory import create_adapter

    cfg = {
        "model": {"name": model_name, "source": "huggingface"},
        "adapter": {
            "type": "slm",
            "quantized_ops": ["all"],
            "build_quantized": True,
            "input_quantization": True,
            "weight_quantization": True,
            "quantize_first_layer": False,
        },
        "quantization": {
            "format": "fp8_e4m3",
            "mode": "chunk",
            "chunk_size": 128,
            "weight_mode": "chunk",
            "weight_chunk_size": 128,
            "act_mode": "chunk",
            "act_chunk_size": 128,
        },
    }
    adapter = create_adapter(cfg)
    applied = getattr(adapter, "applied_architecture_wrappers", {}) or {}
    quant_count = sum(1 for module in adapter.model.modules() if module.__class__.__name__.startswith("Quant"))
    return f"build ok; quant_modules={quant_count}; wrappers={applied or {}}"


def render_report(run_build_probes: bool = False) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# SLM Model Support Matrix",
        "",
        f"Generated by `runspace/tools/probe_slm_support.py` on {now}.",
        "",
        "This file distinguishes missing low-level ops from missing architecture wrappers. "
        "A wrapper gap means QBench already has the primitive Quant* ops, but the HuggingFace "
        "block must be rewritten/wrapped so those ops are actually used.",
        "",
        "Allowed FP by design: token/position embeddings may remain FP gather-side unless a "
        "separate storage-quantization experiment explicitly targets them.",
        "",
        "| Model | Family | Status | Existing/Missing Quant* wrappers | Missing ops/layers | Notes |",
        "|---|---|---|---|---|---|",
    ]

    for entry in MODEL_SUPPORT:
        wrappers = ", ".join(entry["wrappers"]) if entry["wrappers"] else "none"
        missing = "<br>".join(entry["missing"]) if entry["missing"] else "none"
        notes = entry["notes"]
        if run_build_probes:
            try:
                notes = f"{notes}<br>Live probe: {_probe_build(entry['model'])}"
            except Exception as exc:
                notes = f"{notes}<br>Live probe failed: {type(exc).__name__}: {exc}"
        lines.append(
            f"| `{entry['model']}` | {entry['family']} | {_status_label(entry['status'])} | "
            f"{wrappers} | {missing} | {notes} |"
        )

    lines.extend([
        "",
        "## Current Implementation Target",
        "",
        "- Start with `facebook/opt-125m` for compliance and tiny bandwidth-aware quantization.",
        "- Use `facebook/opt-350m` next when local memory allows; it should reuse `QuantOPTAttention`.",
        "- Do not add new low-level ops in the OPT-first branch. Non-OPT entries stay unsupported until their wrappers/norm ops are implemented.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate SLM model support matrix.")
    parser.add_argument("--output", default=DEFAULT_REPORT, help="Markdown report path.")
    parser.add_argument("--run-build-probes", action="store_true",
                        help="Try local quantized builds. Requires cached HF deps/checkpoints.")
    args = parser.parse_args()

    report = render_report(run_build_probes=args.run_build_probes)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
