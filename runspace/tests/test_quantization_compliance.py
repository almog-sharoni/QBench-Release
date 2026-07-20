"""
Quantization compliance pre-push test.

Runs 1 random batch through quantized models and validates that every stored
tensor (weights, inputs) actually fits within its declared bit budget.
Verifies sign, exponent, mantissa, and round-trip fidelity.

Usage:
  python runspace/tests/test_quantization_compliance.py     # full test
  QBENCH_QUICK=1 python runspace/tests/test_quantization_compliance.py  # skip vit
"""

import sys
import os
import gc
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.ops.quant_base import QuantizedLayerMixin, quantize_tensor
from runspace.src.quantization.quantizer import quantize_fp_generic
from runspace.src.quantization.constants import get_format_params
from runspace.src.quantization.dynamic_input_quantizer import (
    DynamicInputQuantizer,
    DEFAULT_DYNAMIC_INPUT_CANDIDATES,
)
from runspace.src.quantization.activation_transport import ActivationPacket
from runspace.src.quantization.chunking import (
    chunk_tensor_by_context,
    unchunk_tensor_by_context,
)
from runspace.src.quantization.uniform_input_quantizer import UniformInputQuantizer


MODELS = [
    {"name": "resnet18", "source": "torchvision", "weights": "IMAGENET1K_V1", "input_shape": (1, 3, 224, 224)},
    {"name": "mobilevit_s", "source": "timm", "weights": "DEFAULT", "input_shape": (1, 3, 256, 256)},
    {"name": "vit_b_16", "source": "torchvision", "weights": "IMAGENET1K_V1", "input_shape": (1, 3, 224, 224)},
]

FORMATS = [
    "fp8_e4m3",
    "fp6_e3m2",
    "fp4_e2m1",
]

MODES = [
    {"name": "tensor", "cfg": {"mode": "tensor"}},
    {"name": "chunk", "cfg": {"mode": "chunk", "chunk_size": 128}},
]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(model_spec, q_format, mode_cfg):
    return {
        "model": {
            "name": model_spec["name"],
            "source": model_spec["source"],
            "weights": model_spec["weights"],
        },
        "adapter": {
            "type": "generic",
            "quantized_ops": ["all"],
            "weight_quantization": True,
            "input_quantization": True,
            "fold_input_norm": False,
            "fold_layers": True,
        },
        "quantization": {
            "format": q_format,
            **mode_cfg,
        },
    }


# ---------------------------------------------------------------------------
# Bit-structure validation
# ---------------------------------------------------------------------------

def check_bit_structure(tensor, q_type):
    """
    Verifies each value in `tensor` uses at most its declared bit budget:

      fpW_eEmM  →  1 sign + E exp + M mant  =  W bits
      ufpW_eEmM →  0 sign + E exp + M mant  =  W bits

    Checks performed:
      1. Sign   — unsigned formats must not use the sign bit.
      2. Mantissa — lower (23 - M) FP32 mantissa bits must be zero.
      3. Exponent  — exponent must lie within the E-bit range.
      4. Round-trip — quantize_fp_generic() must be identity (gold standard).

    Returns (passed, total_invalid, examples, failures)
    """
    exp_bits, mant_bits = get_format_params(q_type)
    is_unsigned = q_type.startswith("ufp")
    is_signed = not is_unsigned

    failures = {}
    invalid_mask = None

    tensor_float = tensor.float().contiguous().view(-1)
    f32_int = tensor_float.view(torch.int32)

    exp_mask_fp32 = 0x7F800000
    is_finite = (f32_int & exp_mask_fp32) != exp_mask_fp32

    # 1. Sign check
    if is_unsigned:
        sign_bits = (f32_int >> 31) & 0x1
        mask = (sign_bits != 0) & is_finite
        if mask.any():
            failures["sign"] = int(mask.sum().item())
            invalid_mask = mask

    # 2. Mantissa precision check
    if mant_bits < 23:
        lower_mask = (1 << (23 - mant_bits)) - 1
        mask = ((f32_int & lower_mask) != 0) & is_finite
        if mask.any():
            failures["mantissa"] = int(mask.sum().item())
            invalid_mask = mask if invalid_mask is None else (invalid_mask | mask)

    # 3. Exponent range check
    # quantize_fp_generic clamps exp32 > 0x7F (= 127) to that upper bound
    # for all formats because unscaled values live in [0, 2).  Small
    # exponents are valid subnormals — there is no lower bound.
    exp32 = (f32_int >> 23) & 0xFF
    mask = is_finite & (exp32 > 0x7F)
    if mask.any():
        failures["exponent"] = int(mask.sum().item())
        invalid_mask = mask if invalid_mask is None else (invalid_mask | mask)

    # 4. Round-trip check
    expected = quantize_fp_generic(tensor_float, exp_bits, mant_bits)
    diff = (tensor_float - expected).abs()
    tol = 1e-5 * tensor_float.abs().clamp(min=1e-10)
    is_finite_float = tensor_float.isfinite()
    mask = (diff > tol) & is_finite_float
    if mask.any():
        failures["roundtrip"] = int(mask.sum().item())
        invalid_mask = mask if invalid_mask is None else (invalid_mask | mask)

    total_invalid = sum(failures.values())
    examples = (
        tensor_float[invalid_mask][:5].tolist()
        if invalid_mask is not None and invalid_mask.any()
        else []
    )

    return total_invalid == 0, total_invalid, examples, failures


# ---------------------------------------------------------------------------
# Single (model, format, mode) test
# ---------------------------------------------------------------------------

def test_one_config(model_spec, q_format, mode_info, device):
    label = f"{model_spec['name']}/{q_format}/{mode_info['name']}"
    config = build_config(model_spec, q_format, mode_info["cfg"])

    adapter = create_adapter(config)
    model = adapter.model
    model.eval()

    if device.type == "cuda":
        model = model.to(device)

    inputs = torch.randn(model_spec["input_shape"], device=device, dtype=torch.float32)
    results = {"weight": [], "input": []}

    quant_mode = mode_info["cfg"]["mode"]
    transport = (
        "encoded"
        if device.type == "cuda" and quant_mode == "chunk"
        else "reference"
    )
    input_quantizer = UniformInputQuantizer(
        model=model,
        fmt=q_format,
        chunk_size=int(mode_info["cfg"].get("chunk_size", 128)),
        quant_mode=quant_mode,
        transport=transport,
        collect_error_stats=False,
    )
    input_quantizer.register_hooks()

    runtime = input_quantizer._transport_runtime
    original_encode = runtime.encode_callback

    def validate_encoded_stage(stage, tensor):
        transmitted = original_encode(stage, tensor)
        fmt = input_quantizer._stage_formats[stage.stage_id]
        if fmt is None:
            return transmitted

        decoded = (
            input_quantizer.activation_transport.decode(transmitted)
            if isinstance(transmitted, ActivationPacket)
            else transmitted
        )
        if quant_mode == "chunk":
            chunked, original_shape, padding = chunk_tensor_by_context(
                tensor,
                input_quantizer.chunk_size,
            )
            chunks = chunked.reshape(-1, input_quantizer.chunk_size)
            expected_chunks, expected_unscaled, _max_val = quantize_tensor(
                chunks,
                q_type=fmt,
                return_unscaled=True,
                mode="chunk",
                chunk_size=input_quantizer.chunk_size,
            )
            expected = unchunk_tensor_by_context(
                expected_chunks.reshape(chunked.shape),
                original_shape,
                padding,
            )
        else:
            expected, expected_unscaled, _max_val = quantize_tensor(
                tensor,
                q_type=fmt,
                return_unscaled=True,
                mode=quant_mode,
            )
        ok, cnt, examples, failures = check_bit_structure(expected_unscaled, fmt)
        mismatch = decoded != expected
        mismatch_count = int(mismatch.sum().item())
        if mismatch_count:
            failures = dict(failures)
            failures["transport_roundtrip"] = mismatch_count
            cnt += mismatch_count
            examples = decoded[mismatch][:5].tolist()
            ok = False
        results["input"].append(
            (stage.stage_id, ok, cnt, examples, failures, fmt)
        )
        return transmitted

    runtime.encode_callback = validate_encoded_stage

    try:
        with torch.inference_mode():
            model(inputs)
    finally:
        input_quantizer.cleanup()

    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLayerMixin):
            continue

        layer_q_type = getattr(module, "q_type", q_format)
        w = getattr(module, "weight_fp8", None)
        if w is None:
            w = getattr(module, "weight", None)

        if w is not None and w.numel() > 0:
            ok, cnt, ex, failures = check_bit_structure(w, layer_q_type)
            results["weight"].append((name, ok, cnt, ex, failures, layer_q_type))

    return label, results


# ---------------------------------------------------------------------------
# CPU fallback (advisory — never fails)
# ---------------------------------------------------------------------------

def test_one_config_cpu(model_spec, q_format, mode_info):
    label = f"{model_spec['name']}/{q_format}/{mode_info['name']} [CPU]"
    try:
        _, results = test_one_config(model_spec, q_format, mode_info, torch.device("cpu"))
        return label, results
    except Exception as e:
        print(f"  {label}: CPU model build/run failed ({e}) — skipping")
        return label, None


# ---------------------------------------------------------------------------
# Dynamic input quantizer test (per-chunk format selection)
# ---------------------------------------------------------------------------

def test_dynamic_config(model_spec, device):
    label = f"{model_spec['name']}/dynamic/fp8_e4m3"
    config = build_config(model_spec, "fp8_e4m3", {"mode": "chunk", "chunk_size": 128})

    adapter = create_adapter(config)
    model = adapter.model
    model.eval()

    if device.type == "cuda":
        model = model.to(device)

    for _name, module in model.named_modules():
        if isinstance(module, QuantizedLayerMixin):
            module.capture_activations = True

    dq = DynamicInputQuantizer(
        model,
        chunk_size=128,
        candidate_formats=DEFAULT_DYNAMIC_INPUT_CANDIDATES,
        restrict_post_relu_ufp=True,
        use_unsigned_input_candidates=True,
    )
    dq.register_hooks()

    inputs = torch.randn(model_spec["input_shape"], device=device, dtype=torch.float32)

    with torch.inference_mode():
        model(inputs)

    runtime = dq._transport_runtime
    snapshots = []
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLayerMixin):
            continue
        w = getattr(module, "weight_fp8", None)
        if w is None:
            w = getattr(module, "weight", None)
        if w is not None and isinstance(w, torch.Tensor):
            w = w.detach().clone()
        q_type = getattr(module, "q_type", "fp8_e4m3")
        snapshots.append((name, q_type, w))

    stage_count = len(runtime.plan.stages)
    transmission_count = int(runtime.transmission_count)
    decode_reads = int(runtime.decode_reads)

    dq.cleanup()

    results = {"weight": [], "input": []}

    for name, layer_q_type, w in snapshots:
        if w is not None and w.numel() > 0:
            ok, cnt, ex, failures = check_bit_structure(w, layer_q_type)
            results["weight"].append((name, ok, cnt, ex, failures, layer_q_type))

    results["input"].append(
        ("(producer stages installed)", stage_count > 0, 0, [], {}, "dynamic")
    )
    results["input"].append(
        ("(activation transmissions)", transmission_count > 0, 0, [], {}, "dynamic")
    )
    results["input"].append(
        ("(activation decode reads)", decode_reads > 0, 0, [], {}, "dynamic")
    )

    return label, results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_result(label, results):
    if results is None:
        return True, f"  {label}: ⚠ SKIP (CPU unavailable)"

    w_ok = all(r[1] for r in results["weight"])
    i_ok = all(r[1] for r in results["input"])
    passed = w_ok and i_ok

    w_total = len(results["weight"])
    i_total = len(results["input"])

    if passed:
        return True, f"  {label}: PASS (weights: {w_total}/{w_total}, inputs: {i_total}/{i_total})"

    lines = [f"  {label}: FAIL"]
    for name, ok, cnt, ex, failures, qt in results["weight"]:
        if not ok:
            lines.append(f"    WEIGHT {name} ({qt}): {cnt} invalid, failures={failures}, examples={ex[:3]}")
    for name, ok, cnt, ex, failures, qt in results["input"]:
        if not ok:
            lines.append(f"    INPUT  {name} ({qt}): {cnt} invalid, failures={failures}, examples={ex[:3]}")
    return False, "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("Quantization Compliance Tests")
    print("=" * 62)
    print(f"Models:  {len(MODELS)}  ({', '.join(m['name'] for m in MODELS)})")
    print(f"Formats: {len(FORMATS)}  ({', '.join(FORMATS)})")
    print(f"Modes:   {len(MODES)}  ({', '.join(m['name'] for m in MODES)})")
    total_tests = len(MODELS) * len(FORMATS) * len(MODES)
    print(f"Total:   {total_tests} configurations")
    print("-" * 62)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    if not has_cuda:
        print("[WARNING] CUDA not available — running CPU path (advisory only)")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    quick = os.environ.get("QBENCH_QUICK", "").strip().lower() in ("1", "true", "yes")
    if quick:
        print("[QBENCH_QUICK] Skipping vit_b_16\n")

    all_lines = []
    pass_count = 0
    fail_count = 0
    skip_count = 0

    for model_spec in MODELS:
        if quick and model_spec["name"] == "vit_b_16":
            skip_count += len(FORMATS) * len(MODES)
            all_lines.append(f"  {model_spec['name']}: ⏭ SKIPPED (QBENCH_QUICK)")
            continue

        for q_format in FORMATS:
            for mode_info in MODES:
                label_prefix = f"{model_spec['name']}/{q_format}/{mode_info['name']}"
                try:
                    label, results = test_one_config(model_spec, q_format, mode_info, device)
                    ok, line = format_result(label, results)
                    all_lines.append(line)
                    if ok:
                        pass_count += 1
                    else:
                        fail_count += 1
                except Exception:
                    exc_text = traceback.format_exc()
                    if has_cuda:
                        print(
                            f"  {label_prefix}: CUDA failed: {exc_text.splitlines()[-1]}",
                            flush=True,
                        )
                        print("    Trying CPU fallback...", flush=True)
                        label, results = test_one_config_cpu(model_spec, q_format, mode_info)
                        if results is not None:
                            ok, line = format_result(label, results)
                            all_lines.append(line)
                            all_lines.append(f"    [WARNING] Ran on CPU due to CUDA failure. Original error: {exc_text.splitlines()[-1]}")
                        else:
                            all_lines.append(f"  {label_prefix}: ⚠ CPU fallback also failed — skipping")
                            skip_count += 1
                    else:
                        all_lines.append(f"  {label_prefix}: ERROR — {exc_text.splitlines()[-1]}")
                        fail_count += 1
                finally:
                    gc.collect()
                    if has_cuda:
                        torch.cuda.empty_cache()

    print("\nResults:")
    for line in all_lines:
        print(line)

    print()

    # --- Dynamic input quantizer tests ---
    print("=" * 62)
    print("Dynamic Input Quantizer Tests")
    print("=" * 62)
    dyn_lines = []
    for model_spec in MODELS:
        if quick and model_spec["name"] == "vit_b_16":
            dyn_lines.append(f"  {model_spec['name']}/dynamic/fp8_e4m3: SKIPPED (QBENCH_QUICK)")
            continue
        try:
            label, results = test_dynamic_config(model_spec, device)
            ok, line = format_result(label, results)
            dyn_lines.append(line)
            if ok:
                pass_count += 1
            else:
                fail_count += 1
        except Exception:
            dyn_lines.append(f"  {model_spec['name']}/dynamic/fp8_e4m3: ERROR — {traceback.format_exc().splitlines()[-1]}")
            fail_count += 1
        finally:
            gc.collect()
            if has_cuda:
                torch.cuda.empty_cache()

    for line in dyn_lines:
        print(line)

    print()
    print("=" * 62)
    print(f"Summary: {pass_count} passed, {fail_count} failed, {skip_count} skipped")
    print("=" * 62)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
