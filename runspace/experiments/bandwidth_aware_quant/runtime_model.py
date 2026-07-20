"""Pure cycle-model helpers for bandwidth-aware quantization."""

from runspace.experiments.asic_cache_simulation.simulate_cache import optimize_layer_bits


def cache_sizes_with_fp32_reference(cache_sizes):
    """Return unique requested cache sizes with the 0M baseline first."""
    return list(dict.fromkeys([0.0, *cache_sizes]))


def compute_model_runtime(layers_with_stay_status, b_bits, bandwidth=1.0, max_bits=8):
    """Compute residual-aware model runtime and per-layer transfer widths.

    Transferred inputs, weights, and outputs begin at ``max_bits`` and are
    reduced together to a floor of ``b_bits`` while a layer is bandwidth
    limited. Setting ``b_bits == max_bits == 32`` evaluates the FP32 reference
    through the exact same transfer path as quantized candidates.
    """
    total_runtime = 0.0
    prev_stay_on_chip = False
    layer_input_bits = {}
    layer_weight_bits = {}
    layer_output_bits = {}
    layer_residual_input_bits = {}
    layer_need_input_transfer = {}

    for idx, layer in enumerate(layers_with_stay_status):
        stay_on_chip = layer.get("stay_on_chip", False)
        layer_name = layer["name"]
        xin_from_cache = layer.get("xin_from_cache", True)

        need_input_transfer = idx == 0 or not prev_stay_on_chip or not xin_from_cache
        layer_need_input_transfer[layer_name] = need_input_transfer
        need_output_transfer = not stay_on_chip
        residual_input_stream_elems = layer.get("residual_input_stream_elems", 0)
        residual_output_elems = layer.get("residual_output_elems", 0)
        fixed_transfers = []
        forced_bits = {}
        residual_output_uses_main_stream = (
            residual_output_elems > 0
            and need_output_transfer
            and residual_output_elems == layer.get("output_elems", 0)
        )

        if residual_output_elems > 0:
            if residual_output_uses_main_stream:
                forced_bits["output"] = b_bits
            else:
                fixed_transfers.append(
                    {
                        "name": "residual_output",
                        "elems": residual_output_elems,
                        "bits": b_bits,
                    }
                )
        if residual_input_stream_elems > 0:
            fixed_transfers.append(
                {
                    "name": "residual_input",
                    "elems": residual_input_stream_elems,
                    "bits": b_bits,
                }
            )
            layer_residual_input_bits[layer_name] = b_bits

        input_bits, weight_bits, output_bits, cycles = optimize_layer_bits(
            layer,
            bandwidth,
            need_input_transfer,
            True,
            need_output_transfer,
            min_bits=b_bits,
            max_bits=max_bits,
            fixed_transfers=fixed_transfers,
            forced_bits=forced_bits,
        )

        layer_input_bits[layer_name] = input_bits
        layer_weight_bits[layer_name] = weight_bits
        layer_output_bits[layer_name] = output_bits
        total_runtime += cycles
        prev_stay_on_chip = stay_on_chip

    return (
        total_runtime,
        layer_input_bits,
        layer_weight_bits,
        layer_output_bits,
        layer_residual_input_bits,
        layer_need_input_transfer,
    )
