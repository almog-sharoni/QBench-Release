import math

from runspace.experiments.asic_cache_simulation import simulate_cache


def _one_chunk_layer():
    return {
        "type": "Linear",
        "input_elems": 128,
        "weight_elems": 128,
        "output_elems": 128,
    }


def _patch_compute(monkeypatch, cycles):
    monkeypatch.setattr(
        simulate_cache,
        "_compute_layer_cycles",
        lambda _layer: float(cycles),
    )


def test_optimize_layer_bits_reduces_all_transferred_parts_together(monkeypatch):
    _patch_compute(monkeypatch, 350)

    in_b, w_b, out_b, cycles = simulate_cache.optimize_layer_bits(
        _one_chunk_layer(),
        bandwidth=1.0,
        need_input_transfer=True,
        need_weight_transfer=True,
        need_output_transfer=True,
        min_bits=3,
        max_bits=8,
    )

    assert (in_b, w_b, out_b) == (7, 7, 7)
    assert cycles == 350


def test_optimize_layer_bits_keeps_non_transferred_parts_at_max_bits(monkeypatch):
    _patch_compute(monkeypatch, 100)

    in_b, w_b, out_b, cycles = simulate_cache.optimize_layer_bits(
        _one_chunk_layer(),
        bandwidth=1.0,
        need_input_transfer=False,
        need_weight_transfer=True,
        need_output_transfer=False,
        min_bits=3,
        max_bits=8,
    )

    assert (in_b, w_b, out_b) == (8, 6, 8)
    assert cycles == 100


def test_optimize_layer_bits_stops_at_min_bits_when_still_bandwidth_limited(monkeypatch):
    _patch_compute(monkeypatch, 100)

    in_b, w_b, out_b, cycles = simulate_cache.optimize_layer_bits(
        _one_chunk_layer(),
        bandwidth=1.0,
        need_input_transfer=True,
        need_weight_transfer=True,
        need_output_transfer=True,
        min_bits=3,
        max_bits=8,
    )

    expected_transfer_cycles = 3 * (16 * 3)
    assert (in_b, w_b, out_b) == (3, 3, 3)
    assert cycles == expected_transfer_cycles


def test_optimize_layer_bits_honors_forced_component_bits(monkeypatch):
    _patch_compute(monkeypatch, 250)

    in_b, w_b, out_b, cycles = simulate_cache.optimize_layer_bits(
        _one_chunk_layer(),
        bandwidth=1.0,
        need_input_transfer=True,
        need_weight_transfer=True,
        need_output_transfer=True,
        min_bits=3,
        max_bits=8,
        forced_bits={"output": 3},
    )

    assert (in_b, w_b, out_b) == (6, 6, 3)
    assert cycles == 250


def test_optimize_layer_bits_counts_fixed_transfers_in_the_decision(monkeypatch):
    _patch_compute(monkeypatch, 160)

    in_b, w_b, out_b, cycles = simulate_cache.optimize_layer_bits(
        _one_chunk_layer(),
        bandwidth=1.0,
        need_input_transfer=False,
        need_weight_transfer=True,
        need_output_transfer=False,
        min_bits=3,
        max_bits=8,
        fixed_transfers=[{"name": "residual_input", "elems": 128, "bits": 3}],
    )

    assert (in_b, w_b, out_b) == (8, 7, 8)
    assert cycles == 160


def test_quant_add_compute_scales_with_number_of_inputs():
    layer = {
        "type": "QuantAdd",
        "output_elems": 257,
        "input_shapes": [(1, 257), (1, 257), (1, 257)],
    }

    assert simulate_cache._quant_add_connection_count(layer) == 3
    assert simulate_cache._compute_layer_cycles(layer) == 3 * math.ceil(257 / 128)
