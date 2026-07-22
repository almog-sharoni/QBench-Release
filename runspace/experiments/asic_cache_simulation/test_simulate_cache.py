import math

import pytest

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
    assert simulate_cache._quant_add_operation_count(layer) == 2
    assert simulate_cache._compute_layer_cycles(layer) == 2 * math.ceil(257 / 128)


def test_binary_quant_add_is_one_elementwise_pass():
    layer = {
        "type": "QuantAdd",
        "output_elems": 257,
        "input_shapes": [(1, 257), (1, 257)],
    }

    assert simulate_cache._quant_add_connection_count(layer) == 2
    assert simulate_cache._quant_add_operation_count(layer) == 1
    assert simulate_cache._compute_layer_cycles(layer) == math.ceil(257 / 128)


def test_optimize_layer_bits_rejects_invalid_runtime_parameters():
    layer = _one_chunk_layer()

    with pytest.raises(ValueError, match="bandwidth"):
        simulate_cache.optimize_layer_bits(layer, 0, True, True, True)
    with pytest.raises(ValueError, match="min_bits"):
        simulate_cache.optimize_layer_bits(
            layer,
            1.0,
            True,
            True,
            True,
            min_bits=9,
            max_bits=8,
        )


def test_build_cache_map_has_fixed_columns_and_zero_for_inactive_residuals():
    layers = [
        {
            "name": "conv1",
            "input_elems": 16,
            "output_elems": 32,
        },
        {
            "name": "conv2",
            "input_elems": 32,
            "output_elems": 16,
        },
        {
            "name": "add1",
            "input_elems": 16,
            "output_elems": 16,
            "residual_inputs": [{"elems": 16, "producer_name": "conv1"}],
        },
        {
            "name": "conv3",
            "input_elems": 16,
            "output_elems": 8,
        },
        {
            "name": "add2",
            "input_elems": 8,
            "output_elems": 8,
            "residual_inputs": [{"elems": 8, "producer_name": "conv3"}],
        },
    ]

    cache_map = simulate_cache.build_cache_map(layers)

    assert cache_map["columns"] == [
        "x_in", "x_out", "total_cache_needed_kb", "residual_0", "residual_1"
    ]
    assert cache_map["rows"] == [
        {"layer": "conv1", "x_in": 0.016, "x_out": 0.032, "total_cache_needed_kb": 0.064, "residual_0": 0.016, "residual_1": 0},
        {"layer": "conv2", "x_in": 0.032, "x_out": 0.016, "total_cache_needed_kb": 0.064, "residual_0": 0.016, "residual_1": 0},
        {"layer": "add1", "x_in": 0.016, "x_out": 0.016, "total_cache_needed_kb": 0.048, "residual_0": 0.016, "residual_1": 0},
        {"layer": "conv3", "x_in": 0.016, "x_out": 0.008, "total_cache_needed_kb": 0.032, "residual_0": 0, "residual_1": 0.008},
        {"layer": "add2", "x_in": 0.008, "x_out": 0.008, "total_cache_needed_kb": 0.024, "residual_0": 0, "residual_1": 0.008},
    ]


def test_build_cache_map_without_residuals_only_has_input_and_output_columns():
    cache_map = simulate_cache.build_cache_map([
        {"name": "linear", "input_elems": 4, "output_elems": 2},
    ])

    assert cache_map["columns"] == ["x_in", "x_out", "total_cache_needed_kb"]
    assert cache_map["rows"] == [
        {"layer": "linear", "x_in": 0.004, "x_out": 0.002, "total_cache_needed_kb": 0.006},
    ]


def test_residual_detection_uses_producer_age_not_operand_position():
    skip = simulate_cache.torch.zeros(4)
    branch = simulate_cache.torch.ones(4)
    producers = {
        id(skip): {"_execution_index": 2},
        id(branch): {"_execution_index": 8},
    }
    is_static = lambda _tensor: False

    assert simulate_cache._residual_input_indices(
        [skip, branch], producers, is_static
    ) == [0]
    assert simulate_cache._residual_input_indices(
        [branch, skip], producers, is_static
    ) == [1]


def test_residual_detection_excludes_static_add_operands():
    activation = simulate_cache.torch.zeros(4)
    positional_embedding = simulate_cache.torch.ones(4)
    producers = {
        id(activation): {"_execution_index": 2},
    }

    residual_indices = simulate_cache._residual_input_indices(
        [activation, positional_embedding],
        producers,
        lambda tensor: tensor is positional_embedding,
    )

    assert residual_indices == []


def test_cache_map_can_keep_layernorm_as_an_explicit_row():
    cache_map = simulate_cache.build_cache_map([
        {"name": "add", "type": "QuantAdd", "input_elems": 8, "output_elems": 8},
        {"name": "ln_1", "type": "QuantLayerNorm", "input_elems": 8, "output_elems": 8},
        {"name": "q_proj", "type": "QuantLinear", "input_elems": 8, "output_elems": 8},
    ])

    assert [row["layer"] for row in cache_map["rows"]] == [
        "add", "ln_1", "q_proj"
    ]


def test_cache_map_auto_detects_qkv_fanout_and_non_adjacent_consumers():
    shared_storage = ("cpu", None, 100)
    q_storage = ("cpu", None, 200)
    k_storage = ("cpu", None, 300)
    v_storage = ("cpu", None, 400)
    layers = [
        {
            "name": "ln_1", "input_elems": 8, "output_elems": 8,
            "output_tensor_id": 10, "output_tensor_storage_key": shared_storage,
        },
        {
            "name": "q_proj", "input_elems": 8, "output_elems": 8,
            "input_tensor_ids": [10], "input_tensor_storage_keys": [shared_storage],
            "output_tensor_id": 20, "output_tensor_storage_key": q_storage,
        },
        {
            "name": "k_proj", "input_elems": 8, "output_elems": 8,
            "input_tensor_ids": [10], "input_tensor_storage_keys": [shared_storage],
            "output_tensor_id": 30, "output_tensor_storage_key": k_storage,
        },
        {
            "name": "v_proj", "input_elems": 8, "output_elems": 8,
            "input_tensor_ids": [10], "input_tensor_storage_keys": [shared_storage],
            "output_tensor_id": 40, "output_tensor_storage_key": v_storage,
        },
        {
            "name": "scaled_dot", "input_elems": 8, "output_elems": 16,
            # Different Python ids simulate reshape/transpose views; storage
            # identity must still resolve these inputs to Q and K.
            "input_tensor_ids": [21, 31],
            "input_tensor_storage_keys": [q_storage, k_storage],
            "output_tensor_id": 50,
            "output_tensor_storage_key": ("cpu", None, 500),
        },
        {
            "name": "softmax", "input_elems": 16, "output_elems": 16,
            "input_tensor_ids": [50],
            "input_tensor_storage_keys": [("cpu", None, 500)],
            "output_tensor_id": 60,
            "output_tensor_storage_key": ("cpu", None, 600),
        },
        {
            "name": "attention", "input_elems": 16, "output_elems": 8,
            "input_tensor_ids": [60, 41],
            "input_tensor_storage_keys": [("cpu", None, 600), v_storage],
            "output_tensor_id": 70,
            "output_tensor_storage_key": ("cpu", None, 700),
        },
    ]

    cache_map = simulate_cache.build_cache_map(layers)
    hold_names = [item["name"] for item in cache_map["held_connections"]]

    assert hold_names == [
        "hold_0_ln_1_to_q_proj_k_proj_v_proj",
        "hold_1_q_proj_to_scaled_dot",
        "hold_2_k_proj_to_scaled_dot",
        "hold_3_v_proj_to_attention",
    ]
    assert cache_map["rows"][4]["hold_1_q_proj_to_scaled_dot"] == 0.008
    assert cache_map["rows"][4]["hold_2_k_proj_to_scaled_dot"] == 0.008
    assert cache_map["rows"][4]["x_in"] == 0
    assert cache_map["rows"][5]["hold_1_q_proj_to_scaled_dot"] == 0
    assert cache_map["rows"][6]["hold_3_v_proj_to_attention"] == 0.008
    assert cache_map["rows"][4]["total_cache_needed_kb"] == 0.04
    assert cache_map["rows"][0]["hold_0_ln_1_to_q_proj_k_proj_v_proj"] == 0
    assert cache_map["rows"][1]["x_in"] == 0


def test_tensor_storage_key_matches_views_not_separate_allocations():
    tensor = simulate_cache.torch.arange(8)
    view = tensor.reshape(2, 4).transpose(0, 1)
    separate = tensor.clone()

    assert simulate_cache._tensor_storage_key(tensor) == simulate_cache._tensor_storage_key(view)
    assert simulate_cache._tensor_storage_key(tensor) != simulate_cache._tensor_storage_key(separate)
