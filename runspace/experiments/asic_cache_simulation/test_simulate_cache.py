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


def _pipeline_layer(
    name, layer_type, producer_indices, tensor_id, storage_key, elems=64,
):
    return {
        'name': name,
        'type': layer_type,
        'weight_elems': 0 if 'ReLU' in layer_type else 1,
        'input_elems': elems,
        'output_elems': elems,
        'input_shapes': [(1, elems)],
        'output_shape': (1, elems),
        'input_producer_layer_indices': producer_indices,
        'input_edges': [
            {
                'input_index': input_index,
                'producer_layer_index': producer_index,
                'elements': elems,
                'is_model_state': False,
            }
            for input_index, producer_index in enumerate(producer_indices)
        ],
        'output_tensor_id': tensor_id,
        'output_tensor_storage_key': storage_key,
    }


def test_pipeline_activation_fusion_rewires_consumers_and_tensor_identity():
    conv1 = _pipeline_layer('conv1', 'QuantConv2d', [None], 10, ('cpu', None, 10))
    relu = _pipeline_layer('relu', 'QuantReLU', [0], 20, ('cpu', None, 20))
    conv2 = _pipeline_layer('conv2', 'QuantConv2d', [1], 30, ('cpu', None, 30))
    conv2['residual_inputs'] = [{
        'producer_layer_index': 1,
        'producer_name': 'relu',
        'elems': 64,
    }]

    fused = simulate_cache.fuse_pipeline_activations([conv1, relu, conv2])

    assert [layer['name'] for layer in fused] == ['conv1', 'conv2']
    assert fused[0]['fused_activations'] == [
        {'name': 'relu', 'type': 'QuantReLU'}
    ]
    assert fused[0]['output_tensor_id'] == 20
    assert fused[0]['output_tensor_storage_key'] == ('cpu', None, 20)
    assert fused[1]['input_producer_layer_indices'] == [0]
    assert fused[1]['input_edges'][0]['producer_layer_index'] == 0
    assert fused[1]['input_edges'][0]['producer_name'] == 'conv1'
    assert fused[1]['residual_inputs'][0]['producer_layer_index'] == 0
    assert fused[1]['residual_inputs'][0]['producer_name'] == 'conv1'

    cache_map = simulate_cache.build_cache_map(fused)
    assert [row['layer'] for row in cache_map['rows']] == ['conv1', 'conv2']


def test_rule_aware_conv_workspace_reuses_final_input_and_adds_loopback():
    workspace = simulate_cache._rule_aware_workspace(
        {
            'type': 'QuantConv2d',
            'jump_back_size_in_banks': 100,
        },
        input_banks=2,
        output_banks=7,
        bank_size=100,
        input_is_reusable=True,
    )

    assert workspace == {
        'rule': 'conv_output_dominated',
        'reuses_input': True,
        'overlap_banks': 2,
        'shared_banks': 2,
        'pipeline_boundary_banks': 1,
        'jumpback_banks': 1,
        'overhead_banks': 2,
    }


def test_rule_aware_workspace_does_not_overwrite_live_fanout_input():
    workspace = simulate_cache._rule_aware_workspace(
        {
            'type': 'QuantConv2d',
            'jump_back_size_in_banks': 100,
        },
        input_banks=2,
        output_banks=7,
        bank_size=100,
        input_is_reusable=False,
    )

    assert workspace['rule'] == 'global_fit'
    assert workspace['reuses_input'] is False
    assert workspace['overlap_banks'] == 0
    assert workspace['overhead_banks'] == 0


def test_rule_aware_add_can_overwrite_a_final_use_input():
    workspace = simulate_cache._rule_aware_workspace(
        {'type': 'QuantAdd'},
        input_banks=7,
        output_banks=7,
        bank_size=100,
        input_is_reusable=True,
    )

    assert workspace['rule'] == 'residual'
    assert workspace['shared_banks'] == 7
    assert workspace['overhead_banks'] == 0


def test_greedy_streaming_reuses_an_alternate_resident_add_input():
    result = simulate_cache._greedy_stream_connections(
        base_banks_by_layer=[12, 7],
        connection_plans=[
            {
                'name': 'main',
                'bank_count': 7,
                'resident_layer_indices': [0, 1],
                'stream_layer_indices': [0, 1],
            },
            {
                'name': 'skip',
                'bank_count': 7,
                'resident_layer_indices': [1],
                'stream_layer_indices': [1],
            },
        ],
        capacity_banks=16,
        streaming_banks=2,
        rule_workspaces=[
            {},
            {
                'overhead_banks': 0,
                'reuse_candidates': [
                    {'connection_index': 0, 'shared_banks': 7},
                    {'connection_index': 1, 'shared_banks': 7},
                ],
            },
        ],
    )

    assert result['streamed_connection_indices'] == [0]
    assert result['total_banks_by_layer'] == [14, 9]


def test_lifetime_plan_evicts_background_before_rule_workspace_overflow():
    layers = [
        {
            'name': 'main', 'type': 'QuantLinear',
            'input_elems': 128, 'output_elems': 7 * 128,
            'weight_elems': 0,
            'input_edges': [{
                'input_index': 0, 'elements': 128,
                'producer_layer_index': None, 'is_model_state': False,
            }],
        },
        {
            'name': 'downsample', 'type': 'QuantConv2d',
            'input_elems': 2 * 128, 'output_elems': 7 * 128,
            'weight_elems': 1, 'jump_back_size_in_banks': 128,
            'input_edges': [{
                'input_index': 0, 'elements': 2 * 128,
                'producer_layer_index': None, 'is_model_state': False,
            }],
        },
        {
            'name': 'add', 'type': 'QuantAdd',
            'input_elems': 7 * 128, 'output_elems': 7 * 128,
            'weight_elems': 0,
            'input_edges': [
                {
                    'input_index': 0, 'elements': 7 * 128,
                    'producer_layer_index': 0, 'is_model_state': False,
                },
                {
                    'input_index': 1, 'elements': 7 * 128,
                    'producer_layer_index': 1, 'is_model_state': False,
                },
            ],
        },
    ]

    plan = simulate_cache._producer_consumer_cache_plan(
        layers, cache_elements=16 * 128, bank_size=128, metadata_bits=0,
    )

    downsample = plan['steps'][1]
    assert downsample['logical_cache_required_elems'] == 18 * 128
    assert downsample['execution_cache_required_elems'] == 11 * 128
    assert downsample['evicted_producer_indices'] == [0]
    assert plan['steps'][0]['output_spilled'] is True


def test_pipeline_activation_fusion_preserves_activation_on_raw_fanout():
    conv = _pipeline_layer('conv', 'QuantConv2d', [None], 10, ('cpu', None, 10))
    relu = _pipeline_layer('relu', 'QuantReLU', [0], 20, ('cpu', None, 20))
    raw_consumer = _pipeline_layer(
        'raw_consumer', 'QuantConv2d', [0], 30, ('cpu', None, 30)
    )

    fused = simulate_cache.fuse_pipeline_activations([conv, relu, raw_consumer])

    assert [layer['name'] for layer in fused] == ['conv', 'relu', 'raw_consumer']


@pytest.mark.parametrize('layer_type', [
    'QuantLayerNorm', 'LayerNorm', 'QuantSoftmax', 'Softmax',
])
def test_pipeline_activation_fusion_keeps_norm_and_softmax_explicit(layer_type):
    producer = _pipeline_layer('producer', 'QuantLinear', [None], 10, ('cpu', None, 10))
    operation = _pipeline_layer('operation', layer_type, [0], 20, ('cpu', None, 20))
    operation['weight_elems'] = 0

    fused = simulate_cache.fuse_pipeline_activations([producer, operation])

    assert [layer['name'] for layer in fused] == ['producer', 'operation']


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


def test_residual_detection_uses_graph_depth_not_operand_position():
    skip = simulate_cache.torch.zeros(4)
    branch = simulate_cache.torch.ones(4)
    producers = {
        # A projected ResNet skip can execute after the deeper main branch.
        id(skip): {"_execution_index": 9, "_graph_depth": 4},
        id(branch): {"_execution_index": 8, "_graph_depth": 9},
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


def test_layout_lineage_repair_bridges_only_an_unconsumed_previous_output():
    layers = [
        {
            'name': 'matmul', 'output_elems': 8,
            'input_producer_layer_indices': [None, None],
        },
        {
            'name': 'out_proj', 'input_elems': 8,
            'input_producer_layer_indices': [None],
        },
        {
            'name': 'other', 'output_elems': 4,
            'input_producer_layer_indices': [1],
        },
        {
            'name': 'unrelated', 'input_elems': 4,
            'input_producer_layer_indices': [None],
        },
        {
            'name': 'existing_consumer', 'input_elems': 4,
            'input_producer_layer_indices': [2],
        },
    ]

    repaired = simulate_cache._repair_unrecorded_layout_lineage(layers)

    assert repaired == 1
    assert layers[1]['input_producer_layer_indices'] == [0]
    assert layers[1]['input_lineage_repaired_from'] == 0
    assert layers[3]['input_producer_layer_indices'] == [None]


def test_prior_input_lineage_repairs_later_layout_residual_use():
    copied_storage = ('cpu', None, 200)
    layers = [
        {
            'name': 'conv_1x1', 'output_elems': 64,
            'input_producer_layer_indices': [],
        },
        {
            'name': 'norm1', 'input_producer_layer_indices': [0],
            'input_edges': [{
                'input_index': 0, 'elements': 64, 'shape': (4, 4, 4),
                'tensor_id': 20, 'storage_key': copied_storage,
                'producer_layer_index': 0, 'producer_name': 'conv_1x1',
                'is_model_state': False,
            }],
        },
        {
            'name': 'attn_proj', 'input_producer_layer_indices': [1],
            'input_edges': [{
                'input_index': 0, 'elements': 64,
                'producer_layer_index': 1, 'producer_name': 'norm1',
                'is_model_state': False,
            }],
        },
        {
            'name': 'add', 'type': 'QuantAdd',
            'input_producer_layer_indices': [None, 2],
            'input_edges': [
                {
                    'input_index': 0, 'elements': 64, 'shape': (4, 4, 4),
                    'tensor_id': 20, 'storage_key': copied_storage,
                    'producer_layer_index': None, 'producer_name': None,
                    'is_model_state': False,
                },
                {
                    'input_index': 1, 'elements': 64,
                    'producer_layer_index': 2, 'producer_name': 'attn_proj',
                    'is_model_state': False,
                },
            ],
        },
    ]

    repaired = simulate_cache._repair_lineage_from_prior_inputs(layers)

    assert repaired == 1
    assert layers[3]['input_producer_layer_indices'] == [0, 2]
    assert layers[3]['input_edges'][0]['producer_layer_index'] == 0
    assert layers[3]['input_edges'][0]['lineage_repaired_from_prior_input'] is True


def test_cache_map_merges_copied_layout_residual_with_its_producer_lifetime():
    layers = [
        {
            'name': 'conv_1x1', 'type': 'QuantConv2d',
            'input_elems': 32, 'output_elems': 64,
            'input_producer_layer_indices': [None],
            'output_tensor_id': 10,
            'output_tensor_storage_key': ('cpu', None, 100),
        },
        {
            'name': 'norm1', 'type': 'QuantLayerNorm',
            'input_elems': 64, 'output_elems': 64,
            'input_producer_layer_indices': [0],
            'output_tensor_id': 30,
            'output_tensor_storage_key': ('cpu', None, 300),
        },
        {
            'name': 'attn_proj', 'type': 'QuantLinear',
            'input_elems': 64, 'output_elems': 64,
            'input_producer_layer_indices': [1],
            'output_tensor_id': 40,
            'output_tensor_storage_key': ('cpu', None, 400),
        },
        {
            'name': 'add', 'type': 'QuantAdd',
            'input_elems': 64, 'output_elems': 64,
            'input_producer_layer_indices': [0, 2],
            'output_tensor_id': 50,
            'output_tensor_storage_key': ('cpu', None, 500),
            'residual_inputs': [{
                'elems': 64,
                # The layout copy has different runtime identities from the
                # Conv2d output, but post-trace lineage resolved its producer.
                'tensor_id': 20,
                'storage_key': ('cpu', None, 200),
                'producer_layer_index': 0,
                'producer_name': 'conv_1x1',
            }],
        },
    ]

    cache_map = simulate_cache.build_cache_map(layers)

    assert len(cache_map['residual_connections']) == 1
    assert cache_map['held_connections'] == []
    assert cache_map['residual_connections'][0]['producer_layer_index'] == 0
    assert [row['residual_0'] for row in cache_map['rows']] == [0, 0.064, 0.064, 0.064]


@pytest.mark.parametrize(
    ("layers", "expected_main", "expected_residual"),
    [
        (
            [
                {"name": "skip", "type": "QuantAdd", "input_producer_layer_indices": []},
                {"name": "qkv", "type": "QuantLinear", "input_producer_layer_indices": [0]},
                {"name": "attention", "type": "QuantMatMul", "input_producer_layer_indices": [1]},
                {"name": "out_proj", "type": "QuantLinear", "input_producer_layer_indices": [2]},
                {
                    "name": "residual_add", "type": "QuantAdd",
                    "input_producer_layer_indices": [3, 0],
                    "input_edges": [
                        {"producer_layer_index": 3, "elements": 8, "tensor_id": 30},
                        {"producer_layer_index": 0, "elements": 8, "tensor_id": 10},
                    ],
                },
            ],
            "out_proj",
            "skip",
        ),
        (
            [
                {"name": "block_input", "type": "QuantConv2d", "input_producer_layer_indices": []},
                {"name": "conv1", "type": "QuantConv2d", "input_producer_layer_indices": [0]},
                {"name": "conv2", "type": "QuantConv2d", "input_producer_layer_indices": [1]},
                {"name": "conv3", "type": "QuantConv2d", "input_producer_layer_indices": [2]},
                # A projected skip executes after conv3 but has a shallower path.
                {"name": "downsample", "type": "QuantConv2d", "input_producer_layer_indices": [0]},
                {
                    "name": "add", "type": "QuantAdd",
                    "input_producer_layer_indices": [3, 4],
                    "input_edges": [
                        {"producer_layer_index": 3, "elements": 8, "tensor_id": 30},
                        {"producer_layer_index": 4, "elements": 8, "tensor_id": 40},
                    ],
                },
            ],
            "conv3",
            "downsample",
        ),
    ],
)
def test_residual_metadata_is_rebuilt_from_completed_graph(
    layers, expected_main, expected_residual
):
    simulate_cache._rebuild_residual_metadata_from_edges(layers)

    add = layers[-1]
    residual = add["residual_inputs"]
    assert residual[0]["producer_name"] == expected_residual
    producer_by_name = {layer["name"]: layer for layer in layers}
    assert producer_by_name[expected_residual]["residual_output_consumers"] == [
        add["name"]
    ]
    assert "residual_output_consumers" not in producer_by_name[expected_main]


def _fanout_lifetime_layers():
    return [
        {
            "name": "producer", "type": "QuantLinear", "output_elems": 256,
            "input_edges": [{
                "input_index": 0, "elements": 256,
                "producer_layer_index": None, "is_model_state": False,
            }],
        },
        {
            "name": "near_consumer", "type": "QuantLinear", "output_elems": 128,
            "input_edges": [{
                "input_index": 0, "elements": 256,
                "producer_layer_index": 0, "producer_name": "producer",
            }],
        },
        {
            "name": "middle", "type": "QuantLinear", "output_elems": 128,
            "input_edges": [{
                "input_index": 0, "elements": 128,
                "producer_layer_index": 1, "producer_name": "near_consumer",
            }],
        },
        {
            "name": "far_add", "type": "QuantAdd", "output_elems": 128,
            "input_edges": [
                {
                    "input_index": 0, "elements": 128,
                    "producer_layer_index": 2, "producer_name": "middle",
                    "tensor_id": 30,
                },
                {
                    "input_index": 1, "elements": 256,
                    "producer_layer_index": 0, "producer_name": "producer",
                    "tensor_id": 10,
                },
            ],
            "residual_inputs": [{
                "elements": 256, "producer_name": "producer", "tensor_id": 10,
            }],
        },
    ]


def test_producer_consumer_cache_plan_tracks_fanout_without_double_counting_xin():
    plan = simulate_cache._producer_consumer_cache_plan(
        _fanout_lifetime_layers(), cache_elements=512, bank_size=128,
        metadata_bits=0,
    )

    assert plan["last_consumer_by_producer"][0] == 3
    assert [
        step["logical_cache_required_elems"] for step in plan["steps"]
    ] == [512, 384, 512, 384]
    assert plan["steps"][3]["cache_rule"] == "residual"
    assert plan["steps"][3]["input_output_overlap_elems"] == 128
    # The producer is still resident at both consumers, so it is never reloaded.
    assert plan["steps"][1]["input_transfer_elems"] == 0
    assert plan["steps"][3]["input_transfer_elems"] == 0
    assert plan["steps"][3]["residual_input_transfer_elems"] == 0
    assert plan["steps"][0]["stay_on_chip"] is True


def test_producer_consumer_cache_plan_spills_farthest_next_use_and_reloads_it():
    plan = simulate_cache._producer_consumer_cache_plan(
        _fanout_lifetime_layers(), cache_elements=256, bank_size=128,
        metadata_bits=0,
    )

    # At layer 1, producer 0 is needed later than producer 1, so it is evicted.
    assert plan["steps"][1]["evicted_producer_indices"] == [0]
    assert plan["steps"][0]["stay_on_chip"] is False
    assert plan["steps"][0]["output_evicted_at"] == 1
    assert plan["steps"][3]["input_transfer_producer_indices"] == [0]
    assert plan["steps"][3]["input_transfer_elems"] == 256
    assert plan["steps"][3]["residual_input_transfer_elems"] == 256


def test_producer_consumer_cache_plan_treats_model_state_as_transfer_not_lifetime():
    layers = [{
        "name": "positional_add", "type": "QuantAdd", "output_elems": 128,
        "input_edges": [
            {
                "input_index": 0, "elements": 128,
                "producer_layer_index": None, "is_model_state": False,
            },
            {
                "input_index": 1, "elements": 256,
                "producer_layer_index": None, "is_model_state": True,
            },
        ],
    }]

    step = simulate_cache._producer_consumer_cache_plan(
        layers, cache_elements=256, bank_size=128, metadata_bits=0,
    )["steps"][0]

    # Input/output overlap saves one bank; the streamed positional state uses
    # the required two-bank model-data buffer.
    assert step["logical_cache_required_elems"] == 384
    assert step["input_transfer_elems"] == 128
    assert step["model_state_transfer_elems"] == 256


def test_greedy_streaming_prefers_smaller_connection_when_solved_count_ties():
    result = simulate_cache._greedy_stream_connections(
        base_banks_by_layer=[2, 1, 2],
        connection_plans=[
            {
                "name": "first_large", "bank_count": 4,
                "resident_layer_indices": [0, 1],
                "stream_layer_indices": [0, 1],
            },
            {
                "name": "second_small", "bank_count": 3,
                "resident_layer_indices": [1, 2],
                "stream_layer_indices": [1, 2],
            },
        ],
        capacity_banks=7,
        streaming_banks=2,
    )

    assert result["streamed_connection_indices"] == [1]
    assert result["choices"][0]["connection_name"] == "second_small"
    assert result["choices"][0]["solved_layer_count"] == 1


def test_greedy_streaming_prefers_first_connection_when_count_and_size_tie():
    result = simulate_cache._greedy_stream_connections(
        base_banks_by_layer=[2, 1, 2],
        connection_plans=[
            {
                "name": "first", "bank_count": 4,
                "resident_layer_indices": [0, 1],
                "stream_layer_indices": [0, 1],
            },
            {
                "name": "second", "bank_count": 4,
                "resident_layer_indices": [1, 2],
                "stream_layer_indices": [1, 2],
            },
        ],
        capacity_banks=7,
        streaming_banks=2,
    )

    assert result["choices"][0]["connection_name"] == "first"


def test_bank_optimized_cache_map_streams_selected_residual_at_endpoints():
    storage_a = ("cpu", None, 100)
    storage_b = ("cpu", None, 200)
    storage_c = ("cpu", None, 300)
    layers = [
        {
            "name": "producer", "type": "QuantLinear",
            "input_elems": 128, "output_elems": 400,
            "input_producer_layer_indices": [None],
            "output_tensor_id": 10, "output_tensor_storage_key": storage_a,
        },
        {
            "name": "near_consumer", "type": "QuantLinear",
            "input_elems": 400, "output_elems": 300,
            "input_producer_layer_indices": [0],
            "output_tensor_id": 20, "output_tensor_storage_key": storage_b,
        },
        {
            "name": "weighted_middle", "type": "QuantLinear",
            "input_elems": 300, "output_elems": 200, "weight_elems": 1,
            "input_producer_layer_indices": [1],
            "output_tensor_id": 30, "output_tensor_storage_key": storage_c,
        },
        {
            "name": "far_add", "type": "QuantAdd",
            "input_elems": 200, "output_elems": 200,
            "input_producer_layer_indices": [2, 0],
            "output_tensor_id": 40,
            "output_tensor_storage_key": ("cpu", None, 400),
            "residual_inputs": [{
                "elems": 400, "tensor_id": 10, "storage_key": storage_a,
                "producer_name": "producer",
            }],
        },
    ]

    cache_map = simulate_cache.build_cache_map(
        layers, cache_elements=1024, bank_size=128, metadata_bits=0,
    )

    residual = cache_map["residual_connections"][0]
    assert residual["placement"] == "streamed"
    assert residual["bank_count"] == 4
    assert cache_map["optimization"]["choices"][0]["connection_name"] == "residual_0"
    assert cache_map["optimization"]["red_layer_indices"] == []
    # Stream the long-lived residual copy at its producer and residual add.
    # The adjacent consumer still receives the producer's on-chip x_out/x_in
    # handoff, so it does not allocate a streaming buffer for the residual.
    assert [row["residual_0"] for row in cache_map["rows"]] == [
        0.256, 0, 0, 0.256
    ]
    assert cache_map["rows"][1]["x_in"] == 0.512
    assert cache_map["rows"][2]["weight_stream"] == 0.256
    assert cache_map["rows"][2]["total_cache_needed_kb"] == 0.896


def test_projected_skip_add_displays_main_branch_as_x_in():
    main_storage = ("cpu", None, 100)
    skip_storage = ("cpu", None, 200)
    cache_map = simulate_cache.build_cache_map([
        {
            "name": "conv3", "type": "QuantConv2d",
            "input_elems": 4, "output_elems": 8,
            "output_tensor_id": 10,
            "output_tensor_storage_key": main_storage,
        },
        {
            "name": "downsample", "type": "QuantConv2d",
            "input_elems": 4, "output_elems": 8,
            "output_tensor_id": 20,
            "output_tensor_storage_key": skip_storage,
        },
        {
            "name": "add", "type": "QuantAdd",
            "input_elems": 8, "output_elems": 8,
            "input_tensor_ids": [10, 20],
            "input_tensor_storage_keys": [main_storage, skip_storage],
            "output_tensor_id": 30,
            "output_tensor_storage_key": ("cpu", None, 300),
            "residual_inputs": [{
                "elems": 8,
                "tensor_id": 20,
                "storage_key": skip_storage,
                "producer_name": "downsample",
            }],
        },
    ])

    hold_name = cache_map["held_connections"][0]["name"]
    add_row = cache_map["rows"][2]
    assert cache_map["rows"][1][hold_name] == 0.008
    assert add_row["x_in"] == 0.008
    assert add_row["residual_0"] == 0.008
    assert add_row[hold_name] == 0
    assert add_row["total_cache_needed_kb"] == 0.024


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
