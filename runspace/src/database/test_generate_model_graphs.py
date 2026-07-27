import json

import pytest

from runspace.experiments.asic_cache_simulation.simulate_cache import build_cache_map
from runspace.src.database.generate_model_graphs import (
    CacheGraphValidationError,
    _graph_operation_metadata,
    generate_cache_map_graph_json,
    graph_json_has_runtime_hover_metadata,
    validate_cache_graph,
    validate_cache_trace,
)


def test_cache_map_graph_preserves_runtime_fanout_and_residual_edges():
    stem_storage = ('cpu', None, 100)
    main_storage = ('cpu', None, 200)
    skip_storage = ('cpu', None, 300)
    layers = [
        {
            'name': 'stem', 'type': 'QuantConv2d',
            'input_elems': 4, 'output_elems': 4,
            'output_tensor_id': 10,
            'output_tensor_storage_key': stem_storage,
        },
        {
            'name': 'conv3', 'type': 'QuantConv2d',
            'input_elems': 4, 'output_elems': 8,
            'input_producer_layer_indices': [0],
            'output_tensor_id': 20,
            'output_tensor_storage_key': main_storage,
        },
        {
            'name': 'downsample', 'type': 'QuantConv2d',
            'input_elems': 4, 'output_elems': 8,
            'input_producer_layer_indices': [0],
            'output_tensor_id': 30,
            'output_tensor_storage_key': skip_storage,
        },
        {
            'name': 'add', 'type': 'QuantAdd',
            'input_elems': 8, 'output_elems': 8,
            'input_producer_layer_indices': [1, 2],
            'input_tensor_ids': [20, 30],
            'input_tensor_storage_keys': [main_storage, skip_storage],
            'output_tensor_id': 40,
            'output_tensor_storage_key': ('cpu', None, 400),
            'residual_inputs': [{
                'elems': 8,
                'tensor_id': 30,
                'storage_key': skip_storage,
                'producer_name': 'downsample',
            }],
        },
    ]

    elements = json.loads(generate_cache_map_graph_json(layers))
    nodes = {
        element['data']['var_name']: element['data']
        for element in elements
        if element['data'].get('type') == 'node'
    }
    edges = [
        element['data']
        for element in elements
        if 'source' in element['data']
    ]
    add_incoming = [
        edge for edge in edges
        if edge['target'] == nodes['add']['id']
    ]

    assert len(add_incoming) == 2
    assert {edge['connection_kind'] for edge in add_incoming} == {
        'activation', 'residual'
    }
    assert {edge['cache_map_column'] for edge in add_incoming} == {
        'x_in', 'residual_0'
    }
    assert all(edge['tensor_elements'] == 8 for edge in add_incoming)
    assert all(edge['tensor_size_kb'] == 0.008 for edge in add_incoming)
    assert all(edge['label'] == '0.008 KB' for edge in add_incoming)
    assert nodes['add']['total_cache_needed_kb'] == 0.024
    assert nodes['add']['active_cache_connections'] == {
        'residual_0': 0.008,
    }
    assert nodes['add']['color'] == '#a7f3d0'
    conv3_to_add = next(
        edge for edge in add_incoming
        if edge['producer_node'] == 'conv3'
    )
    assert conv3_to_add['cache_map_column'] == 'x_in'
    assert conv3_to_add['cache_lifetime_column'] == 'hold_1_conv3_to_add'


def test_trace_validation_rejects_an_unresolved_internal_tensor():
    layers = [
        {
            'name': 'stem', 'type': 'QuantConv2d',
            'input_elems': 4, 'output_elems': 4,
            'input_edges': [{
                'input_index': 0, 'elements': 4,
                'producer_layer_index': None,
                'is_model_input': True,
                'is_model_state': False,
            }],
        },
        {
            'name': 'later_add', 'type': 'QuantAdd',
            'input_elems': 4, 'output_elems': 4,
            'input_edges': [
                {
                    'input_index': 0, 'elements': 4,
                    'producer_layer_index': 0,
                    'is_model_input': False,
                    'is_model_state': False,
                },
                {
                    'input_index': 1, 'elements': 4,
                    'producer_layer_index': None,
                    'is_model_input': False,
                    'is_model_state': False,
                },
            ],
        },
    ]

    with pytest.raises(
        CacheGraphValidationError, match='unresolved internal tensor'
    ):
        generate_cache_map_graph_json(layers)


def test_trace_validation_accepts_a_declared_model_input_at_a_later_add():
    layers = [
        {
            'name': 'branch', 'type': 'QuantLinear',
            'input_elems': 4, 'output_elems': 4,
            'input_edges': [{
                'input_index': 0, 'elements': 4, 'tensor_id': 10,
                'producer_layer_index': None,
                'is_model_input': True,
                'is_model_state': False,
            }],
        },
        {
            'name': 'add', 'type': 'QuantAdd',
            'input_elems': 4, 'output_elems': 4,
            'input_edges': [
                {
                    'input_index': 0, 'elements': 4,
                    'producer_layer_index': 0,
                    'is_model_input': False,
                    'is_model_state': False,
                },
                {
                    'input_index': 1, 'elements': 4, 'tensor_id': 20,
                    'producer_layer_index': None,
                    'is_model_input': True,
                    'is_model_state': False,
                },
            ],
        },
    ]

    summary = validate_cache_trace(layers)
    elements = json.loads(generate_cache_map_graph_json(layers))
    input_nodes = [
        element['data'] for element in elements
        if element['data'].get('var_name') == 'input'
    ]

    assert summary['unresolved_internal_inputs'] == 0
    assert summary['model_input_edges'] == 2
    assert len(input_nodes) == 2


def test_trace_validation_rejects_backward_producer_indices():
    layers = [
        {'name': 'first', 'type': 'QuantLinear', 'input_elems': 4, 'output_elems': 4},
        {
            'name': 'second', 'type': 'QuantLinear',
            'input_elems': 4, 'output_elems': 4,
            'input_edges': [{
                'input_index': 0, 'elements': 4,
                'producer_layer_index': 1,
                'is_model_input': False,
                'is_model_state': False,
            }],
        },
    ]

    with pytest.raises(
        CacheGraphValidationError, match='must be earlier'
    ):
        validate_cache_trace(layers)


def test_graph_validation_rejects_cache_edge_size_drift():
    layers = [
        {
            'name': 'producer', 'type': 'QuantLinear',
            'input_elems': 4, 'output_elems': 8,
            'input_producer_layer_indices': [None],
            'output_tensor_id': 10,
            'output_tensor_storage_key': ('cpu', None, 100),
        },
        {
            'name': 'middle', 'type': 'QuantLinear',
            'input_elems': 8, 'output_elems': 8,
            'input_producer_layer_indices': [0],
            'output_tensor_id': 20,
            'output_tensor_storage_key': ('cpu', None, 200),
        },
        {
            'name': 'add', 'type': 'QuantAdd',
            'input_elems': 8, 'output_elems': 8,
            'input_producer_layer_indices': [1, 0],
            'residual_inputs': [{
                'elems': 8, 'tensor_id': 10,
                'storage_key': ('cpu', None, 100),
                'producer_layer_index': 0,
                'producer_name': 'producer',
            }],
        },
    ]
    cache_map = build_cache_map(layers)
    elements = json.loads(generate_cache_map_graph_json(layers))
    residual_edge = next(
        element['data'] for element in elements
        if element['data'].get('cache_map_column') == 'residual_0'
    )
    residual_edge['tensor_elements'] = 7

    with pytest.raises(
        CacheGraphValidationError, match='edge size 7 != lifetime size 8'
    ):
        validate_cache_graph(elements, layers, cache_map)


def test_cache_map_graph_keeps_static_quantadd_operand_as_an_edge():
    layers = [
        {
            'name': 'tokens', 'type': 'QuantCat',
            'input_elems': 8, 'output_elems': 8,
            'output_tensor_id': 10,
            'output_tensor_storage_key': ('cpu', None, 100),
            'input_edges': [{
                'input_index': 0, 'elements': 8,
                'producer_layer_index': None, 'is_model_state': False,
            }],
        },
        {
            'name': 'add', 'type': 'QuantAdd',
            'input_elems': 8, 'output_elems': 8,
            'input_producer_layer_indices': [0, None],
            'input_edges': [
                {
                    'input_index': 0, 'elements': 8,
                    'producer_layer_index': 0, 'is_model_state': False,
                },
                {
                    'input_index': 1, 'elements': 8,
                    'producer_layer_index': None, 'is_model_state': True,
                    'tensor_id': 99,
                },
            ],
        },
    ]

    elements = json.loads(generate_cache_map_graph_json(layers))
    add_node = next(
        element['data'] for element in elements
        if element['data'].get('var_name') == 'add'
    )
    incoming = [
        element['data'] for element in elements
        if element['data'].get('target') == add_node['id']
    ]

    assert len(incoming) == 2
    assert {edge['connection_kind'] for edge in incoming} == {
        'activation', 'model_state'
    }
    assert all(edge['is_multi_input'] for edge in incoming)


def test_cache_optimized_graph_marks_only_residual_stream_endpoints_and_adds_weight_arrow():
    storage_a = ('cpu', None, 100)
    storage_b = ('cpu', None, 200)
    storage_c = ('cpu', None, 300)
    layers = [
        {
            'name': 'producer', 'type': 'QuantLinear',
            'input_elems': 128, 'output_elems': 400,
            'input_producer_layer_indices': [None],
            'output_tensor_id': 10, 'output_tensor_storage_key': storage_a,
        },
        {
            'name': 'near_consumer', 'type': 'QuantLinear',
            'input_elems': 400, 'output_elems': 300,
            'input_producer_layer_indices': [0],
            'output_tensor_id': 20, 'output_tensor_storage_key': storage_b,
        },
        {
            'name': 'weighted_middle', 'type': 'QuantLinear',
            'input_elems': 300, 'output_elems': 200, 'weight_elems': 1,
            'input_producer_layer_indices': [1],
            'output_tensor_id': 30, 'output_tensor_storage_key': storage_c,
        },
        {
            'name': 'far_add', 'type': 'QuantAdd',
            'input_elems': 200, 'output_elems': 200,
            'input_producer_layer_indices': [2, 0],
            'residual_inputs': [{
                'elems': 400, 'tensor_id': 10, 'storage_key': storage_a,
                'producer_name': 'producer',
            }],
        },
    ]

    elements = json.loads(generate_cache_map_graph_json(
        layers,
        cache_elements=1024,
        bank_size=128,
        include_weight_streams=True,
    ))
    nodes = {
        element['data']['var_name']: element['data']
        for element in elements
        if element['data'].get('type') == 'node'
    }
    edges = [element['data'] for element in elements if 'source' in element['data']]

    streamed_elements = [
        element for element in elements
        if element.get('data', {}).get('streamed_out')
    ]
    assert all(element.get('classes') == 'streamed-out' for element in streamed_elements)
    streamed_edges = [element['data'] for element in streamed_elements]
    assert {
        (edge['source'], edge['target']) for edge in streamed_edges
    } == {
        (nodes['producer']['id'], nodes['far_add']['id']),
    }
    assert all(edge['streamed_connection'] == 'residual_0' for edge in streamed_edges)
    assert all(edge['label'].startswith('STREAM ') for edge in streamed_edges)
    assert all(edge['stream_buffer_kb'] == 0.256 for edge in streamed_edges)
    near_edge = next(
        edge for edge in edges
        if edge['source'] == nodes['producer']['id']
        and edge['target'] == nodes['near_consumer']['id']
    )
    assert near_edge['streamed_out'] is False

    weight_nodes = [
        element['data'] for element in elements
        if element['data'].get('node_kind') == 'streamed_weight'
    ]
    weight_edges = [
        edge for edge in edges if edge.get('connection_kind') == 'weight_stream'
    ]
    assert len(weight_nodes) == 1
    assert len(weight_edges) == 1
    weight_node_element = next(
        element for element in elements
        if element['data'].get('node_kind') == 'streamed_weight'
    )
    weight_edge_element = next(
        element for element in elements
        if element['data'].get('connection_kind') == 'weight_stream'
    )
    assert weight_node_element.get('classes') == 'streamed-weight'
    assert weight_edge_element.get('classes') == 'weight-stream'
    assert weight_edges[0]['target'] == nodes['weighted_middle']['id']
    assert weight_edges[0]['stream_direction'] == 'in'
    assert weight_edges[0]['label'] == 'W stream'


def test_graph_marks_an_oversized_direct_output_as_a_producer_consumer_stream():
    layers = [
        {
            'name': 'oversized', 'type': 'QuantLinear',
            'input_elems': 128, 'output_elems': 400,
            'input_producer_layer_indices': [None],
            'output_tensor_id': 10,
            'output_tensor_storage_key': ('cpu', None, 100),
        },
        {
            'name': 'consumer', 'type': 'QuantLinear',
            'input_elems': 400, 'output_elems': 128,
            'input_producer_layer_indices': [0],
            'output_tensor_id': 20,
            'output_tensor_storage_key': ('cpu', None, 200),
        },
    ]

    elements = json.loads(generate_cache_map_graph_json(
        layers,
        cache_elements=256,
        bank_size=128,
    ))
    nodes = {
        element['data']['var_name']: element['data']
        for element in elements
        if element['data'].get('type') == 'node'
    }
    edge = next(
        element['data'] for element in elements
        if element['data'].get('source') == nodes['oversized']['id']
        and element['data'].get('target') == nodes['consumer']['id']
    )

    assert nodes['oversized']['output_streamed'] is True
    assert nodes['consumer']['input_streamed'] is True
    assert edge['streamed_out'] is True
    assert edge['streamed_connection'] == 'x_out'
    assert edge['stream_reason'] == 'producer_consumer_transfer'
    assert edge['label'] == 'STREAM 0.4 KB'


def test_cache_map_graph_exposes_layer_hover_metadata():
    layers = [{
        'name': 'features.0',
        'type': 'QuantConv2d',
        'input_elems': 3 * 32 * 32,
        'output_elems': 16 * 30 * 30,
        'weight_elems': 16 * 3 * 3 * 3,
        'input_shapes': [(1, 3, 32, 32)],
        'output_shape': (1, 16, 30, 30),
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'padding': (0, 0),
        'dilation': (1, 1),
        'padding_mode': 'zeros',
        'groups': 1,
        'in_channels': 3,
        'out_channels': 16,
    }]

    elements = json.loads(generate_cache_map_graph_json(
        layers,
        cache_elements=64_000,
        bank_size=4_000,
        include_weight_streams=True,
    ))
    node = next(
        element['data'] for element in elements
        if element['data'].get('var_name') == 'features.0'
    )

    assert node['input_shapes'] == [[1, 3, 32, 32]]
    assert node['input_shape'] == [1, 3, 32, 32]
    assert node['output_shape'] == [1, 16, 30, 30]
    assert node['input_size_kb'] == 3.072
    assert node['output_size_kb'] == 14.4
    assert node['weight_size_kb'] == 0.432
    assert node['weight_streamed'] is True
    assert node['weight_stream_kb'] == 8.0
    assert node['operation_metadata'] == {
        'kernel_size': [3, 3],
        'stride': [1, 1],
        'padding': [0, 0],
        'dilation': [1, 1],
        'padding_mode': 'zeros',
        'groups': 1,
        'in_channels': 3,
        'out_channels': 16,
    }


def test_operation_hover_metadata_is_specific_to_layer_type():
    assert _graph_operation_metadata({
        'type': 'QuantLinear',
        'in_features': 768,
        'out_features': 3072,
    }) == {
        'in_features': 768,
        'out_features': 3072,
    }
    assert _graph_operation_metadata({
        'type': 'QuantMatMul',
        'input_shapes': [(1, 12, 197, 64), (1, 12, 64, 197)],
    }) == {
        'input_count': 2,
        'reduction_dim': 64,
    }
    assert _graph_operation_metadata({
        'type': 'QuantAdd',
        'input_shapes': [(1, 197, 768), (1, 197, 768)],
        'residual_inputs': [{'producer_name': 'skip'}],
    }) == {
        'input_count': 2,
        'residual_connections': 1,
    }
    assert _graph_operation_metadata({
        'type': 'LayerNorm',
        'normalized_shape': (768,),
        'eps': 1e-6,
    }) == {
        'normalized_shape': [768],
        'eps': 1e-6,
    }
    assert _graph_operation_metadata({
        'type': 'QuantMul',
        'input_shapes': [(4, 4, 256, 36)],
        'operand_count': 2,
        'constant_operands': [{
            'input_index': 1,
            'value': 1 / 6,
            'type': 'float',
        }],
    }) == {
        'operand_count': 2,
        'constant_operands': [
            'input 2 = 0.16666666666666666 (float)'
        ],
        'operation': 'tensor × 0.16666666666666666',
    }
    assert _graph_operation_metadata({
        'type': 'QuantConv2d',
        'fused_activations': [
            {'name': 'layer1.1.relu', 'type': 'QuantReLU'},
        ],
    }) == {
        'fused_activations': ['QuantReLU'],
    }


def test_scalar_quantmul_shows_its_constant_as_an_auxiliary_operand():
    layers = [{
        'name': 'scale',
        'type': 'QuantMul',
        'input_elems': 16,
        'output_elems': 16,
        'input_shapes': [(1, 16)],
        'output_shape': (1, 16),
        'operand_count': 2,
        'constant_operands': [{
            'input_index': 1,
            'value': 0.125,
            'type': 'float',
        }],
        'input_edges': [{
            'input_index': 0,
            'elements': 16,
            'producer_layer_index': None,
            'is_model_input': True,
            'is_model_state': False,
        }],
    }]

    elements = json.loads(generate_cache_map_graph_json(layers))
    node = next(
        element['data'] for element in elements
        if element['data'].get('execution_index') == 0
    )
    incoming = [
        element['data'] for element in elements
        if element['data'].get('target') == node['id']
    ]

    constant_node = next(
        element for element in elements
        if element['data'].get('node_kind') == 'constant_operand'
    )
    constant_edge = next(
        edge for edge in incoming
        if edge.get('connection_kind') == 'constant'
    )

    assert node['label'] == 'QuantMul'
    assert node['operation_type'] == 'QuantMul'
    assert node['constant_operand_count'] == 1
    assert node['operation_metadata']['operation'] == 'tensor × 0.125'
    assert node['operation_metadata']['constant_operands'] == [
        'input 2 = 0.125 (float)'
    ]
    assert len(incoming) == 2
    assert {edge['connection_kind'] for edge in incoming} == {
        'input', 'constant'
    }
    assert constant_node.get('classes') == 'constant-operand'
    assert constant_node['data']['label'] == 'C'
    assert constant_node['data']['constant_value'] == 0.125
    assert constant_edge['source'] == constant_node['data']['id']
    assert constant_edge['consumer_input_index'] == 1
    assert constant_edge['constant_value'] == 0.125
    assert constant_edge['tensor_elements'] == 0
    assert constant_edge['tensor_size_kb'] == 0
    assert constant_edge['stream_buffer_kb'] == 0
    assert constant_edge['streamed_out'] is False
    assert constant_edge['label'] == 'C = 0.125'


def test_legacy_cached_graph_is_detected_as_missing_runtime_hover_metadata():
    legacy_graph = json.dumps([{
        'data': {
            'id': '0',
            'type': 'node',
            'var_name': 'conv1',
            'input_shape': [1, 3, 224, 224],
            'output_shape': [1, 64, 112, 112],
            'module_args': 'kernel_size=(7, 7)',
        },
    }])
    runtime_graph = generate_cache_map_graph_json([{
        'name': 'conv1',
        'type': 'QuantConv2d',
        'input_elems': 150528,
        'output_elems': 802816,
        'input_shapes': [(1, 3, 224, 224)],
        'output_shape': (1, 64, 112, 112),
        'kernel_size': (7, 7),
    }])

    assert graph_json_has_runtime_hover_metadata(legacy_graph) is False
    assert graph_json_has_runtime_hover_metadata(runtime_graph) is True
