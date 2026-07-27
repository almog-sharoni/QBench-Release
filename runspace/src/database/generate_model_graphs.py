#!/usr/bin/env python3
"""
Generate and store quantization graphs for all models in the database.

This script:
1. Loads available models from YAML config
2. Traces the cache-map runtime producers and consumers
3. Stores compressed Cytoscape graph JSON in the database with metadata
4. Can be re-run to update graphs for specific models

Usage:
    python generate_model_graphs.py                    # Generate for all models
    python generate_model_graphs.py --models resnet18 vit_b_16  # Specific models
    python generate_model_graphs.py --skip-existing    # Skip models that already have graphs
"""

import os
import sys
import argparse
import json
import yaml
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.database.handler import RunDatabase
from runspace.src.adapters.generic_adapter import GenericAdapter

# Try to import torch and visualization tools, but make them optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  Warning: torch not installed. Graph generation may fail.")

try:
    from runspace.src.utils.architecture_viz import generate_hierarchical_json
    from runspace.src.utils.model_input_utils import resolve_model_input_size
    HIERARCHICAL_GRAPH_AVAILABLE = True
except Exception as e:
    HIERARCHICAL_GRAPH_AVAILABLE = False
    print(f"⚠️  Warning: architecture_viz not available: {e}")

try:
    from runspace.experiments.asic_cache_simulation.simulate_cache import (
        _producer_consumer_cache_plan,
        analyze_model,
        build_cache_map,
    )
    CACHE_GRAPH_AVAILABLE = True
except Exception as e:
    CACHE_GRAPH_AVAILABLE = False
    print(f"⚠️  Warning: cache-map tracing not available: {e}")


def _format_kb(size_kb):
    return f"{size_kb:.3f}".rstrip('0').rstrip('.') or '0'


GRAPH_SCHEMA_VERSION = 4


class CacheGraphValidationError(ValueError):
    """Raised when a runtime trace cannot produce a trustworthy cache graph."""


def _normalized_input_edges(layers, consumer_index):
    """Return explicit edges, with a compatibility form for synthetic traces."""
    layer = layers[consumer_index]
    input_edges = [dict(edge) for edge in layer.get('input_edges', [])]
    if input_edges:
        for edge_position, edge in enumerate(input_edges):
            edge.setdefault('input_index', edge_position)
            # Older hand-built test/compatibility traces did not tag their
            # first input. Runtime traces always set this field explicitly.
            if (
                edge.get('producer_layer_index') is None
                and not edge.get('is_model_state')
                and 'is_model_input' not in edge
            ):
                edge['is_model_input'] = consumer_index == 0
        return input_edges

    producer_indices = list(
        layer.get('input_producer_layer_indices', [])
    )
    if not producer_indices and consumer_index == 0 and layer.get('input_elems', 0):
        producer_indices = [None]
    return [
        {
            'input_index': input_index,
            'producer_layer_index': producer_index,
            'elements': (
                layers[producer_index].get('output_elems', 0)
                if isinstance(producer_index, int)
                and 0 <= producer_index < consumer_index
                else layer.get('input_elems', 0)
            ),
            'is_model_state': False,
            'is_model_input': (
                producer_index is None and consumer_index == 0
            ),
        }
        for input_index, producer_index in enumerate(producer_indices)
    ]


def validate_cache_trace(layers):
    """Fail when runtime tensor provenance is incomplete or contradictory."""
    errors = []
    model_input_count = 0
    model_state_count = 0

    for consumer_index, layer in enumerate(layers):
        layer_name = layer.get('name', f'layer_{consumer_index}')
        input_edges = _normalized_input_edges(layers, consumer_index)
        seen_input_indices = set()
        for edge_position, edge in enumerate(input_edges):
            input_index = edge.get('input_index', edge_position)
            if input_index in seen_input_indices:
                errors.append(
                    f"{layer_name}: duplicate input index {input_index}"
                )
            seen_input_indices.add(input_index)

            producer_index = edge.get('producer_layer_index')
            if producer_index is None:
                if edge.get('is_model_state'):
                    model_state_count += 1
                elif edge.get('is_model_input'):
                    model_input_count += 1
                else:
                    errors.append(
                        f"{layer_name}[input {input_index}]: unresolved internal "
                        "tensor (not a declared model input or model state)"
                    )
                continue
            if not isinstance(producer_index, int):
                errors.append(
                    f"{layer_name}[input {input_index}]: non-integer producer "
                    f"{producer_index!r}"
                )
            elif not 0 <= producer_index < consumer_index:
                errors.append(
                    f"{layer_name}[input {input_index}]: producer layer "
                    f"{producer_index} must be earlier than consumer layer "
                    f"{consumer_index}"
                )

        if layer.get('type') == 'QuantAdd' and len(input_edges) < 2:
            errors.append(
                f"{layer_name}: QuantAdd has {len(input_edges)} traced operand(s); "
                "expected at least 2"
            )

        for residual in layer.get('residual_inputs', []):
            producer_index = residual.get('producer_layer_index')
            if producer_index is not None and not (
                isinstance(producer_index, int)
                and 0 <= producer_index < consumer_index
            ):
                errors.append(
                    f"{layer_name}: residual producer {producer_index!r} must "
                    f"be earlier than layer {consumer_index}"
                )

    if errors:
        raise CacheGraphValidationError(
            "Invalid cache runtime trace:\n- " + "\n- ".join(errors)
        )
    return {
        'model_input_edges': model_input_count,
        'model_state_edges': model_state_count,
        'unresolved_internal_inputs': 0,
    }


def validate_cache_graph(elements, layers, cache_map):
    """Validate topology, sizes, and cache-lifetime/UI edge consistency."""
    errors = []
    node_by_id = {
        element.get('data', {}).get('id'): element.get('data', {})
        for element in elements
        if element.get('data', {}).get('id') is not None
        and 'source' not in element.get('data', {})
    }
    edges = [
        element.get('data', {}) for element in elements
        if 'source' in element.get('data', {})
    ]
    edge_ids = set()
    named_edges = {}
    incoming_operands = {index: [] for index in range(len(layers))}
    allocation_pair_names = {}

    for edge in edges:
        edge_id = edge.get('id')
        if edge_id in edge_ids:
            errors.append(f"duplicate graph edge id {edge_id!r}")
        edge_ids.add(edge_id)
        source = node_by_id.get(edge.get('source'))
        target = node_by_id.get(edge.get('target'))
        if source is None or target is None:
            errors.append(
                f"{edge_id}: missing source or target node"
            )
            continue
        source_index = source.get('execution_index')
        target_index = target.get('execution_index')
        if source_index is not None and target_index is not None:
            if source_index >= target_index:
                errors.append(
                    f"{edge_id}: backward/non-forward edge {source_index} -> "
                    f"{target_index}"
                )
        if target_index is not None and edge.get('connection_kind') != 'weight_stream':
            incoming_operands[target_index].append(edge)

        column = (
            edge.get('cache_lifetime_column')
            or edge.get('cache_map_column')
        )
        if column and (column.startswith('residual_') or column.startswith('hold_')):
            named_edges.setdefault(column, []).append(edge)
            allocation_key = (
                edge.get('cache_allocation_id'), edge.get('target')
            )
            allocation_pair_names.setdefault(allocation_key, set()).add(column)

    for allocation_key, names in allocation_pair_names.items():
        if len(names) > 1:
            errors.append(
                f"allocation {allocation_key[0]} -> {allocation_key[1]} is "
                f"represented by multiple cache connections: {sorted(names)}"
            )

    connection_by_name = {
        connection['name']: connection
        for connection in cache_map.get('connections', [])
    }
    lifetime_keys = {}
    for connection in cache_map.get('connections', []):
        name = connection['name']
        if not named_edges.get(name):
            errors.append(f"cache connection {name} has no graph arrow")
        expected_elements = int(connection.get('elements', 0))
        for edge in named_edges.get(name, []):
            if int(edge.get('tensor_elements', 0)) != expected_elements:
                errors.append(
                    f"{edge.get('id')}: {name} edge size "
                    f"{edge.get('tensor_elements')} != lifetime size "
                    f"{expected_elements}"
                )

        lifetime_key = (
            connection.get('producer_layer_index'),
            tuple(connection.get('consumer_layer_indices', [])),
            expected_elements,
        )
        lifetime_keys.setdefault(lifetime_key, set()).add(
            connection.get('kind')
        )

        if connection.get('placement') != 'streamed':
            resident_indices = connection.get('resident_layer_indices')
            if resident_indices:
                expected_indices = list(range(
                    resident_indices[0], resident_indices[-1] + 1
                ))
                if resident_indices != expected_indices:
                    errors.append(
                        f"{name}: resident lifetime is not contiguous: "
                        f"{resident_indices}"
                    )

    for lifetime_key, kinds in lifetime_keys.items():
        if 'residual' in kinds and 'hold' in kinds:
            errors.append(
                f"allocation lifetime {lifetime_key} is duplicated as both "
                "residual and hold"
            )

    rows = cache_map.get('rows', [])
    for layer_index, layer in enumerate(layers):
        layer_name = layer.get('name', f'layer_{layer_index}')
        expected_operands = len(_normalized_input_edges(layers, layer_index))
        actual_operands = len(incoming_operands[layer_index])
        if expected_operands != actual_operands:
            errors.append(
                f"{layer_name}: graph has {actual_operands} incoming operand "
                f"arrow(s), trace has {expected_operands}"
            )
        node = node_by_id.get(f'layer_{layer_index:04d}', {})
        active = node.get('active_cache_connections', {})
        expected_active = {
            name: rows[layer_index].get(name)
            for name in connection_by_name
            if rows[layer_index].get(name, 0)
        }
        if active != expected_active:
            errors.append(
                f"{layer_name}: node active cache metadata does not match "
                "the cache-map row"
            )
        for name in active:
            if not named_edges.get(name):
                errors.append(
                    f"{layer_name}: active connection {name} has no arrow to "
                    "highlight"
                )

    if errors:
        raise CacheGraphValidationError(
            "Invalid generated cache graph:\n- " + "\n- ".join(errors)
        )
    return {
        'validated_nodes': len(layers),
        'validated_edges': len(edges),
        'cache_connections': len(connection_by_name),
    }


def graph_json_has_runtime_hover_metadata(graph_json):
    """Whether a stored graph uses the cache-map node metadata schema."""
    try:
        elements = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
    except (TypeError, json.JSONDecodeError):
        return False
    return any(
        element.get('data', {}).get('type') == 'node'
        and element.get('data', {}).get('node_kind') != 'streamed_weight'
        and 'input_shapes' in element.get('data', {})
        and 'operation_metadata' in element.get('data', {})
        for element in (elements or [])
    )


def _graph_operation_metadata(layer):
    """Return compact, JSON-safe operation-specific hover metadata."""
    metadata = {}
    for key in (
        'kernel_size', 'stride', 'padding', 'dilation', 'padding_mode',
        'groups', 'in_channels', 'out_channels',
        'in_features', 'out_features',
        'normalized_shape', 'eps', 'num_features', 'num_groups',
        'num_channels', 'ceil_mode', 'output_size',
    ):
        value = layer.get(key)
        if value is not None:
            metadata[key] = list(value) if isinstance(value, tuple) else value

    input_shapes = layer.get('input_shapes') or []
    layer_type = str(layer.get('type', ''))
    if len(input_shapes) > 1:
        metadata['input_count'] = len(input_shapes)
    if layer_type in ('QuantMatMul', 'QuantBMM') and input_shapes:
        first_shape = input_shapes[0]
        if first_shape:
            metadata['reduction_dim'] = first_shape[-1]
    if layer_type == 'QuantAdd':
        metadata['residual_connections'] = len(layer.get('residual_inputs', []))
    fused_activations = [
        activation.get('type', 'unknown')
        for activation in layer.get('fused_activations', [])
    ]
    if fused_activations:
        metadata['fused_activations'] = fused_activations
    return metadata


def resolve_graph_cache_config(
    db, model_name, cache_size_m=None, num_banks=None, metadata_bits=None
):
    """Resolve graph cache settings from overrides, latest simulation, or defaults."""
    latest = db.get_latest_cache_simulation(model_name) or {}
    resolved_cache_size_m = float(
        cache_size_m
        if cache_size_m is not None
        else latest.get('cache_size_M') or 2.0
    )
    resolved_num_banks = int(
        num_banks
        if num_banks is not None
        else latest.get('num_banks') or 16
    )
    resolved_metadata_bits = int(
        metadata_bits
        if metadata_bits is not None
        else latest.get('metadata_bits') or 0
    )
    cache_elements = int(resolved_cache_size_m * 1_000_000)
    bank_size = cache_elements // resolved_num_banks
    return {
        'cache_size_m': resolved_cache_size_m,
        'cache_elements': cache_elements,
        'num_banks': resolved_num_banks,
        'bank_size': bank_size,
        'metadata_bits': resolved_metadata_bits,
    }


def generate_cache_map_graph_json(
    layers,
    cache_elements=None,
    bank_size=None,
    metadata_bits=0,
    streaming_banks=2,
    include_weight_streams=False,
):
    """Generate Cytoscape JSON from the exact cache-map runtime trace."""
    validate_cache_trace(layers)
    cache_map = build_cache_map(
        layers,
        cache_elements=cache_elements,
        bank_size=bank_size,
        metadata_bits=metadata_bits,
        streaming_banks=streaming_banks,
    )
    producer_consumer_plan = None
    if cache_elements is not None and bank_size and bank_size > 0:
        producer_consumer_plan = _producer_consumer_cache_plan(
            layers,
            cache_elements=cache_elements,
            bank_size=bank_size,
            metadata_bits=metadata_bits,
            streaming_banks=streaming_banks,
        )
    producer_transfer_pairs = set()
    producer_plan_steps = (
        producer_consumer_plan.get('steps', [])
        if producer_consumer_plan else []
    )
    for consumer_index, step in enumerate(producer_plan_steps):
        producer_transfer_pairs.update(
            (producer_index, consumer_index)
            for producer_index in step.get(
                'input_transfer_producer_indices', []
            )
        )
    rows = cache_map['rows']
    node_ids = [f'layer_{index:04d}' for index in range(len(layers))]
    elements = []

    for layer_index, (layer, row) in enumerate(zip(layers, rows)):
        layer_type = str(layer.get('type', 'unknown'))
        input_shapes = [list(shape) for shape in layer.get('input_shapes', [])]
        output_shape = layer.get('output_shape')
        if isinstance(output_shape, tuple):
            output_shape = list(output_shape)
        input_elements = int(layer.get('input_elems', 0))
        output_elements = int(layer.get('output_elems', 0))
        weight_elements = int(layer.get('weight_elems', 0) or 0)
        active_connections = {
            connection['name']: row[connection['name']]
            for connection in cache_map['connections']
            if row.get(connection['name'], 0)
        }
        elements.append({
            'data': {
                'id': node_ids[layer_index],
                'label': layer_type,
                'type': 'node',
                'color': '#a7f3d0' if 'quant' in layer_type.lower() else '#fde68a',
                'var_name': layer.get('name', 'unknown'),
                'execution_index': layer_index,
                'input_shapes': input_shapes,
                'input_shape': input_shapes[0] if len(input_shapes) == 1 else input_shapes,
                'output_shape': output_shape,
                'input_elements': input_elements,
                'output_elements': output_elements,
                'weight_elements': weight_elements,
                'input_size_kb': round(input_elements / 1_000.0, 3),
                'output_size_kb': round(output_elements / 1_000.0, 3),
                'weight_size_kb': round(weight_elements / 1_000.0, 3),
                'x_in_kb': row.get('x_in', 0),
                'x_out_kb': row.get('x_out', 0),
                'total_cache_needed_kb': row.get('total_cache_needed_kb', 0),
                'weight_stream_kb': row.get('weight_stream', 0),
                'pipeline_boundary_kb': row.get('pipeline_boundary', 0),
                'jumpback_kb': row.get('jumpback', 0),
                'input_output_overlap_kb': row.get(
                    'input_output_overlap_kb', 0
                ),
                'cache_rule': row.get('cache_rule'),
                'weight_streamed': weight_elements > 0,
                'active_cache_connections': active_connections,
                'operation_metadata': _graph_operation_metadata(layer),
                'cache_green': (
                    layer_index in cache_map['optimization']['green_layer_indices']
                    if cache_map.get('optimization') else None
                ),
                'output_streamed': (
                    bool(producer_plan_steps[layer_index].get('output_spilled'))
                    if layer_index < len(producer_plan_steps) else False
                ),
                'output_evicted_at': (
                    producer_plan_steps[layer_index].get('output_evicted_at')
                    if layer_index < len(producer_plan_steps) else None
                ),
                'input_streamed': (
                    bool(producer_plan_steps[layer_index].get(
                        'input_transfer_producer_indices', []
                    ))
                    if layer_index < len(producer_plan_steps) else False
                ),
            }
        })

    connections_by_pair = {}
    streamed_connections_by_pair = {}
    for connection in cache_map['connections']:
        producer_index = connection.get('producer_layer_index')
        edge_consumers = (
            connection.get('residual_consumer_layer_indices')
            or connection.get('consumer_layer_indices', [])
            if connection.get('kind') == 'residual'
            else connection.get('consumer_layer_indices', [])
        )
        if connection.get('placement') == 'streamed':
            for consumer_index in edge_consumers:
                streamed_connections_by_pair.setdefault(
                    (producer_index, consumer_index), connection
                )
        for consumer_index in edge_consumers:
            connections_by_pair.setdefault(
                (producer_index, consumer_index), []
            ).append(connection)

    producer_consumers = {}
    for consumer_index, layer in enumerate(layers):
        for producer_index in layer.get('input_producer_layer_indices', []):
            if producer_index is not None and producer_index < consumer_index:
                producer_consumers.setdefault(producer_index, set()).add(
                    consumer_index
                )
    for connection in cache_map['connections']:
        producer_index = connection.get('producer_layer_index')
        if producer_index is not None:
            producer_consumers.setdefault(producer_index, set()).update(
                connection.get('consumer_layer_indices', [])
            )

    edge_index = 0
    emitted_connections = set()
    incoming_targets = set()
    auxiliary_nodes = {}

    def emit_edge(
        producer_index, consumer_index, input_index, connection=None,
        input_count=None, cache_lifetime_connection=None,
    ):
        nonlocal edge_index
        if not (0 <= producer_index < len(layers)):
            return
        tensor_elements = int(
            connection.get('elements', 0)
            if connection is not None
            else layers[producer_index].get('output_elems', 0)
        )
        tensor_size_kb = round(tensor_elements / 1_000.0, 3)
        connection_name = connection.get('name') if connection is not None else None
        connection_kind = (
            connection.get('kind') if connection is not None else 'activation'
        )
        stream_connection = connection or streamed_connections_by_pair.get(
            (producer_index, consumer_index)
        )
        connection_streamed = bool(
            stream_connection is not None
            and stream_connection.get('placement') == 'streamed'
        )
        producer_transfer_streamed = (
            producer_index, consumer_index
        ) in producer_transfer_pairs
        streamed_out = connection_streamed or producer_transfer_streamed
        stream_reason = (
            'connection_placement'
            if connection_streamed
            else ('producer_consumer_transfer' if producer_transfer_streamed else None)
        )
        stream_buffer_kb = (
            round(streaming_banks * bank_size / 1_000.0, 3)
            if streamed_out and bank_size else 0
        )
        input_producers = [
            item for item in layers[consumer_index].get(
                'input_producer_layer_indices', []
            ) if item is not None
        ]
        edge_element = {
            'data': {
                'id': f'e{edge_index}',
                'source': node_ids[producer_index],
                'target': node_ids[consumer_index],
                'producer_node': layers[producer_index].get('name', 'unknown'),
                'cache_allocation_id': node_ids[producer_index],
                'cache_map_column': connection_name or 'x_in',
                'cache_lifetime_column': (
                    cache_lifetime_connection.get('name')
                    if cache_lifetime_connection is not None else None
                ),
                'connection_kind': connection_kind,
                'consumer_nodes': [layers[consumer_index].get('name', 'unknown')],
                'consumer_input_index': input_index,
                'is_fanout': len(producer_consumers.get(producer_index, ())) > 1,
                'is_multi_input': (
                    input_count > 1 if input_count is not None
                    else len(input_producers) > 1
                ),
                'tensor_elements': tensor_elements,
                'tensor_size_kb': tensor_size_kb,
                'streamed_out': streamed_out,
                'streamed_connection': (
                    stream_connection.get('name')
                    if connection_streamed else ('x_out' if streamed_out else None)
                ),
                'stream_reason': stream_reason,
                'stream_direction': 'out_in' if streamed_out else None,
                'stream_buffer_kb': stream_buffer_kb,
                'label': (
                    f"STREAM {_format_kb(tensor_size_kb)} KB"
                    if streamed_out else f"{_format_kb(tensor_size_kb)} KB"
                ),
            }
        }
        if streamed_out:
            edge_element['classes'] = 'streamed-out'
        elements.append(edge_element)
        incoming_targets.add(consumer_index)
        edge_index += 1
        if connection_name is not None:
            emitted_connections.add(
                (producer_index, consumer_index, connection_name)
            )

    def emit_unresolved_edge(consumer_index, input_edge, input_count):
        nonlocal edge_index
        is_model_state = bool(input_edge.get('is_model_state'))
        if is_model_state:
            identity = repr(
                input_edge.get('storage_key') or input_edge.get('tensor_id')
            )
            auxiliary_key = ('model_state', identity)
            label = 'Model state'
            connection_kind = 'model_state'
        else:
            if not input_edge.get('is_model_input'):
                raise CacheGraphValidationError(
                    f"{layers[consumer_index].get('name', consumer_index)}: "
                    "refusing to render an unresolved internal tensor as Input"
                )
            identity = repr(
                input_edge.get('storage_key')
                or input_edge.get('tensor_id')
                or input_edge.get('input_index')
            )
            auxiliary_key = ('model_input', identity)
            label = 'Input'
            connection_kind = 'input'
        source_id = auxiliary_nodes.get(auxiliary_key)
        tensor_elements = int(input_edge.get('elements', 0))
        tensor_size_kb = round(tensor_elements / 1_000.0, 3)
        if source_id is None:
            source_id = f'aux_{len(auxiliary_nodes):04d}'
            auxiliary_nodes[auxiliary_key] = source_id
            elements.append({
                'data': {
                    'id': source_id,
                    'label': label,
                    'type': 'node',
                    'color': '#e2e8f0',
                    'var_name': label.lower().replace(' ', '_'),
                    'output_elements': tensor_elements,
                    'output_size_kb': tensor_size_kb,
                }
            })
        elements.append({
            'data': {
                'id': f'e{edge_index}',
                'source': source_id,
                'target': node_ids[consumer_index],
                'producer_node': label,
                'cache_allocation_id': source_id,
                'cache_map_column': connection_kind,
                'connection_kind': connection_kind,
                'consumer_nodes': [layers[consumer_index].get('name', 'unknown')],
                'consumer_input_index': input_edge.get('input_index'),
                'is_fanout': False,
                'is_multi_input': input_count > 1,
                'tensor_elements': tensor_elements,
                'tensor_size_kb': tensor_size_kb,
                'label': f"{_format_kb(tensor_size_kb)} KB",
            }
        })
        incoming_targets.add(consumer_index)
        edge_index += 1

    for consumer_index, layer in enumerate(layers):
        pair_offsets = {}
        input_edges = _normalized_input_edges(layers, consumer_index)
        for input_edge in input_edges:
            input_index = input_edge.get('input_index')
            producer_index = input_edge.get('producer_layer_index')
            if producer_index is None:
                emit_unresolved_edge(consumer_index, input_edge, len(input_edges))
                continue
            if producer_index >= consumer_index:
                continue
            pair = (producer_index, consumer_index)
            candidates = connections_by_pair.get(pair, [])
            offset = pair_offsets.get(pair, 0)
            connection = candidates[offset] if offset < len(candidates) else None
            pair_offsets[pair] = offset + 1
            cache_lifetime_connection = None
            if (
                connection is not None
                and connection.get('kind') == 'hold'
                and layer.get('type') == 'QuantAdd'
                and input_index == 0
                and connection.get('placement') != 'streamed'
            ):
                cache_lifetime_connection = connection
                emitted_connections.add(
                    (producer_index, consumer_index, connection['name'])
                )
                connection = None
            emit_edge(
                producer_index, consumer_index, input_index, connection,
                input_count=len(input_edges),
                cache_lifetime_connection=cache_lifetime_connection,
            )

    # Preserve compatibility connections whose runtime input record could not
    # be resolved by tensor identity.
    for connection in cache_map['connections']:
        producer_index = connection.get('producer_layer_index')
        edge_consumers = (
            connection.get('residual_consumer_layer_indices')
            or connection.get('consumer_layer_indices', [])
            if connection.get('kind') == 'residual'
            else connection.get('consumer_layer_indices', [])
        )
        for consumer_index in edge_consumers:
            key = (producer_index, consumer_index, connection['name'])
            if key in emitted_connections or producer_index is None:
                continue
            emit_edge(producer_index, consumer_index, None, connection)

    if (
        layers
        and _normalized_input_edges(layers, 0)
        and 0 not in incoming_targets
    ):
        input_elements = int(layers[0].get('input_elems', 0))
        input_size_kb = round(input_elements / 1_000.0, 3)
        elements.append({
            'data': {
                'id': 'aux_fallback_input',
                'label': 'Input',
                'type': 'node',
                'color': '#e2e8f0',
                'var_name': 'model_input',
                'output_elements': input_elements,
                'output_size_kb': input_size_kb,
            }
        })
        elements.append({
            'data': {
                'id': f'e{edge_index}',
                'source': 'aux_fallback_input',
                'target': node_ids[0],
                'producer_node': 'model_input',
                'cache_allocation_id': 'aux_fallback_input',
                'cache_map_column': 'x_in',
                'connection_kind': 'input',
                'consumer_nodes': [layers[0].get('name', 'unknown')],
                'consumer_input_index': 0,
                'is_fanout': False,
                'is_multi_input': False,
                'tensor_elements': input_elements,
                'tensor_size_kb': input_size_kb,
                'label': f"{_format_kb(input_size_kb)} KB",
            }
        })
        edge_index += 1

    if include_weight_streams:
        for layer_index, layer in enumerate(layers):
            weight_elements = int(layer.get('weight_elems', 0) or 0)
            if weight_elements <= 0:
                continue
            weight_size_kb = round(weight_elements / 1_000.0, 3)
            weight_node_id = f'weight_{layer_index:04d}'
            elements.append({
                'classes': 'streamed-weight',
                'data': {
                    'id': weight_node_id,
                    'label': 'W',
                    'type': 'node',
                    'node_kind': 'streamed_weight',
                    'color': '#dbeafe',
                    'var_name': f"{layer.get('name', 'unknown')} weights",
                    'weight_elements': weight_elements,
                    'weight_size_kb': weight_size_kb,
                }
            })
            elements.append({
                'classes': 'weight-stream',
                'data': {
                    'id': f'e{edge_index}',
                    'source': weight_node_id,
                    'target': node_ids[layer_index],
                    'producer_node': 'external_weights',
                    'cache_allocation_id': weight_node_id,
                    'cache_map_column': 'weight_stream',
                    'connection_kind': 'weight_stream',
                    'consumer_nodes': [layer.get('name', 'unknown')],
                    'is_fanout': False,
                    'is_multi_input': False,
                    'tensor_elements': weight_elements,
                    'tensor_size_kb': weight_size_kb,
                    'streamed_out': False,
                    'stream_direction': 'in',
                    'stream_buffer_kb': (
                        round(streaming_banks * bank_size / 1_000.0, 3)
                        if bank_size else 0
                    ),
                    'label': 'W stream',
                }
            })
            edge_index += 1

    validate_cache_graph(elements, layers, cache_map)
    return json.dumps(elements)


def load_model_names(config_file=None):
    """Load model names from config file or use defaults."""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            if isinstance(config, list):
                return [m.get('name') for m in config if 'name' in m]
    
    # Fallback: Common model names
    default_models = [
        # 'resnet18', 'resnet50', 
        # 'resnet152',
        # 'vit_b_16', 
        # 'efficientnet_b0', 'efficientnet_v2_l',
        # 'mobilenet_v3_large',
        # # 'densenet121', 'densenet161',
        # 'inception_v3',
        # 'alexnet', 'vgg19_bn',
        # 'googlenet',
        'mobilevit_s',
    ]
    return default_models

def get_model_from_name(model_name, quantized=True):
    """
    Load a model by name, optionally quantized.
    
    Args:
        model_name (str): Name of the model (e.g., 'resnet50')
        quantized (bool): Whether to load the quantized version.
        
    Returns:
        torch.nn.Module: Loaded model
    """
    if not TORCH_AVAILABLE:
        return None
    
    try:
        if quantized:
            print(f"  Loading quantized {model_name} via GenericAdapter...")
            # Replace supported layers with Quant* modules so they render as
            # quantized, but avoid tracing activation quantization kernels.
            adapter = GenericAdapter(
                model_name=model_name,
                quantized_ops=["all"],
                input_quantization=False,
                enable_fx_quantization=False,  # keep module hierarchy intact for torchview
                skip_calibration=True,
            )
            return adapter.model
        else:
            # Try torchvision models
            from torchvision import models
            
            model_fn = getattr(models, model_name, None)
            if model_fn:
                print(f"  Loading vanilla {model_name} from torchvision...")
                model = model_fn(weights=None)
                model.eval()
                return model
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
    
    return None

def generate_graph_for_model(
    model_name,
    db,
    force=False,
    quantized=True,
    graph_depth=12,
    legacy_hierarchical=False,
    cache_size_m=None,
    num_banks=None,
    metadata_bits=None,
):
    """
    Generate quantization graph for a single model.
    
    Args:
        model_name (str): Name of the model
        db (RunDatabase): Database instance
        force (bool): Force regeneration even if exists
        quantized (bool): Whether to generate graph for quantized model
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check dependencies
    if not TORCH_AVAILABLE:
        print(f"✗ Skipping {model_name}: torch not available")
        return False
    
    graph_backend_available = (
        HIERARCHICAL_GRAPH_AVAILABLE
        if legacy_hierarchical
        else CACHE_GRAPH_AVAILABLE
    )
    if not graph_backend_available:
        backend = 'architecture_viz' if legacy_hierarchical else 'cache-map tracer'
        print(f"✗ Skipping {model_name}: {backend} not available")
        return False
    
    # Check if already exists
    if not force and db.has_model_graph(model_name):
        existing_metadata = db.get_model_graph_metadata(model_name) or {}
        existing_schema = int(
            existing_metadata.get('graph_schema_version') or 0
        )
        if existing_schema >= GRAPH_SCHEMA_VERSION:
            print(f"✓ Graph already exists for {model_name}, skipping...")
            return True
        print(
            f"↻ Regenerating {model_name}: graph schema {existing_schema} "
            f"is older than {GRAPH_SCHEMA_VERSION}"
        )
    
    print(f"\n📊 Generating {'quantized' if quantized else 'vanilla'} graph for {model_name}...", end=" ", flush=True)
    
    try:
        # Generate JSON graph representation. The default graph uses the same
        # runtime trace as cache_map_<model>.csv. The former TorchView hierarchy
        # remains available for compatibility with existing stored graphs.
        trace_validation = None
        if legacy_hierarchical:
            model = get_model_from_name(model_name, quantized=quantized)
            if model is None:
                print(f"✗ Could not load {model_name}")
                return False
            model.eval()
            print("  Tracing and generating hierarchical JSON...")
            input_size = resolve_model_input_size(model)
            graph_json = generate_hierarchical_json(
                model,
                input_size=input_size,
                model_name=model_name,
                depth=graph_depth,
            )
            graph_kind = 'hierarchical'
        else:
            print("  Tracing cache-map producers and consumers...")
            cache_config = resolve_graph_cache_config(
                db,
                model_name,
                cache_size_m=cache_size_m,
                num_banks=num_banks,
                metadata_bits=metadata_bits,
            )
            _, cache_map_layers = analyze_model(
                {'name': model_name, 'weights': None},
                batch_size=1,
                device='cpu',
                adapter_cfg={
                    'type': 'generic',
                    'build_quantized': quantized,
                },
                return_cache_map_layers=True,
            )
            trace_validation = validate_cache_trace(cache_map_layers)
            graph_json = generate_cache_map_graph_json(
                cache_map_layers,
                cache_elements=cache_config['cache_elements'],
                bank_size=cache_config['bank_size'],
                metadata_bits=cache_config['metadata_bits'],
                include_weight_streams=True,
            )
            graph_kind = 'cache_map_runtime'
        
        if not graph_json:
            print(f"✗ Graph generation produced empty JSON for {model_name}")
            return False

        # Extract metadata from the JSON (count nodes, etc.)
        parsed_json = json.loads(graph_json)
        num_nodes = sum(1 for e in parsed_json if e.get('data', {}).get('type') in ('node', 'compound'))
        num_quantized = sum(1 for e in parsed_json if '#a7f3d0' in str(e.get('data', {}).get('color', '')))
        num_streamed_edges = sum(
            1 for element in parsed_json
            if element.get('data', {}).get('streamed_out') is True
        )
        num_weight_streams = sum(
            1 for element in parsed_json
            if element.get('data', {}).get('connection_kind') == 'weight_stream'
        )

        metadata = {
            'num_nodes': num_nodes,
            'num_quantized_layers': num_quantized,
            'num_streamed_edges': num_streamed_edges,
            'num_weight_streams': num_weight_streams,
            'graph_kind': graph_kind,
            'graph_schema_version': GRAPH_SCHEMA_VERSION,
            'generated_at': datetime.now().isoformat()
        }
        if not legacy_hierarchical:
            metadata.update(cache_config)
            metadata.update({
                'graph_validation': 'passed',
                **(trace_validation or {}),
            })

        # Store in database
        db.store_model_graph(model_name, graph_json, metadata)
        print(f"✓ Success")
        return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--models', 
        nargs='+',
        help='Specific models to generate graphs for (default: all available)'
    )
    parser.add_argument(
        '--config',
        help='Path to model config YAML file'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip models that already have graphs in database'
    )
    parser.add_argument(
        '--db-path',
        help='Path to database file (default: runspace/database/runs.db)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate graphs even if they exist'
    )
    parser.add_argument(
        '--vanilla',
        action='store_true',
        help='Generate graph for unquantized (vanilla) model instead'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=12,
        help='Tracing depth for hierarchical graph generation (higher exposes more internals)'
    )
    parser.add_argument(
        '--legacy-hierarchical',
        action='store_true',
        help='Use the former TorchView hierarchical graph instead of producer/consumer stages',
    )
    parser.add_argument(
        '--cache-size',
        type=float,
        default=None,
        help='Cache size in millions of elements (default: latest simulation, then 2.0)',
    )
    parser.add_argument(
        '--num-banks',
        type=int,
        default=None,
        help='Cache bank count (default: latest simulation, then 16)',
    )
    parser.add_argument(
        '--metadata-bits',
        type=int,
        default=None,
        help='Metadata bits per 128-element chunk (default: latest simulation, then 0)',
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TORCH_AVAILABLE:
        print("❌ Error: torch is required for graph generation.")
        print("\nTo install torch in your Apptainer environment:")
        print("  1. Run: ./run_apptainer.sh pip install torch torchvision")
        print("  2. Or install locally: pip install torch torchvision")
        print("\nWithin Apptainer, torch should already be available.")
        return
    
    required_backend_available = (
        HIERARCHICAL_GRAPH_AVAILABLE
        if args.legacy_hierarchical
        else CACHE_GRAPH_AVAILABLE
    )
    if not required_backend_available:
        backend = 'architecture_viz' if args.legacy_hierarchical else 'cache-map tracer'
        print(f"❌ Error: {backend} is not available.")
        return
    
    # Initialize database
    db = RunDatabase(db_path=args.db_path) if args.db_path else RunDatabase()
    
    # Get list of models to process
    if args.models:
        models_to_process = args.models
    else:
        models_to_process = load_model_names(args.config)
    
    print(f"🚀 Generating quantization graphs for {len(models_to_process)} models...")
    print(f"Database: {db.db_path}")
    print("-" * 60)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for model_name in models_to_process:
        if args.skip_existing and db.has_model_graph(model_name):
            print(f"⊘ Skipping {model_name} (already exists)")
            skip_count += 1
            continue
        
        if generate_graph_for_model(
            model_name,
            db,
            force=args.force,
            quantized=not args.vanilla,
            graph_depth=args.depth,
            legacy_hierarchical=args.legacy_hierarchical,
            cache_size_m=args.cache_size,
            num_banks=args.num_banks,
            metadata_bits=args.metadata_bits,
        ):
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    print("\n" + "-" * 60)
    print(f"✓ Completed: {success_count} graphs generated")
    if skip_count > 0:
        print(f"⊘ Skipped: {skip_count} graphs (already exist)")
    if fail_count > 0:
        print(f"✗ Failed: {fail_count} models")
    
    # Show storage info
    print("\n📦 Storage Summary:")
    graphs_df = db.get_all_model_graphs()
    if not graphs_df.empty:
        total_compressed = graphs_df['graph_size_compressed'].sum()
        total_original = graphs_df['graph_size_original'].sum()
        avg_reduction = 100 * (1 - total_compressed / total_original) if total_original > 0 else 0
        
        print(f"  Total graphs: {len(graphs_df)}")
        print(f"  Original size: {total_original / (1024*1024):.2f} MB")
        print(f"  Compressed size: {total_compressed / (1024*1024):.2f} MB")
        print(f"  Compression ratio: {avg_reduction:.1f}%")
        
        # Show per-model sizes
        print("\n  Per-model breakdown:")
        for _, row in graphs_df.iterrows():
            reduction = 100 * (1 - row['graph_size_compressed'] / row['graph_size_original']) if row['graph_size_original'] > 0 else 0
            print(f"    {row['model_name']:20s}: {row['graph_size_original']/1024:7.1f}KB → {row['graph_size_compressed']/1024:7.1f}KB ({reduction:5.1f}%)")

if __name__ == "__main__":
    main()
