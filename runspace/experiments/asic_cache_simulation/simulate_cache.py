import os
import sys
import csv
import copy
import json
import weakref
import torch
import torch.nn as nn
import argparse
import math
from datetime import datetime
from runspace.src.registry.op_registry import OpRegistry

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def get_footprint_elements(num_elements: int, metadata_bits: int) -> int:
    """
    Calculate total element-equivalent footprint including metadata overhead.

    Tensors are allocated in 128-element chunks. If a tensor uses any part
    of a chunk, it occupies the full chunk.

    Metadata bytes are counted as element-equivalents (1 byte = 1 FP8 element)
    and are also computed per chunk.
    """
    if num_elements <= 0:
        return 0
    chunk_size = 128
    num_chunks = math.ceil(num_elements / chunk_size)
    chunk_elems = num_chunks * chunk_size
    metadata_elems = math.ceil(num_chunks * metadata_bits / 8)
    return chunk_elems + metadata_elems


def round_to_banks(size_elems: int, bank_size: int) -> int:
    """Round up to the nearest bank boundary (in elements)."""
    if size_elems <= 0:
        return 0
    if bank_size <= 0:
        return size_elems
    return math.ceil(size_elems / bank_size) * bank_size


def fmt_elems(n: int) -> str:
    """Human-readable element count."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Rule system (imported from rules.py)
# ---------------------------------------------------------------------------
try:
    from runspace.experiments.asic_cache_simulation.rules import RULES, LAYER_RULES
except ImportError:
    from rules import RULES, LAYER_RULES


def _next_layer_viable(next_layer: dict, metadata_bits: int, bank_size: int,
                       cache_elements: int, xin_in_cache: bool) -> bool:
    """
    Check whether the next layer has at least one valid rule (1-level lookahead).

    xin_in_cache: whether the next layer's xin arrives from cache (True) or
                  external memory (False). Only rules whose xin_from_cache matches
                  are considered.
    """
    if next_layer is None:
        return True
    ni = round_to_banks(get_footprint_elements(next_layer['input_elems'],  metadata_bits), bank_size)
    no = round_to_banks(get_footprint_elements(next_layer['output_elems'], metadata_bits), bank_size)
    nw = round_to_banks(get_footprint_elements(next_layer['weight_elems'], metadata_bits), bank_size)
    ctx = {
        'input_banked':   ni, 'output_banked': no, 'weight_banked': nw,
        'cache_elements': cache_elements, 'bank_size': bank_size,
        'jump_back_size_in_banks': next_layer.get('jump_back_size_in_banks', 0),
    }
    layer_type = next_layer.get('type', '__default__')
    rule_keys  = LAYER_RULES.get(layer_type, LAYER_RULES['__default__'])
    for key in rule_keys:
        rule = RULES[key]
        if rule['xin_from_cache'] != xin_in_cache:
            continue
        guard = rule.get('ctx_guard')
        if guard is not None and not guard(ctx):
            continue
        if rule['stay'](ctx):
            return True
    return False


def evaluate_stay(layer: dict, ctx: dict,
                  next_layer, metadata_bits: int, bank_size: int, cache_elements: int) -> tuple:
    """
    Returns (stay_on_chip, perm_elems, possible, rule_name).

    Looks up the layer type in LAYER_RULES and tries each rule key in order.
    A rule is confirmed if ctx_guard passes (if present), stay() passes, and
    the next layer has a compatible rule given the xin source implied by on_chip.
    If no rule confirms → stay_on_chip=False, rule_name='FLAGGED'.
    """
    layer_type = layer.get('type', '__default__')
    rule_keys  = LAYER_RULES.get(layer_type, LAYER_RULES['__default__'])
    for key in rule_keys:
        rule  = RULES[key]
        guard = rule.get('ctx_guard')
        if guard is not None and not guard(ctx):
            continue
        if not rule['stay'](ctx):
            continue
        on_chip = rule['on_chip']
        if not _next_layer_viable(next_layer, metadata_bits, bank_size, cache_elements,
                                  xin_in_cache=on_chip):
            continue
        return on_chip, rule['perm'](ctx), True, key
    return False, 0, False, 'FLAGGED'


def _rule_aware_workspace(
    layer: dict,
    input_banks: int,
    output_banks: int,
    bank_size: int,
    *,
    input_is_reusable: bool,
    input_output_already_shared: bool = False,
) -> dict:
    """Select the first compatible hardware rule and its cache adjustment.

    Producer-consumer lifetimes decide which tensors must remain live. The
    selected rule only describes the current operator's workspace: whether a
    final-use input can be overwritten by xout, plus pipeline/jumpback banks.
    Capacity is evaluated later with all concurrent lifetimes included.
    """
    layer_type = layer.get('type', '__default__')
    rule_keys = LAYER_RULES.get(layer_type, LAYER_RULES['__default__'])
    ctx = {
        'input_banked': int(input_banks) * int(bank_size),
        'output_banked': int(output_banks) * int(bank_size),
        'weight_banked': 0,
        'cache_elements': 0,
        'bank_size': int(bank_size),
        'jump_back_size_in_banks': int(
            layer.get('jump_back_size_in_banks', 0) or 0
        ),
    }

    for rule_name in rule_keys:
        rule = RULES[rule_name]
        guard = rule.get('ctx_guard')
        if guard is not None and not guard(ctx):
            continue
        reuses_input = bool(rule.get('reuse_input_for_output'))
        if reuses_input and not input_is_reusable:
            continue

        overlap_banks = 0
        shared_banks = 0
        pipeline_banks = 0
        jumpback_banks = 0
        if reuses_input:
            shared_banks = min(int(input_banks), int(output_banks))
            if not input_output_already_shared:
                overlap_banks = shared_banks
            pipeline_banks = int(rule.get('pipeline_banks', 0) or 0)
            raw_jumpback_banks = math.ceil(
                int(layer.get('jump_back_size_in_banks', 0) or 0)
                / int(bank_size)
            ) if bank_size else 0
            if (
                rule.get('jumpback_mode') == 'input_output_ratio'
                and output_banks > 0
            ):
                raw_jumpback_banks = math.ceil(
                    raw_jumpback_banks * input_banks / output_banks
                )
            jumpback_banks = raw_jumpback_banks

        return {
            'rule': rule_name,
            'reuses_input': reuses_input,
            'overlap_banks': overlap_banks,
            'shared_banks': shared_banks,
            'pipeline_boundary_banks': pipeline_banks,
            'jumpback_banks': jumpback_banks,
            'overhead_banks': pipeline_banks + jumpback_banks,
        }

    return {
        'rule': 'producer_consumer_default',
        'reuses_input': False,
        'overlap_banks': 0,
        'shared_banks': 0,
        'pipeline_boundary_banks': 0,
        'jumpback_banks': 0,
        'overhead_banks': 0,
    }


def is_pipeline_fusable_activation(layer_type: str) -> bool:
    """Whether a unary pointwise activation can share its producer pipeline."""
    cleaned_type = str(layer_type).lower()
    if cleaned_type.startswith('quant'):
        cleaned_type = cleaned_type[5:]

    # Reductions and normalization stages need their own scheduled/cache row.
    if 'softmax' in cleaned_type or 'norm' in cleaned_type:
        return False

    try:
        if OpRegistry.is_activation(layer_type):
            return True
    except Exception:
        pass

    return cleaned_type in {
        'relu', 'relu6', 'gelu', 'silu', 'swish', 'mish',
        'hardswish', 'hardsigmoid', 'leakyrelu', 'prelu', 'elu',
        'selu', 'celu', 'tanh', 'sigmoid', 'softplus', 'softsign',
    }


def is_collapsible(layer_type: str) -> bool:
    """Compatibility alias for hardware-pipeline activation fusion."""
    return is_pipeline_fusable_activation(layer_type)


def is_registry_activation(layer_type: str) -> bool:
    """Check whether a layer type is marked as activation in OpRegistry."""
    try:
        if OpRegistry.is_activation(layer_type):
            return True
        for original_cls, quantized_cls in OpRegistry.get_supported_ops().items():
            if (
                quantized_cls.__name__ == layer_type
                and OpRegistry.is_activation(quantized_cls.__name__)
            ):
                return True
            if (
                original_cls.__name__ == layer_type
                and OpRegistry.is_activation(quantized_cls.__name__)
            ):
                return True
    except Exception:
        pass
    return False


def fuse_pipeline_activations(layers: list[dict]) -> list[dict]:
    """Fold safe unary pointwise activations into their direct producer.

    Fusion is deliberately conservative: the activation must be shape
    preserving, immediately follow one captured producer, and be that
    producer's only consumer. This prevents changing semantics when another
    branch still needs the pre-activation value. Producer indices and tensor
    identity are rewritten so the returned list remains a valid runtime
    producer-consumer trace for both the cache map and dashboard graph.
    """
    source_layers = copy.deepcopy(layers)
    consumers_by_producer = {index: set() for index in range(len(source_layers))}

    def _producer_indices(layer):
        indices = list(layer.get('input_producer_layer_indices', []))
        if not indices:
            indices = [
                edge.get('producer_layer_index')
                for edge in layer.get('input_edges', [])
            ]
        return [index for index in indices if index is not None]

    for consumer_index, layer in enumerate(source_layers):
        for producer_index in set(_producer_indices(layer)):
            if 0 <= producer_index < consumer_index:
                consumers_by_producer[producer_index].add(consumer_index)

    fused_layers = []
    old_to_new = {}
    producer_name_aliases = {}

    def _remap_producer_index(producer_index):
        if producer_index is None:
            return None
        return old_to_new.get(producer_index, producer_index)

    for old_index, layer in enumerate(source_layers):
        original_producers = _producer_indices(layer)
        if 'input_producer_layer_indices' in layer:
            layer['input_producer_layer_indices'] = [
                _remap_producer_index(index)
                for index in layer.get('input_producer_layer_indices', [])
            ]
        for edge in layer.get('input_edges', []):
            edge['producer_layer_index'] = _remap_producer_index(
                edge.get('producer_layer_index')
            )
            producer_index = edge.get('producer_layer_index')
            if producer_index is not None and 0 <= producer_index < len(fused_layers):
                edge['producer_name'] = fused_layers[producer_index].get('name')

        input_shapes = layer.get('input_shapes', [])
        shape_preserving = (
            int(layer.get('input_elems', 0)) == int(layer.get('output_elems', 0))
            and (
                not input_shapes
                or not layer.get('output_shape')
                or tuple(input_shapes[0]) == tuple(layer.get('output_shape'))
            )
        )
        original_producer = original_producers[0] if len(original_producers) == 1 else None
        remapped_producer = (
            old_to_new.get(original_producer)
            if original_producer is not None else None
        )
        safe_to_fuse = (
            is_pipeline_fusable_activation(layer.get('type', ''))
            and int(layer.get('weight_elems', 0) or 0) == 0
            and len(input_shapes) <= 1
            and shape_preserving
            and original_producer is not None
            and consumers_by_producer.get(original_producer) == {old_index}
            and remapped_producer == len(fused_layers) - 1
        )

        if safe_to_fuse:
            producer = fused_layers[remapped_producer]
            producer.setdefault('fused_activations', []).append({
                'name': layer.get('name', 'unknown'),
                'type': layer.get('type', 'unknown'),
            })
            for key in (
                'output_elems', 'output_shape', 'output_tensor_id',
                'output_tensor_storage_key', '_output_tensor',
            ):
                if key in layer:
                    producer[key] = layer[key]
            if layer.get('residual_output_elems', 0):
                producer['residual_output_elems'] = max(
                    producer.get('residual_output_elems', 0),
                    layer['residual_output_elems'],
                )
                producer.setdefault('residual_output_consumers', []).extend(
                    layer.get('residual_output_consumers', [])
                )
            producer_name_aliases[layer.get('name')] = producer.get('name')
            old_to_new[old_index] = remapped_producer
            continue

        old_to_new[old_index] = len(fused_layers)
        fused_layers.append(layer)

    for new_index, layer in enumerate(fused_layers):
        layer['_execution_index'] = new_index
        for residual in layer.get('residual_inputs', []):
            residual['producer_layer_index'] = _remap_producer_index(
                residual.get('producer_layer_index')
            )
            producer_name = residual.get('producer_name')
            if producer_name in producer_name_aliases:
                residual['producer_name'] = producer_name_aliases[producer_name]

    return fused_layers


# ---------------------------------------------------------------------------

COMPUTE_PU_WIDTH = 128


def _cycles_for_ops(num_ops: float) -> int:
    if num_ops <= 0:
        return 0
    return math.ceil(num_ops / COMPUTE_PU_WIDTH)


def _cycles_for_reduction_outputs(num_outputs: int, reduction_dim: float) -> int:
    if num_outputs <= 0 or reduction_dim <= 0:
        return 0
    chunks_per_output = math.ceil(reduction_dim / COMPUTE_PU_WIDTH)
    return int(num_outputs) * chunks_per_output + 1


def _numel_from_shape(shape) -> int:
    if not shape:
        return 0
    return math.prod(shape)


def _collect_tensor_shapes(value) -> list[tuple]:
    shapes = []
    if isinstance(value, torch.Tensor):
        shapes.append(tuple(value.shape))
    elif isinstance(value, (tuple, list)):
        for item in value:
            shapes.extend(_collect_tensor_shapes(item))
    return shapes


def _collect_tensor_ids(value) -> list[int]:
    ids = []
    if isinstance(value, torch.Tensor):
        ids.append(id(value))
    elif isinstance(value, (tuple, list)):
        for item in value:
            ids.extend(_collect_tensor_ids(item))
    return ids


def _tensor_storage_key(tensor: torch.Tensor):
    """Return a key shared by views but not by recycled allocations."""
    try:
        storage = tensor.untyped_storage()
        return (
            tensor.device.type,
            tensor.device.index,
            storage._cdata,
        )
    except (AttributeError, RuntimeError):
        return None


def _collect_tensor_storage_keys(value) -> list:
    keys = []
    if isinstance(value, torch.Tensor):
        keys.append(_tensor_storage_key(value))
    elif isinstance(value, (tuple, list)):
        for item in value:
            keys.extend(_collect_tensor_storage_keys(item))
    return keys


def _collect_tensors(value) -> list[torch.Tensor]:
    tensors = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(_collect_tensors(item))
    return tensors


def _residual_input_indices(input_tensors: list[torch.Tensor],
                            tensor_producers: dict,
                            is_static_tensor) -> list[int]:
    """Identify bypass operands of an activation add from tensor provenance.

    Model parameters/buffers (for example a ViT positional embedding) are not
    residuals.  Of the remaining activation operands, the tensor with the
    deepest producer path is the computed branch and every shallower operand
    is a bypass that must remain live until the add.  Execution order breaks
    depth ties.  This handles projected ResNet skips, which execute after the
    main branch despite being shallower, and does not depend on whether a
    model writes ``branch + skip`` or ``skip + branch``.
    """
    activation_operands = []
    for input_index, tensor in enumerate(input_tensors):
        if is_static_tensor(tensor):
            continue
        producer = tensor_producers.get(id(tensor))
        producer_index = (
            producer.get('_execution_index') if producer is not None else None
        )
        producer_depth = (
            producer.get('_graph_depth', 0) if producer is not None else 0
        )
        activation_operands.append((input_index, producer_depth, producer_index))

    if len(activation_operands) < 2:
        return []

    produced_operands = [
        operand for operand in activation_operands if operand[2] is not None
    ]
    if not produced_operands:
        return []

    branch_input_index, _, _ = max(
        produced_operands,
        key=lambda operand: (operand[1], operand[2]),
    )
    return [
        input_index for input_index, _, _ in activation_operands
        if input_index != branch_input_index
    ]


def _repair_unrecorded_layout_lineage(layers: list[dict]) -> int:
    """Bridge a producer across an unrecorded shape/layout-only tensor chain.

    A copying ``contiguous()`` breaks both Python identity and storage identity.
    Repair only the unambiguous schedule case: every recorded input producer is
    unresolved, the immediately preceding output has the same element count,
    and that output otherwise has no recorded consumer.
    """
    consumed_producers = {
        producer_index
        for layer in layers
        for producer_index in layer.get('input_producer_layer_indices', [])
        if producer_index is not None
    }
    repaired = 0
    for layer_index in range(1, len(layers)):
        layer = layers[layer_index]
        producers = list(layer.get('input_producer_layer_indices', []))
        if not producers or any(index is not None for index in producers):
            continue
        producer_index = layer_index - 1
        producer = layers[producer_index]
        producer_elements = int(producer.get('output_elems', 0))
        if (
            producer_index in consumed_producers
            or producer_elements <= 0
            or producer_elements != int(layer.get('input_elems', 0))
        ):
            continue
        producers[0] = producer_index
        layer['input_producer_layer_indices'] = producers
        layer['input_lineage_repaired_from'] = producer_index
        input_edges = layer.get('input_edges', [])
        if input_edges:
            input_edges[0]['producer_layer_index'] = producer_index
            input_edges[0]['producer_name'] = producer.get('name')
            input_edges[0]['lineage_repaired'] = True
        consumed_producers.add(producer_index)
        repaired += 1
    return repaired


def _repair_lineage_from_prior_inputs(layers: list[dict]) -> int:
    """Reuse producer knowledge learned at an earlier tensor consumer.

    An unrecorded layout chain may copy a producer output, so the first
    recorded consumer resolves through ``_base`` while a later residual use
    no longer does.  Both consumers still receive the same copied tensor.
    Remember the resolved producer by that input's runtime identity (or its
    live storage plus shape) and apply it to later unresolved edges.

    This is what connects MobileViT's unfolded transformer input to both its
    first LayerNorm and the later attention residual add.
    """
    known_by_tensor = {}
    known_by_storage = {}
    repaired = 0

    def _shape_key(shape):
        return tuple(shape) if shape is not None else None

    for consumer_index, layer in enumerate(layers):
        input_edges = layer.get('input_edges', [])
        producer_indices = list(
            layer.get('input_producer_layer_indices', [])
        )
        for edge_position, edge in enumerate(input_edges):
            if edge.get('is_model_state'):
                continue
            tensor_id = edge.get('tensor_id')
            storage_key = edge.get('storage_key')
            elements = int(edge.get('elements', 0) or 0)
            shape = _shape_key(edge.get('shape'))
            storage_identity = (
                (tuple(storage_key), elements, shape)
                if storage_key is not None else None
            )
            producer_index = edge.get('producer_layer_index')

            if producer_index is None:
                known = (
                    known_by_tensor.get(tensor_id)
                    if tensor_id is not None else None
                )
                if known is None and storage_identity is not None:
                    known = known_by_storage.get(storage_identity)
                if known is not None and known[0] < consumer_index:
                    producer_index, producer_name = known
                    edge['producer_layer_index'] = producer_index
                    edge['producer_name'] = producer_name
                    edge['lineage_repaired_from_prior_input'] = True
                    input_index = int(edge.get('input_index', edge_position))
                    while len(producer_indices) <= input_index:
                        producer_indices.append(None)
                    producer_indices[input_index] = producer_index
                    repaired += 1

            if (
                isinstance(producer_index, int)
                and 0 <= producer_index < consumer_index
            ):
                known = (
                    producer_index,
                    layers[producer_index].get('name'),
                )
                if tensor_id is not None:
                    known_by_tensor[tensor_id] = known
                if storage_identity is not None:
                    known_by_storage[storage_identity] = known

        if input_edges:
            layer['input_producer_layer_indices'] = producer_indices
    return repaired


def _rebuild_residual_metadata_from_edges(layers: list[dict]) -> None:
    """Classify residual operands after the complete runtime graph is known."""
    residual_layer_keys = (
        'residual_inputs', 'residual_input_elems',
        'residual_input_stream_elems', 'residual_input_shape',
        'residual_input_tensor_id', 'residual_producer_name',
    )
    for layer in layers:
        for key in residual_layer_keys:
            layer.pop(key, None)
        layer.pop('residual_output_elems', None)
        layer.pop('residual_output_consumers', None)

    for layer_index, layer in enumerate(layers):
        parent_depths = [
            layers[producer_index].get('_graph_depth', 0)
            for producer_index in layer.get('input_producer_layer_indices', [])
            if producer_index is not None and producer_index < layer_index
        ]
        layer['_graph_depth'] = 1 + max(parent_depths, default=0)

    for consumer_index, layer in enumerate(layers):
        if layer.get('type') != 'QuantAdd':
            continue
        activation_edges = [
            edge for edge in layer.get('input_edges', [])
            if not edge.get('is_model_state')
            and edge.get('producer_layer_index') is not None
        ]
        if len(activation_edges) < 2:
            continue
        main_edge = max(
            activation_edges,
            key=lambda edge: (
                layers[edge['producer_layer_index']].get('_graph_depth', 0),
                edge['producer_layer_index'],
            ),
        )
        residual_edges = [edge for edge in activation_edges if edge is not main_edge]
        residual_inputs = []
        for edge in residual_edges:
            producer_index = edge['producer_layer_index']
            producer = layers[producer_index]
            residual = {
                'elems': int(edge.get('elements', 0)),
                'shape': edge.get('shape'),
                'tensor_id': edge.get('tensor_id'),
                'storage_key': edge.get('storage_key'),
                'producer_layer_index': producer_index,
                'producer_name': producer.get('name'),
            }
            residual_inputs.append(residual)
            producer['residual_output_elems'] = max(
                producer.get('residual_output_elems', 0), residual['elems']
            )
            producer.setdefault('residual_output_consumers', []).append(
                layer.get('name', 'unknown')
            )
        layer['residual_inputs'] = residual_inputs
        layer['residual_input_elems'] = sum(
            residual['elems'] for residual in residual_inputs
        )
        layer['residual_input_stream_elems'] = layer['residual_input_elems']
        layer['residual_producer_name'] = residual_inputs[0]['producer_name']
        if len(residual_inputs) == 1:
            layer['residual_input_shape'] = residual_inputs[0]['shape']
            layer['residual_input_tensor_id'] = residual_inputs[0]['tensor_id']


def _producer_consumer_cache_plan(
    layers: list[dict], cache_elements: int, bank_size: int, metadata_bits: int,
    streaming_banks: int = 2,
) -> dict:
    """Simulate activation-cache residency from explicit tensor lifetimes.

    Every captured layer output is one allocation.  It becomes live when its
    producer executes and remains live through its final consumer.  A current
    ``x_in`` that refers to a resident producer is therefore never allocated a
    second time.  When all live buffers do not fit, a deterministic Belady-like
    policy evicts the buffer whose next use is farthest in the future.

    Model parameters/buffers are reported as transfers but are intentionally
    excluded from activation-cache lifetimes, matching ``build_cache_map``.
    """
    layer_count = len(layers)
    cache_elements = max(0, int(cache_elements))

    def _banked(elements: int) -> int:
        return round_to_banks(
            get_footprint_elements(int(elements or 0), metadata_bits),
            bank_size,
        )

    normalized_edges = []
    consumers_by_producer = {index: [] for index in range(layer_count)}
    for consumer_index, layer in enumerate(layers):
        edges = list(layer.get('input_edges', []))
        if not edges:
            edges = [
                {
                    'input_index': input_index,
                    'elements': layer.get('input_elems', 0),
                    'producer_layer_index': producer_index,
                    'producer_name': (
                        layers[producer_index].get('name')
                        if isinstance(producer_index, int)
                        and 0 <= producer_index < consumer_index
                        else None
                    ),
                    'is_model_state': False,
                }
                for input_index, producer_index in enumerate(
                    layer.get('input_producer_layer_indices', [])
                )
            ]
        clean_edges = []
        for edge in edges:
            edge = dict(edge)
            producer_index = edge.get('producer_layer_index')
            if not (
                isinstance(producer_index, int)
                and 0 <= producer_index < consumer_index
            ):
                producer_index = None
            edge['producer_layer_index'] = producer_index
            clean_edges.append(edge)
            if producer_index is not None:
                consumers_by_producer[producer_index].append(consumer_index)
        normalized_edges.append(clean_edges)

    for producer_index in consumers_by_producer:
        consumers_by_producer[producer_index] = sorted(set(
            consumers_by_producer[producer_index]
        ))

    output_banked = [
        _banked(layer.get('output_elems', 0)) for layer in layers
    ]
    last_consumer = [
        max(consumers_by_producer[index], default=index)
        for index in range(layer_count)
    ]

    def _next_use(producer_index: int, after_index: int) -> float:
        for consumer_index in consumers_by_producer[producer_index]:
            if consumer_index > after_index:
                return consumer_index
        return math.inf

    def _edge_identity(edge: dict, fallback_index: int):
        storage_key = edge.get('storage_key')
        if storage_key is not None:
            return ('storage', tuple(storage_key))
        tensor_id = edge.get('tensor_id')
        if tensor_id is not None:
            return ('tensor', tensor_id)
        return ('input', fallback_index)

    def _is_residual_edge(layer: dict, edge: dict) -> bool:
        for residual in layer.get('residual_inputs', []):
            if (
                residual.get('tensor_id') is not None
                and residual.get('tensor_id') == edge.get('tensor_id')
            ):
                return True
            if (
                residual.get('storage_key') is not None
                and residual.get('storage_key') == edge.get('storage_key')
            ):
                return True
            if (
                residual.get('producer_name') is not None
                and residual.get('producer_name') == edge.get('producer_name')
            ):
                return True
        return False

    resident = set()
    external_backed = set()
    output_spilled = [False] * layer_count
    output_evicted_at = [None] * layer_count
    steps = []

    for layer_index, layer in enumerate(layers):
        resident = {
            producer_index for producer_index in resident
            if last_consumer[producer_index] >= layer_index
        }
        resident_before = set(resident)
        resident_before_execution_eviction = set(resident_before)

        missing_producer_sizes = {}
        missing_residual_producer_sizes = {}
        external_input_sizes = {}
        external_residual_sizes = {}
        model_state_sizes = {}
        for edge_index, edge in enumerate(normalized_edges[layer_index]):
            elements = int(edge.get('elements', 0) or 0)
            producer_index = edge.get('producer_layer_index')
            if producer_index is not None:
                if producer_index not in resident_before:
                    target = (
                        missing_residual_producer_sizes
                        if _is_residual_edge(layer, edge)
                        else missing_producer_sizes
                    )
                    target[producer_index] = max(
                        target.get(producer_index, 0), elements
                    )
                    external_backed.add(producer_index)
                continue

            identity = _edge_identity(edge, edge_index)
            if edge.get('is_model_state'):
                model_state_sizes[identity] = max(
                    model_state_sizes.get(identity, 0), elements
                )
            else:
                target = (
                    external_residual_sizes
                    if _is_residual_edge(layer, edge)
                    else external_input_sizes
                )
                target[identity] = max(target.get(identity, 0), elements)

        logical_live_producers = [
            producer_index for producer_index in range(layer_index + 1)
            if output_banked[producer_index] > 0
            and last_consumer[producer_index] >= layer_index
        ]
        external_input_banked = sum(
            _banked(elements)
            for elements in (
                list(external_input_sizes.values())
                + list(external_residual_sizes.values())
            )
        )
        logical_required = (
            sum(output_banked[index] for index in logical_live_producers)
            + external_input_banked
        )
        reusable_input_banks = []
        candidate_edges = (
            normalized_edges[layer_index]
            if layer.get('type') == 'QuantAdd'
            else normalized_edges[layer_index][:1]
        )
        for edge in candidate_edges:
            producer_index = edge.get('producer_layer_index')
            if (
                producer_index is not None
                and last_consumer[producer_index] == layer_index
            ):
                reusable_input_banks.append(output_banked[producer_index] // bank_size)
            elif (
                producer_index is None
                and not edge.get('is_model_state')
                and edge is normalized_edges[layer_index][0]
            ):
                reusable_input_banks.append(
                    _banked(edge.get('elements', layer.get('input_elems', 0)))
                    // bank_size
                )
        workspace = _rule_aware_workspace(
            layer,
            max(reusable_input_banks or [
                _banked(layer.get('input_elems', 0)) // bank_size
            ]),
            output_banked[layer_index] // bank_size,
            bank_size,
            input_is_reusable=bool(reusable_input_banks),
        )
        if workspace.get('reuses_input') and reusable_input_banks:
            logical_required -= min(
                max(reusable_input_banks),
                output_banked[layer_index] // bank_size,
            ) * bank_size
            logical_required += workspace['overhead_banks'] * bank_size
        has_streamed_model_data = bool(layer.get('weight_elems', 0)) or bool(
            model_state_sizes
        )
        if has_streamed_model_data:
            logical_required += int(streaming_banks) * int(bank_size)

        # Spill background lifetimes before execution when the current
        # rule-aware workspace would otherwise overflow. Current input
        # producers are protected because this operator still consumes them.
        current_input_producers = {
            edge.get('producer_layer_index')
            for edge in normalized_edges[layer_index]
            if edge.get('producer_layer_index') is not None
        }
        execution_required = logical_required
        execution_evicted = []
        while execution_required > cache_elements:
            victims = [
                producer_index for producer_index in resident_before
                if producer_index not in current_input_producers
                and producer_index != layer_index
            ]
            if not victims:
                break
            victim = max(
                victims,
                key=lambda producer_index: (
                    _next_use(producer_index, layer_index - 1),
                    output_banked[producer_index],
                    producer_index,
                ),
            )
            resident_before.remove(victim)
            resident.discard(victim)
            execution_required -= output_banked[victim]
            execution_evicted.append(victim)
            if victim not in external_backed:
                output_spilled[victim] = True
                output_evicted_at[victim] = layer_index
                external_backed.add(victim)

        # After this layer has consumed its inputs, retain only tensors with a
        # later use.  Reloaded fan-out inputs may be cached again for that use.
        candidates = {
            producer_index for producer_index in resident_before
            if last_consumer[producer_index] > layer_index
        }
        candidates.update(
            producer_index
            for producer_index in (
                list(missing_producer_sizes)
                + list(missing_residual_producer_sizes)
            )
            if last_consumer[producer_index] > layer_index
        )
        if consumers_by_producer[layer_index]:
            candidates.add(layer_index)

        evicted = list(execution_evicted)
        candidate_elements = sum(output_banked[index] for index in candidates)
        while candidates and candidate_elements > cache_elements:
            victim = max(
                candidates,
                key=lambda producer_index: (
                    _next_use(producer_index, layer_index),
                    output_banked[producer_index],
                    producer_index,
                ),
            )
            candidates.remove(victim)
            candidate_elements -= output_banked[victim]
            evicted.append(victim)
            if victim not in external_backed:
                output_spilled[victim] = True
                output_evicted_at[victim] = layer_index
                external_backed.add(victim)

        if layer_index not in candidates:
            output_spilled[layer_index] = True
            external_backed.add(layer_index)

        resident = candidates
        steps.append({
            'layer_index': layer_index,
            'logical_live_producer_indices': logical_live_producers,
            'logical_cache_required_elems': logical_required,
            'execution_cache_required_elems': execution_required,
            'logical_cache_fits': execution_required <= cache_elements,
            'cache_rule': workspace.get('rule'),
            'input_output_overlap_elems': (
                workspace.get('shared_banks', 0) * bank_size
                if workspace.get('reuses_input') else 0
            ),
            'pipeline_boundary_elems': (
                workspace.get('pipeline_boundary_banks', 0) * bank_size
            ),
            'jumpback_elems': workspace.get('jumpback_banks', 0) * bank_size,
            'resident_before_producer_indices': sorted(resident_before),
            'resident_before_execution_eviction_indices': sorted(
                resident_before_execution_eviction
            ),
            'resident_after_producer_indices': sorted(resident),
            'resident_after_elems': candidate_elements,
            'evicted_producer_indices': evicted,
            'input_transfer_producer_indices': sorted(
                set(missing_producer_sizes) | set(missing_residual_producer_sizes)
            ),
            'input_transfer_elems': (
                sum(missing_producer_sizes.values())
                + sum(missing_residual_producer_sizes.values())
                + sum(external_input_sizes.values())
                + sum(external_residual_sizes.values())
            ),
            'residual_input_transfer_elems': (
                sum(missing_residual_producer_sizes.values())
                + sum(external_residual_sizes.values())
            ),
            'model_state_transfer_elems': sum(model_state_sizes.values()),
        })

    for layer_index, step in enumerate(steps):
        step['output_spilled'] = output_spilled[layer_index]
        step['output_evicted_at'] = output_evicted_at[layer_index]
        step['stay_on_chip'] = not output_spilled[layer_index]

    return {
        'steps': steps,
        'consumers_by_producer': consumers_by_producer,
        'last_consumer_by_producer': last_consumer,
        'output_banked_elems': output_banked,
        'policy': 'producer_consumer_lifetime_rule_aware_farthest_next_use',
    }


def _compute_layer_cycles(layer: dict) -> float:
    """Compute cycles with 128-wide chunks, including collapsed children."""
    l_type = layer['type']
    if is_registry_activation(l_type):
        return 0.0

    compute_cycles = 0

    if 'Conv' in l_type:
        in_c = layer.get('in_channels', 0)
        groups = layer.get('groups', 1)
        fh = layer.get('filter_height', 0)
        fw = layer.get('filter_width', 0)
        output_elems = layer.get('output_elems', 0)
        reduction_dim = (in_c / groups) * fh * fw if groups else 0
        compute_cycles = _cycles_for_reduction_outputs(output_elems, reduction_dim)
    elif 'Linear' in l_type:
        in_features = layer.get('in_features', 0)
        out_features = layer.get('out_features', 0)
        weight_elems = layer.get('weight_elems', 0)
        if not in_features and out_features:
            in_features = weight_elems / out_features
        output_elems = layer.get('output_elems', 0)
        compute_cycles = _cycles_for_reduction_outputs(output_elems, in_features)
        if compute_cycles == 0 and weight_elems:
            compute_cycles = _cycles_for_reduction_outputs(out_features or 1, in_features or weight_elems)
    elif l_type in ('QuantMatMul', 'QuantBMM'):
        input_shapes = layer.get('input_shapes', [])
        output_shape = layer.get('output_shape')
        output_elems = layer.get('output_elems', _numel_from_shape(output_shape))
        reduction_dim = input_shapes[0][-1] if input_shapes and input_shapes[0] else 0
        compute_cycles = _cycles_for_reduction_outputs(output_elems, reduction_dim)
    elif l_type == 'QuantAdd':
        add_passes = _quant_add_operation_count(layer)
        compute_cycles = add_passes * _cycles_for_ops(layer.get('output_elems', 0))
    elif l_type in ('QuantSub', 'QuantMul', 'QuantDiv', 'Residual'):
        compute_cycles = _cycles_for_ops(layer.get('output_elems', 0))
    elif l_type == 'QuantCat':
        compute_cycles = 0
    else:
        in_elems = layer.get('input_elems', 0)
        out_elems = layer.get('output_elems', 0)
        compute_cycles = _cycles_for_ops(max(in_elems, out_elems))

    for collapsed in layer.get('collapsed_layers', []):
        if is_registry_activation(collapsed.get('type', '')):
            continue
        collapsed_out_elems = collapsed.get('output_elems', 0)
        compute_cycles += _cycles_for_ops(collapsed_out_elems)

    return float(compute_cycles)


def _quant_add_connection_count(layer: dict) -> int:
    if layer.get('type') != 'QuantAdd':
        return 0
    num_inputs = len(layer.get('input_shapes', []))
    return max(1, num_inputs)


def _quant_add_operation_count(layer: dict) -> int:
    """Return the number of binary additions required for the captured inputs."""
    return max(1, _quant_add_connection_count(layer) - 1)


def optimize_layer_bits(layer: dict, bandwidth: float,
                        need_input_transfer: bool,
                        need_weight_transfer: bool,
                        need_output_transfer: bool,
                        min_bits: int = 3, max_bits: int = 8,
                        fixed_transfers: list[dict] = None,
                        forced_bits: dict = None,
                        input_transfer_elems: int = None):
    """
    Layer-wide bit-width optimization for BW-limited transfers.

    Starting from *max_bits* for every transferred component, if the layer is
    overall BW-limited (total_transfer > compute), all non-forced transferred
    components are reduced by 1 bit simultaneously. Repeats until the layer
    becomes compute-limited or all reducible components reach *min_bits*.

    fixed_transfers model side-band transfers that consume bandwidth but do not
    have an optimizable bit-width here, e.g. residual spill/reload at min_bits.

    Returns
    -------
    input_bits : int
    weight_bits : int
    output_bits : int
    total_cycles : float   – max(compute_cycles, total_transfer_cycles)
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be greater than zero")
    if min_bits > max_bits:
        raise ValueError("min_bits cannot exceed max_bits")

    compute = _compute_layer_cycles(layer)
    fixed_transfers = fixed_transfers or []
    forced_bits = forced_bits or {}

    def _transfer_cycles_for_elems(elems, bits):
        if elems <= 0:
            return 0.0
        num_chunks = math.ceil(elems / 128)
        bytes_per_chunk = 16 * bits
        return (num_chunks * bytes_per_chunk) / bandwidth

    def _fixed_transfer_cycles():
        total = 0.0
        for transfer in fixed_transfers:
            total += _transfer_cycles_for_elems(
                transfer.get('elems', 0),
                transfer.get('bits', min_bits),
            )
        return total

    def _transfer_cycles(name, bits):
        elems = 0
        if name == 'weight':
            if not need_weight_transfer:
                return 0.0
            elems = layer.get('weight_elems', 0)
        elif name == 'input':
            if not need_input_transfer:
                return 0.0
            elems = (
                layer.get('input_elems', 0)
                if input_transfer_elems is None
                else input_transfer_elems
            )
        elif name == 'output':
            if not need_output_transfer:
                return 0.0
            elems = layer.get('output_elems', 0)
        if elems <= 0:
            return 0.0
        bits = forced_bits.get(name, bits)
        return _transfer_cycles_for_elems(elems, bits)

    input_bits = forced_bits.get('input', max_bits)
    weight_bits = forced_bits.get('weight', max_bits)
    output_bits = forced_bits.get('output', max_bits)
    fixed_transfer_cycles = _fixed_transfer_cycles()

    while True:
        w_t = _transfer_cycles('weight', weight_bits)
        i_t = _transfer_cycles('input', input_bits)
        o_t = _transfer_cycles('output', output_bits)
        total_transfer = fixed_transfer_cycles + w_t + i_t + o_t

        if total_transfer <= compute:
            break

        reduced = False
        if need_weight_transfer and 'weight' not in forced_bits and weight_bits > min_bits:
            weight_bits -= 1
            reduced = True
        if need_input_transfer and 'input' not in forced_bits and input_bits > min_bits:
            input_bits -= 1
            reduced = True
        if need_output_transfer and 'output' not in forced_bits and output_bits > min_bits:
            output_bits -= 1
            reduced = True

        if not reduced:
            break

    total_transfer = fixed_transfer_cycles + (
        _transfer_cycles('weight', weight_bits) +
        _transfer_cycles('input', input_bits) +
        _transfer_cycles('output', output_bits)
    )
    total_cycles = max(compute, total_transfer)

    return input_bits, weight_bits, output_bits, total_cycles


def analyze_model(model_cfg_or_name, batch_size: int, device: str = "cpu", adapter_cfg: dict = None,
                  cache_elements: int = 0, bank_size: int = 0, metadata_bits: int = 0,
                  dummy_input=None, return_cache_map_layers: bool = False):
    """Trace model to get layer element counts in execution order.

    dummy_input: optional caller-supplied trace input (e.g. token ids for an
    LLM). When None, an image tensor is synthesized from resolve_model_input_size
    (the original vision behavior).
    """
    import torch.nn.functional as F
    import yaml
    from runspace.src.adapters.adapter_factory import create_adapter

    if isinstance(model_cfg_or_name, dict):
        config = {
            'model': model_cfg_or_name,
            'adapter': dict({'type': 'generic', 'build_quantized': True}, **(adapter_cfg or {}))
        }
    elif isinstance(model_cfg_or_name, str) and (model_cfg_or_name.endswith('.yaml') or model_cfg_or_name.endswith('.yml') or os.path.isfile(model_cfg_or_name)):
        with open(model_cfg_or_name, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            if isinstance(config, list):
                item = config[0]
                config = {'model': item if isinstance(item, dict) else {'name': item, 'weights': None}}
            else:
                raise ValueError(f"Loaded YAML from {model_cfg_or_name} is not a valid dictionary or list.")
        if 'model' not in config:
            config = {'model': config}
        if 'adapter' not in config:
            config['adapter'] = {}
        config['adapter']['build_quantized'] = True
    else:
        config = {
            'model': {'name': model_cfg_or_name, 'weights': None},
            'adapter': {'type': 'generic', 'build_quantized': True}
        }

    # Cache analysis records topology, tensor shapes, and transfer sizes. It is
    # not an activation-accuracy run, so make that intent explicit instead of
    # falling through to the retired module-level fake-quant path.
    adapter_config = config.setdefault('adapter', {})
    adapter_config['input_quantization'] = False
    adapter_config['output_quantization'] = False

    adapter = create_adapter(config)
    model = adapter.model
    model.eval()
    model.to(device)

    model_state_tensor_ids = {
        id(tensor) for tensor in list(model.parameters()) + list(model.buffers())
    }
    model_state_storage_keys = set()
    model_input_tensor_ids = set()
    model_input_storage_keys = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        try:
            storage_key = _tensor_storage_key(tensor)
            if storage_key is not None:
                model_state_storage_keys.add(storage_key)
        except (AttributeError, RuntimeError):
            pass

    def _is_model_state_tensor(tensor: torch.Tensor) -> bool:
        if id(tensor) in model_state_tensor_ids:
            return True
        try:
            storage_key = _tensor_storage_key(tensor)
        except (AttributeError, RuntimeError):
            return False
        return storage_key is not None and storage_key in model_state_storage_keys

    def _is_model_input_tensor(tensor: torch.Tensor) -> bool:
        if id(tensor) in model_input_tensor_ids:
            return True
        storage_key = _tensor_storage_key(tensor)
        return (
            storage_key is not None
            and storage_key in model_input_storage_keys
        )

    execution_order = []
    hooks        = []
    _scope_stack = []   # tracks innermost active module name for functional-op naming

    _residual_block_types = []
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
        _residual_block_types = [BasicBlock, Bottleneck]
    except ImportError:
        pass

    _POOL_TYPES = (
        nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
        nn.MaxPool1d, nn.AvgPool1d,
    )
    _NORM_TYPES = (
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm,
    )

    def _current_scope() -> str:
        return _scope_stack[-1] if _scope_stack else 'unknown'

    # --- Op identification ---
    # Register hooks on modules that match the DynamicInputQuantizer's hooking strategy.
    # We want to hook "leaf" operations and exclude high-level containers like DecomposedMultiheadAttention.
    supported_ops = set(OpRegistry.get_supported_ops().values())
    functional_ops = []
    # These match DynamicInputQuantizer._FUNCTIONAL_OP_NAMES
    for op_name in ["QuantMatMul", "QuantBMM", "QuantAdd", "QuantSub", "QuantMul", "QuantDiv", "QuantCat"]:
        try:
            functional_ops.append(OpRegistry.get(op_name))
        except Exception:
            continue
    
    # Filter out decomposed containers that are NOT hooked by the quantizer.
    # Note: We now exclude DecomposedMultiheadAttention as well, because we recurse into its 
    # sub-blocks (ScaledDotProduct, etc.) to hook the individual matmuls.
    EXCLUDED_CONTAINERS = ("DecomposedMlpBlock", "DecomposedQkvAttention", "DecomposedMultiheadAttention")
    quantized_types = tuple(
        cls for cls in set(list(supported_ops) + functional_ops)
        if cls.__name__ not in EXCLUDED_CONTAINERS
    )

    # --- module recording hooks ---
    tensor_producers = {}
    storage_producers = {}

    def _tensor_and_bases(tensor: torch.Tensor) -> list[torch.Tensor]:
        lineage = []
        current = tensor
        seen_ids = set()
        while isinstance(current, torch.Tensor) and id(current) not in seen_ids:
            seen_ids.add(id(current))
            lineage.append(current)
            current = getattr(current, '_base', None)
        return lineage

    def _find_tensor_producer(tensor: torch.Tensor):
        """Resolve a tensor, view base, or shared storage to its producer.

        Some models build residual branches through an unrecorded chain of
        ``reshape``/``transpose`` operations.  The final view does not always
        retain the captured tensor as its direct ``_base`` (notably after a
        container returns it), but it still refers to the same live storage.
        Preserve that allocation identity so the later consumer is not
        mistaken for a new external/model input.
        """
        for current in _tensor_and_bases(tensor):
            entry = tensor_producers.get(id(current))
            if entry is None:
                continue
            tensor_ref, producer = entry
            if tensor_ref() is current:
                return producer
        storage_key = _tensor_storage_key(tensor)
        if storage_key is not None:
            storage_entry = storage_producers.get(storage_key)
            if storage_entry is not None:
                tensor_ref, producer = storage_entry
                source_tensor = tensor_ref()
                if (
                    source_tensor is not None
                    and source_tensor.numel() == tensor.numel()
                    and _tensor_storage_key(source_tensor) == storage_key
                ):
                    return producer
        return None

    def _remember_tensor_lineage(tensor: torch.Tensor, producer: dict) -> None:
        for current in _tensor_and_bases(tensor):
            tensor_producers[id(current)] = (weakref.ref(current), producer)
            storage_key = _tensor_storage_key(current)
            if storage_key is not None:
                storage_producers[storage_key] = (
                    weakref.ref(current), producer
                )

    def _input_producer_layer_indices(value) -> list:
        return [
            producer.get('_execution_index') if producer is not None else None
            for producer in (
                _find_tensor_producer(tensor) for tensor in _collect_tensors(value)
            )
        ]

    def _input_edge_records(value) -> list[dict]:
        """Record one explicit runtime edge for every tensor operand."""
        records = []
        for input_index, tensor in enumerate(_collect_tensors(value)):
            producer = _find_tensor_producer(tensor)
            records.append({
                'input_index': input_index,
                'elements': tensor.numel(),
                'shape': tuple(tensor.shape),
                'tensor_id': id(tensor),
                'storage_key': _tensor_storage_key(tensor),
                'producer_layer_index': (
                    producer.get('_execution_index')
                    if producer is not None else None
                ),
                'producer_name': (
                    producer.get('name') if producer is not None else None
                ),
                'is_model_state': _is_model_state_tensor(tensor),
                'is_model_input': _is_model_input_tensor(tensor),
            })
        return records

    def _remember_output_producer(info: dict):
        info['_execution_index'] = len(execution_order) - 1
        parent_depths = [
            execution_order[producer_index].get('_graph_depth', 0)
            for producer_index in info.get('input_producer_layer_indices', [])
            if producer_index is not None
        ]
        info['_graph_depth'] = 1 + max(parent_depths, default=0)
        output_tensor = info.pop('_output_tensor', None)
        if output_tensor is not None:
            _remember_tensor_lineage(output_tensor, info)

    lineage_container_starts = {}

    def _lineage_container_pre_hook(module, _input):
        lineage_container_starts[id(module)] = len(execution_order)
        input_tensors = _collect_tensors(_input)
        if len(input_tensors) != 1 or not execution_order:
            return
        input_tensor = input_tensors[0]
        if _find_tensor_producer(input_tensor) is not None:
            return
        producer = execution_order[-1]
        if input_tensor.numel() != producer.get('output_elems', 0):
            return
        # A container boundary is the one safe place to bridge an unrecorded
        # layout/copy chain before its first recorded child runs.  MobileViT,
        # for example, unfolds a Conv2d output into transformer tokens using
        # reshape/transpose/reshape, with the last reshape sometimes copying.
        _remember_tensor_lineage(input_tensor, producer)

    def _lineage_container_hook(module, _input, output):
        """Carry an inner producer across unrecorded layout/copy operations."""
        start_index = lineage_container_starts.pop(id(module), len(execution_order))
        if len(execution_order) <= start_index:
            return
        producer = execution_order[-1]
        output_tensors = _collect_tensors(output)
        if len(output_tensors) != 1:
            return
        output_tensor = output_tensors[0]
        if output_tensor.numel() != producer.get('output_elems', 0):
            return
        _remember_tensor_lineage(output_tensor, producer)

    def _mark_residual_stream(info: dict, residual_input: torch.Tensor):
        residual_elems = residual_input.numel()
        residual = {
            'elems': residual_elems,
            'shape': tuple(residual_input.shape),
            'tensor_id': id(residual_input),
            'storage_key': _tensor_storage_key(residual_input),
            'producer_name': None,
        }
        info.setdefault('residual_inputs', []).append(residual)

        # Keep the original aggregate fields for the cache/bandwidth report.
        # input_elems already accounts for the first QuantAdd operand; every
        # additional operand is a separate residual stream.
        info['residual_input_elems'] = sum(
            item['elems'] for item in info['residual_inputs']
        )
        info['residual_input_stream_elems'] = info['residual_input_elems']
        if len(info['residual_inputs']) == 1:
            info['residual_input_shape'] = residual['shape']
            info['residual_input_tensor_id'] = residual['tensor_id']

        producer = _find_tensor_producer(residual_input)
        if producer is None:
            return

        residual['producer_name'] = producer['name']
        producer['residual_output_elems'] = max(
            producer.get('residual_output_elems', 0),
            residual_elems,
        )
        producer.setdefault('residual_output_consumers', []).append(info['name'])
        info['residual_producer_name'] = producer['name']

    def hook_fn(module, input, output):
        if isinstance(module, (nn.Conv2d, nn.Linear) + quantized_types):
            input_tensors = _collect_tensors(input)
            info = {
                'name':         getattr(module, 'layer_name', 'unknown'),
                'type':         module.__class__.__name__,
                'weight_elems': module.weight.numel() if getattr(module, 'weight', None) is not None else 0,
                'input_elems':  input[0].numel() if isinstance(input[0], torch.Tensor) else 0,
                'output_elems': output.numel()   if isinstance(output,   torch.Tensor) else 0,
                'input_shapes': _collect_tensor_shapes(input),
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else None,
                'input_tensor_ids': _collect_tensor_ids(input),
                'input_tensor_storage_keys': _collect_tensor_storage_keys(input),
                'input_producer_layer_indices': _input_producer_layer_indices(input),
                'input_edges': _input_edge_records(input),
                'output_tensor_id': id(output) if isinstance(output, torch.Tensor) else None,
                'output_tensor_storage_key': _tensor_storage_key(output) if isinstance(output, torch.Tensor) else None,
                '_output_tensor': output if isinstance(output, torch.Tensor) else None,
            }
            if hasattr(module, 'last_operand_count'):
                info['operand_count'] = int(module.last_operand_count)
                info['constant_operands'] = copy.deepcopy(
                    getattr(module, 'last_constant_operands', [])
                )
            if module.__class__.__name__ == 'QuantAdd' and len(input_tensors) >= 2:
                resolved_producers = {
                    id(tensor): _find_tensor_producer(tensor)
                    for tensor in input_tensors
                }
                residual_indices = _residual_input_indices(
                    input_tensors, resolved_producers, _is_model_state_tensor
                )
                for residual_index in residual_indices:
                    _mark_residual_stream(info, input_tensors[residual_index])
                # Retain this legacy field for existing result consumers.
                if info.get('residual_inputs'):
                    info['residual_producer_name'] = info['residual_inputs'][0]['producer_name']
            if isinstance(module, nn.Conv2d):
                ks = module.kernel_size
                if isinstance(ks, tuple):
                    filter_height, filter_width = ks
                else:
                    filter_height = filter_width = ks
                in_t = input[0] if isinstance(input[0], torch.Tensor) else None
                out_t = output   if isinstance(output,   torch.Tensor) else None

                info['in_channels']            = module.in_channels
                info['out_channels']           = module.out_channels
                info['filter_height']          = filter_height
                info['filter_width']           = filter_width
                info['kernel_size']            = ks
                info['stride']                 = module.stride
                info['padding']                = module.padding
                info['dilation']               = module.dilation
                info['padding_mode']           = module.padding_mode
                info['groups']                 = module.groups
                info['input_channel_height']   = in_t.shape[-2] if in_t is not None and in_t.ndim >= 4 else 0
                info['input_channel_width']    = in_t.shape[-1] if in_t is not None and in_t.ndim >= 4 else 0
                info['output_channel_height']  = out_t.shape[-2] if out_t is not None and out_t.ndim >= 4 else 0
                info['output_channel_width']   = out_t.shape[-1] if out_t is not None and out_t.ndim >= 4 else 0
                info['jump_back_size_in_banks'] = round_to_banks(info['filter_width'] * info['in_channels'] * (info['input_channel_width'])//128 * 128, bank_size)
            elif isinstance(module, nn.Linear):
                info['in_features']  = module.in_features
                info['out_features'] = module.out_features
            # Quantized pooling and normalization modules are included in
            # ``quantized_types`` above, so collect their operation attributes
            # here as well as in the plain PyTorch branches below.
            for attribute in (
                'kernel_size', 'stride', 'padding', 'dilation',
                'padding_mode', 'groups', 'in_channels', 'out_channels',
                'in_features', 'out_features', 'ceil_mode', 'output_size',
                'normalized_shape', 'eps', 'num_features', 'num_groups',
                'num_channels',
            ):
                if hasattr(module, attribute):
                    info[attribute] = getattr(module, attribute)
            execution_order.append(info)
            _remember_output_producer(info)
        elif isinstance(module, _POOL_TYPES):
            info = {
                'name':         getattr(module, 'layer_name', 'unknown'),
                'type':         module.__class__.__name__,
                'weight_elems': 0,
                'input_elems':  input[0].numel() if isinstance(input[0], torch.Tensor) else 0,
                'output_elems': output.numel()   if isinstance(output,   torch.Tensor) else 0,
                'input_shapes': _collect_tensor_shapes(input),
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else None,
                'input_tensor_ids': _collect_tensor_ids(input),
                'input_tensor_storage_keys': _collect_tensor_storage_keys(input),
                'input_producer_layer_indices': _input_producer_layer_indices(input),
                'input_edges': _input_edge_records(input),
                'output_tensor_id': id(output) if isinstance(output, torch.Tensor) else None,
                'output_tensor_storage_key': _tensor_storage_key(output) if isinstance(output, torch.Tensor) else None,
                '_output_tensor': output if isinstance(output, torch.Tensor) else None,
            }
            for attribute in (
                'kernel_size', 'stride', 'padding', 'dilation',
                'ceil_mode', 'output_size',
            ):
                if hasattr(module, attribute):
                    info[attribute] = getattr(module, attribute)
            execution_order.append(info)
            _remember_output_producer(info)
        elif isinstance(module, _NORM_TYPES):
            in_t  = input[0] if isinstance(input[0], torch.Tensor) else None
            out_t = output   if isinstance(output,   torch.Tensor) else None
            wt    = module.weight.numel() if getattr(module, 'weight', None) is not None else 0
            info = {
                'name':         getattr(module, 'layer_name', 'unknown'),
                'type':         module.__class__.__name__,
                'weight_elems': wt,
                'input_elems':  in_t.numel()  if in_t  is not None else 0,
                'output_elems': out_t.numel() if out_t is not None else 0,
                'input_shapes': _collect_tensor_shapes(input),
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else None,
                'input_tensor_ids': _collect_tensor_ids(input),
                'input_tensor_storage_keys': _collect_tensor_storage_keys(input),
                'input_producer_layer_indices': _input_producer_layer_indices(input),
                'input_edges': _input_edge_records(input),
                'output_tensor_id': id(output) if isinstance(output, torch.Tensor) else None,
                'output_tensor_storage_key': _tensor_storage_key(output) if isinstance(output, torch.Tensor) else None,
                '_output_tensor': output if isinstance(output, torch.Tensor) else None,
            }
            for attribute in (
                'normalized_shape', 'eps', 'num_features', 'num_groups',
                'num_channels',
            ):
                if hasattr(module, attribute):
                    info[attribute] = getattr(module, attribute)
            execution_order.append(info)
            _remember_output_producer(info)

    def residual_hook_fn(module, input, output):
        skip = input[0] if isinstance(input[0], torch.Tensor) else None
        info = {
            'name':           getattr(module, 'layer_name', 'unknown'),
            'type':           'Residual',
            'weight_elems':   0,
            'input_elems':    skip.numel() if skip is not None else 0,
            'output_elems':   output.numel() if isinstance(output, torch.Tensor) else 0,
            'input_shapes':    _collect_tensor_shapes(input),
            'output_shape':    tuple(output.shape) if isinstance(output, torch.Tensor) else None,
            'input_tensor_ids': _collect_tensor_ids(input),
            'input_tensor_storage_keys': _collect_tensor_storage_keys(input),
            'input_producer_layer_indices': _input_producer_layer_indices(input),
            'input_edges': _input_edge_records(input),
            'output_tensor_id': id(output) if isinstance(output, torch.Tensor) else None,
            'output_tensor_storage_key': _tensor_storage_key(output) if isinstance(output, torch.Tensor) else None,
            '_output_tensor': output if isinstance(output, torch.Tensor) else None,
            'has_downsample': module.downsample is not None,
        }
        execution_order.append(info)
        _remember_output_producer(info)

    # --- scope tracking: push/pop for every module so functional ops get sensible names ---
    for name, module in model.named_modules():
        def _make_scope_hooks(n):
            def _pre(mod, inp):
                _scope_stack.append(n)
            def _post(mod, inp, out):
                if _scope_stack and _scope_stack[-1] == n:
                    _scope_stack.pop()
            return _pre, _post
        _pre, _post = _make_scope_hooks(name)
        hooks.append(module.register_forward_pre_hook(_pre))
        hooks.append(module.register_forward_hook(_post))

    # --- recording hooks (registered after scope hooks so scope is still live during forward) ---

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, _POOL_TYPES, _NORM_TYPES) + quantized_types):
            module.layer_name = name
            hooks.append(module.register_forward_hook(hook_fn))
        elif _residual_block_types and isinstance(module, tuple(_residual_block_types)):
            module.layer_name = name
            hooks.append(module.register_forward_hook(residual_hook_fn))
        is_layout_lineage_container = (
            type(module).__name__ in {
                'AttentionWeightedValues', 'ScaledDotProduct'
            }
            # Recursive quantization flattens MobileViT's Sequential into a
            # plain ``nn.Module`` container while preserving this name.
            or name.rsplit('.', 1)[-1] == 'transformer'
        )
        if is_layout_lineage_container:
            hooks.append(module.register_forward_pre_hook(_lineage_container_pre_hook))
            hooks.append(module.register_forward_hook(_lineage_container_hook))

    # --- functional op patching ---
    # Patches Python-level calls to torch.matmul / torch.bmm / softmax.
    # Catches custom attention implementations (e.g. MobileViT) that use these directly.
    # nn.MultiheadAttention's C++ fast path is handled above by mha_hook_fn instead.
    # nn.Linear uses F.linear → torch.addmm (not matmul), so no double-counting.
    if dummy_input is None:
        from runspace.src.utils.model_input_utils import resolve_model_input_size
        input_shape = resolve_model_input_size(model, batch_size=batch_size)
        dummy_input = torch.randn(*input_shape).to(device)
    else:
        dummy_input = dummy_input.to(device)

    for tensor in _collect_tensors(dummy_input):
        model_input_tensor_ids.add(id(tensor))
        storage_key = _tensor_storage_key(tensor)
        if storage_key is not None:
            model_input_storage_keys.add(storage_key)

    try:
        with torch.no_grad():
            try:
                # Try standard single tensor input first
                model(dummy_input)
            except Exception as e1:
                try:
                    # Fallback for models requiring (x, None) tuple
                    model((dummy_input, None))
                except Exception as e2:
                    print(f"Warning: Dummy forward failed with both tensor and tuple.")
                    print(f"  Tensor error: {e1}")
                    print(f"  Tuple error: {e2}")
    finally:
        for h in hooks:
            h.remove()

    _repair_unrecorded_layout_lineage(execution_order)
    _repair_lineage_from_prior_inputs(execution_order)
    _rebuild_residual_metadata_from_edges(execution_order)
    execution_order = fuse_pipeline_activations(execution_order)
    cache_map_order = copy.deepcopy(execution_order)

    if return_cache_map_layers:
        return execution_order, cache_map_order
    return execution_order


def _invert_layer_rules() -> dict:
    """Build rule_name -> list of layer types that include it (from LAYER_RULES)."""
    applies: dict = {}
    for layer_type, rule_keys in LAYER_RULES.items():
        for key in rule_keys:
            applies.setdefault(key, [])
            if layer_type not in applies[key]:
                applies[key].append(layer_type)
    return applies


def serialize_rules() -> list:
    """Return serializable rule metadata in RULES insertion order.

    'applies_to' is derived by inverting LAYER_RULES, so it stays in sync
    with LAYER_RULES without any duplication.
    """
    applies = _invert_layer_rules()
    result  = []
    for name, rule in RULES.items():
        layer_types = applies.get(name, [])
        if '__default__' in layer_types:
            applies_to = 'All layers'
        else:
            applies_to = ', '.join(layer_types) or 'N/A'
        result.append({
            'name':            name,
            'on_chip':         rule['on_chip'],
            'xin_from_cache':  rule['xin_from_cache'],
            'applies_to':      applies_to,
            'stay_condition':  rule.get('stay_condition', ''),
            'permanents':      rule.get('permanents', ''),
            'pipeline_banks':  rule.get('pipeline_banks', 0),
            'notes':           rule.get('notes', ''),
        })
    return result


def serialize_producer_consumer_policy() -> list:
    """Return dashboard metadata for the lifetime-based cache policy."""
    common = {
        'applies_to': 'All runtime tensor producers',
        'stay_condition': 'Live activation buffers fit after farthest-next-use eviction',
        'permanents': 'Outputs with future consumers; released after final consumer',
        'pipeline_banks': 0,
    }
    return [
        {
            'name': 'producer_consumer_resident',
            'on_chip': True,
            'xin_from_cache': True,
            'notes': (
                'Output remains resident through its consumers unless a later '
                'capacity conflict evicts it.'
            ),
            **common,
        },
        {
            'name': 'producer_consumer_spill',
            'on_chip': False,
            'xin_from_cache': False,
            'notes': (
                'Output is written externally because it has no future '
                'consumer, exceeds capacity, or is selected for eviction.'
            ),
            **common,
        },
    ]


def _greedy_stream_connections(
    base_banks_by_layer: list[int],
    connection_plans: list[dict],
    capacity_banks: int,
    streaming_banks: int = 2,
    rule_workspaces: list[dict] = None,
) -> dict:
    """Greedily replace resident lifetimes with streaming buffers.

    Candidates are ranked by the number of currently overflowing layers they
    make fit, then by smaller bank-rounded allocation, then by original cache
    map column order.
    """
    streamed = set()
    choices = []
    rule_workspaces = rule_workspaces or []

    def _totals(streamed_indices: set[int]) -> list[int]:
        totals = list(base_banks_by_layer)
        for connection_index, connection in enumerate(connection_plans):
            if connection_index in streamed_indices:
                for layer_index in connection['stream_layer_indices']:
                    totals[layer_index] += streaming_banks
            else:
                for layer_index in connection['resident_layer_indices']:
                    totals[layer_index] += connection['bank_count']
        for layer_index, workspace in enumerate(rule_workspaces):
            available_candidates = []
            for candidate in workspace.get('reuse_candidates', []):
                connection_index = candidate.get('connection_index')
                if connection_index is None:
                    available_candidates.append(candidate)
                    continue
                if connection_index in streamed_indices:
                    continue
                if layer_index in connection_plans[connection_index][
                    'resident_layer_indices'
                ]:
                    available_candidates.append(candidate)
            if available_candidates:
                selected_candidate = max(
                    available_candidates,
                    key=lambda candidate: candidate.get('shared_banks', 0),
                )
                totals[layer_index] += workspace.get('overhead_banks', 0)
                totals[layer_index] -= selected_candidate.get(
                    'shared_banks', 0
                )
        return totals

    while True:
        current_totals = _totals(streamed)
        currently_red = {
            layer_index for layer_index, total in enumerate(current_totals)
            if total > capacity_banks
        }
        if not currently_red:
            break

        ranked_candidates = []
        for connection_index, connection in enumerate(connection_plans):
            if connection_index in streamed:
                continue
            trial_totals = _totals(streamed | {connection_index})
            solved_layers = sorted(
                layer_index for layer_index in currently_red
                if trial_totals[layer_index] <= capacity_banks
            )
            if not solved_layers:
                continue
            ranked_candidates.append((
                -len(solved_layers),
                connection['bank_count'],
                connection_index,
                solved_layers,
            ))

        if not ranked_candidates:
            break
        _, _, selected_index, solved_layers = min(ranked_candidates)
        streamed.add(selected_index)
        choices.append({
            'connection_index': selected_index,
            'connection_name': connection_plans[selected_index]['name'],
            'solved_layer_indices': solved_layers,
            'solved_layer_count': len(solved_layers),
            'bank_count': connection_plans[selected_index]['bank_count'],
        })

    final_totals = _totals(streamed)
    return {
        'streamed_connection_indices': sorted(streamed),
        'choices': choices,
        'total_banks_by_layer': final_totals,
        'green_layer_indices': [
            index for index, total in enumerate(final_totals)
            if total <= capacity_banks
        ],
        'red_layer_indices': [
            index for index, total in enumerate(final_totals)
            if total > capacity_banks
        ],
    }


def build_cache_map(
    layers: list[dict],
    cache_elements: int = None,
    bank_size: int = None,
    metadata_bits: int = 0,
    streaming_banks: int = 2,
) -> dict:
    """Build a layer-by-layer element-count matrix for activation streams.

    ``x_in`` and ``x_out`` are the current layer's tensor sizes. Residual skips
    receive ``residual_N`` columns. Any other produced tensor with multiple or
    non-adjacent consumers receives an automatically named ``hold_N`` column.
    Storage identity is used in addition to Python object identity so views
    such as reshaped/transposed Q, K, and V tensors keep their producer.

    When cache and bank sizes are supplied, every allocation is rounded to a
    whole bank and a greedy pass selects residual/hold lifetimes to stream.
    """
    layer_indices_by_name = {}
    for index, layer in enumerate(layers):
        layer_indices_by_name.setdefault(layer.get('name'), []).append(index)

    residuals_by_layer = {}
    for consumer_index, layer in enumerate(layers):
        residual_inputs = list(layer.get('residual_inputs', []))
        if not residual_inputs and layer.get('residual_input_stream_elems', 0):
            # Compatibility with layer dictionaries produced before detailed
            # residual-edge tracking was introduced.
            residual_inputs = [{
                'elems': layer['residual_input_stream_elems'],
                'producer_name': layer.get('residual_producer_name'),
            }]
        residuals_by_layer[consumer_index] = residual_inputs

    active_by_id = {}
    active_by_storage = {}
    records_by_producer_index = {}
    input_record_indices_by_layer = {}
    primary_input_resolved_by_layer = {}
    primary_input_record_index_by_layer = {}
    tensor_records = []
    for layer_index, layer in enumerate(layers):
        seen_records = set()
        if 'input_producer_layer_indices' in layer:
            input_records = [
                records_by_producer_index.get(producer_index)
                for producer_index in layer['input_producer_layer_indices']
            ]
        else:
            input_ids = layer.get('input_tensor_ids', [])
            input_storage_keys = layer.get('input_tensor_storage_keys', [])
            input_records = []
            for input_position in range(max(len(input_ids), len(input_storage_keys))):
                tensor_id = input_ids[input_position] if input_position < len(input_ids) else None
                storage_key = (
                    input_storage_keys[input_position]
                    if input_position < len(input_storage_keys) else None
                )
                record = active_by_id.get(tensor_id) if tensor_id is not None else None
                if record is None and storage_key is not None:
                    record = active_by_storage.get(storage_key)
                input_records.append(record)

        primary_input_resolved_by_layer[layer_index] = bool(
            input_records and input_records[0] is not None
        )
        primary_input_record_index_by_layer[layer_index] = (
            input_records[0]['record_index']
            if input_records and input_records[0] is not None else None
        )
        for record in input_records:
            if record is None or record['record_index'] in seen_records:
                continue
            if record['producer_layer_index'] >= layer_index:
                continue
            record['consumer_layer_indices'].append(layer_index)
            record['consumers'].append(layer.get('name', 'unknown'))
            seen_records.add(record['record_index'])
        input_record_indices_by_layer[layer_index] = set(seen_records)

        output_id = layer.get('output_tensor_id')
        output_storage_key = layer.get('output_tensor_storage_key')
        if output_id is None and output_storage_key is None:
            continue
        record = {
            'record_index': len(tensor_records),
            'tensor_id': output_id,
            'storage_key': output_storage_key,
            'producer': layer.get('name', 'unknown'),
            'producer_layer_index': layer_index,
            'elements': int(layer.get('output_elems', 0)),
            'consumer_layer_indices': [],
            'consumers': [],
        }
        tensor_records.append(record)
        records_by_producer_index[layer_index] = record
        if output_id is not None:
            active_by_id[output_id] = record
        if output_storage_key is not None:
            active_by_storage[output_storage_key] = record

    matched_residuals = set()
    retained_tensors = []
    for record in tensor_records:
        for consumer_index in record['consumer_layer_indices']:
            for residual_index, residual in enumerate(residuals_by_layer[consumer_index]):
                id_matches = (
                    residual.get('tensor_id') is not None
                    and residual.get('tensor_id') == record['tensor_id']
                )
                storage_matches = (
                    residual.get('storage_key') is not None
                    and residual.get('storage_key') == record['storage_key']
                )
                producer_matches = (
                    residual.get('producer_layer_index')
                    == record['producer_layer_index']
                    if residual.get('producer_layer_index') is not None
                    else (
                        residual.get('producer_name') is not None
                        and residual.get('producer_name') == record['producer']
                    )
                )
                if id_matches or storage_matches or producer_matches:
                    record['is_residual'] = True
                    residual_consumers = record.setdefault(
                        'residual_consumer_layer_indices', []
                    )
                    if consumer_index not in residual_consumers:
                        residual_consumers.append(consumer_index)
                    matched_residuals.add((consumer_index, residual_index))

        consumers = record['consumer_layer_indices']
        has_non_adjacent_consumer = any(
            consumer_index > record['producer_layer_index'] + 1
            for consumer_index in consumers
        )
        if consumers and (
            record.get('is_residual')
            or len(consumers) > 1
            or has_non_adjacent_consumer
        ):
            retained_tensors.append(record)

    residual_connections = [
        record for record in retained_tensors if record.get('is_residual')
    ]
    held_connections = [
        record for record in retained_tensors if not record.get('is_residual')
    ]

    for residual_number, connection in enumerate(residual_connections):
        connection['name'] = f"residual_{residual_number}"
        connection['kind'] = 'residual'
        connection['consumer_layer_index'] = max(connection['consumer_layer_indices'])

    def _short_name(name: str) -> str:
        short_name = str(name).rsplit('.', 1)[-1]
        return ''.join(
            character if character.isalnum() else '_'
            for character in short_name
        ).strip('_') or 'unknown'

    for hold_number, connection in enumerate(held_connections):
        consumer_names = []
        for consumer in connection['consumers']:
            short_consumer = _short_name(consumer)
            if short_consumer not in consumer_names:
                consumer_names.append(short_consumer)
        consumer_label = '_'.join(consumer_names)
        connection['name'] = (
            f"hold_{hold_number}_{_short_name(connection['producer'])}"
            f"_to_{consumer_label}"
        )
        connection['kind'] = 'hold'
        connection['consumer_layer_index'] = max(connection['consumer_layer_indices'])

    # Preserve residuals whose producer is outside the captured layer set,
    # such as a model-input bypass around the first captured operation.
    for consumer_index, residual_inputs in residuals_by_layer.items():
        for residual_index, residual in enumerate(residual_inputs):
            if (consumer_index, residual_index) in matched_residuals:
                continue
            producer_name = residual.get('producer_name')
            producer_index = residual.get('producer_layer_index')
            if producer_index is None:
                producer_candidates = [
                    index
                    for index in layer_indices_by_name.get(producer_name, [])
                    if index <= consumer_index
                ]
                producer_index = (
                    producer_candidates[-1] if producer_candidates else 0
                )
            residual_connections.append({
                'name': f"residual_{len(residual_connections)}",
                'kind': 'residual',
                'producer': producer_name,
                'consumers': [layers[consumer_index].get('name', 'unknown')],
                'producer_layer_index': producer_index,
                'consumer_layer_index': consumer_index,
                'consumer_layer_indices': [consumer_index],
                'elements': int(residual.get('elems', 0)),
            })

    connections = residual_connections + held_connections
    columns = ['x_in', 'x_out', 'total_cache_needed_kb'] + [
        connection['name'] for connection in connections
    ]

    records_by_index = {
        record['record_index']: record for record in tensor_records
    }

    def _kb(elements: int):
        if not elements:
            return 0
        return round(elements / 1_000.0, 3)

    def _connection_is_visible(connection: dict, layer_index: int) -> bool:
        # A captured tensor is displayed as x_out on its producer row, then
        # moves into its named lifetime column on the following row. Legacy
        # connections without a captured producer remain inclusive.
        first_visible_layer = connection['producer_layer_index']
        if connection.get('record_index') is not None:
            first_visible_layer += 1
        is_visible = (
            first_visible_layer
            <= layer_index
            <= connection['consumer_layer_index']
        )
        if not is_visible:
            return False

        # A projected ResNet skip is scheduled after the main branch. Keep the
        # main result in a hold column while that projection runs, then display
        # it as x_in at the add itself; the skip remains in residual_N.
        is_primary_add_input = (
            connection.get('kind') == 'hold'
            and layer_index == connection['consumer_layer_index']
            and layers[layer_index].get('type') == 'QuantAdd'
            and connection.get('record_index')
            == primary_input_record_index_by_layer.get(layer_index)
        )
        return not is_primary_add_input

    rows = []
    baseline_bank_totals = []

    def _bank_count(elements: int) -> int:
        if not elements:
            return 0
        if not bank_size:
            return 0
        footprint = get_footprint_elements(int(elements), metadata_bits)
        return math.ceil(footprint / bank_size)

    for layer_index, layer in enumerate(layers):
        live_connections = [
            connection for connection in connections
            if _connection_is_visible(connection, layer_index)
        ]
        live_record_indices = set(input_record_indices_by_layer.get(layer_index, set()))
        output_record = records_by_producer_index.get(layer_index)
        if output_record is not None:
            live_record_indices.add(output_record['record_index'])
        for connection in live_connections:
            if connection.get('record_index') is not None:
                live_record_indices.add(connection['record_index'])

        unique_allocations = {}
        for record_index in live_record_indices:
            record = records_by_index[record_index]
            allocation_key = (
                ('storage', record['storage_key'])
                if record.get('storage_key') is not None
                else ('record', record_index)
            )
            unique_allocations[allocation_key] = max(
                unique_allocations.get(allocation_key, 0),
                record['elements'],
            )

        allocation_element_sizes = list(unique_allocations.values())
        total_cache_elements = sum(allocation_element_sizes)
        if not primary_input_resolved_by_layer.get(layer_index, False):
            unresolved_input_elements = int(layer.get('input_elems', 0))
            total_cache_elements += unresolved_input_elements
            allocation_element_sizes.append(unresolved_input_elements)
        if output_record is None:
            uncaptured_output_elements = int(layer.get('output_elems', 0))
            total_cache_elements += uncaptured_output_elements
            allocation_element_sizes.append(uncaptured_output_elements)
        for connection in live_connections:
            if connection.get('record_index') is None:
                total_cache_elements += connection['elements']
                allocation_element_sizes.append(connection['elements'])
        if bank_size:
            baseline_bank_totals.append(sum(
                _bank_count(elements) for elements in allocation_element_sizes
            ))

        live_connection_record_indices = {
            connection['record_index'] for connection in live_connections
            if connection.get('record_index') is not None
        }
        primary_input_is_held = (
            primary_input_record_index_by_layer.get(layer_index)
            in live_connection_record_indices
        )

        row = {
            'layer': layer.get('name', 'unknown'),
            'x_in': 0 if primary_input_is_held else _kb(int(layer.get('input_elems', 0))),
            'x_out': _kb(int(layer.get('output_elems', 0))),
            'total_cache_needed_kb': _kb(total_cache_elements),
        }
        for connection in connections:
            is_live = _connection_is_visible(connection, layer_index)
            row[connection['name']] = _kb(connection['elements']) if is_live else 0
        rows.append(row)

    optimization = None
    if cache_elements is not None and bank_size and bank_size > 0:
        capacity_banks = max(0, int(cache_elements) // int(bank_size))
        connection_plans = []
        for connection_index, connection in enumerate(connections):
            first_resident_layer = connection['producer_layer_index']
            if connection.get('record_index') is not None:
                first_resident_layer += 1
            resident_layer_indices = list(range(
                max(0, first_resident_layer),
                min(len(layers), connection['consumer_layer_index'] + 1),
            ))
            stream_consumer_indices = (
                connection.get('residual_consumer_layer_indices', [])
                if connection.get('kind') == 'residual'
                else connection.get('consumer_layer_indices', [])
            )
            stream_layer_indices = sorted({
                layer_index for layer_index in (
                    [connection['producer_layer_index']]
                    + list(stream_consumer_indices)
                )
                if 0 <= layer_index < len(layers)
            })
            connection_plan = {
                'name': connection['name'],
                'connection_index': connection_index,
                'bank_count': _bank_count(connection['elements']),
                'resident_layer_indices': resident_layer_indices,
                'stream_layer_indices': stream_layer_indices,
            }
            connection_plans.append(connection_plan)
            connection['bank_count'] = connection_plan['bank_count']
            connection['resident_layer_indices'] = resident_layer_indices
            connection['stream_layer_indices'] = stream_layer_indices

        connection_index_by_record = {
            connection['record_index']: connection_index
            for connection_index, connection in enumerate(connections)
            if connection.get('record_index') is not None
        }
        rule_workspaces = []
        for layer_index, layer in enumerate(layers):
            primary_record_index = primary_input_record_index_by_layer.get(
                layer_index
            )
            primary_record = records_by_index.get(primary_record_index)
            output_record = records_by_producer_index.get(layer_index)
            unresolved_primary_is_reusable = (
                primary_record is None
                and not primary_input_resolved_by_layer.get(layer_index, False)
                and bool(layer.get('input_elems', 0))
                and not (
                    layer.get('input_edges')
                    and layer['input_edges'][0].get('is_model_state')
                )
            )
            possible_record_indices = (
                input_record_indices_by_layer.get(layer_index, set())
                if layer.get('type') == 'QuantAdd'
                else {primary_record_index}
            )
            reuse_candidates = []
            for record_index in possible_record_indices:
                record = records_by_index.get(record_index)
                consumers = (
                    record.get('consumer_layer_indices', [])
                    if record is not None else []
                )
                if not consumers or max(consumers) != layer_index:
                    continue
                reuse_candidates.append({
                    'record_index': record_index,
                    'connection_index': connection_index_by_record.get(
                        record_index
                    ),
                    'shared_banks': min(
                        _bank_count(record.get('elements', 0)),
                        _bank_count(layer.get('output_elems', 0)),
                    ),
                })
            if unresolved_primary_is_reusable:
                reuse_candidates.append({
                    'record_index': None,
                    'connection_index': None,
                    'shared_banks': min(
                        _bank_count(layer.get('input_elems', 0)),
                        _bank_count(layer.get('output_elems', 0)),
                    ),
                })

            rule_input_banks = max(
                [candidate['shared_banks'] for candidate in reuse_candidates]
                or [_bank_count(layer.get('input_elems', 0))]
            )
            workspace = _rule_aware_workspace(
                layer,
                rule_input_banks,
                _bank_count(layer.get('output_elems', 0)),
                bank_size,
                input_is_reusable=bool(reuse_candidates),
                # Normalize trace aliases into independent input/output
                # components below, then let the hardware rule add the
                # permitted overlap back explicitly.
                input_output_already_shared=False,
            )
            if not workspace.get('reuses_input'):
                reuse_candidates = []

            trace_shared_banks = 0
            if output_record is not None:
                for record_index in input_record_indices_by_layer.get(
                    layer_index, set()
                ):
                    record = records_by_index.get(record_index)
                    if (
                        record is not None
                        and record.get('storage_key') is not None
                        and record.get('storage_key')
                        == output_record.get('storage_key')
                    ):
                        trace_shared_banks = max(
                            trace_shared_banks,
                            min(
                                _bank_count(record.get('elements', 0)),
                                _bank_count(layer.get('output_elems', 0)),
                            ),
                        )
            workspace.update({
                'primary_record_index': primary_record_index,
                'primary_connection_index': connection_index_by_record.get(
                    primary_record_index
                ),
                'reuse_candidates': reuse_candidates,
                'trace_shared_banks': trace_shared_banks,
            })
            rule_workspaces.append(workspace)

        base_banks_by_layer = list(baseline_bank_totals)
        for layer_index, workspace in enumerate(rule_workspaces):
            base_banks_by_layer[layer_index] += workspace.get(
                'trace_shared_banks', 0
            )
        for connection_plan in connection_plans:
            for layer_index in connection_plan['resident_layer_indices']:
                base_banks_by_layer[layer_index] = max(
                    0,
                    base_banks_by_layer[layer_index]
                    - connection_plan['bank_count'],
                )
        for layer_index, layer in enumerate(layers):
            has_streamed_model_data = bool(layer.get('weight_elems', 0)) or any(
                edge.get('is_model_state')
                for edge in layer.get('input_edges', [])
            )
            if has_streamed_model_data:
                base_banks_by_layer[layer_index] += streaming_banks

        optimization = _greedy_stream_connections(
            base_banks_by_layer,
            connection_plans,
            capacity_banks,
            streaming_banks=streaming_banks,
            rule_workspaces=rule_workspaces,
        )
        streamed_indices = set(optimization['streamed_connection_indices'])

        for connection_index, connection in enumerate(connections):
            connection['placement'] = (
                'streamed' if connection_index in streamed_indices else 'resident'
            )
        for choice in optimization['choices']:
            choice['solved_layers'] = [
                layers[index].get('name', 'unknown')
                for index in choice['solved_layer_indices']
            ]

        def _bank_kb(bank_count: int):
            return _kb(int(bank_count) * int(bank_size))

        optimized_rows = []
        for layer_index, layer in enumerate(layers):
            primary_record_index = primary_input_record_index_by_layer.get(
                layer_index
            )
            primary_connection_index = connection_index_by_record.get(
                primary_record_index
            )
            primary_is_streamed = (
                primary_connection_index in streamed_indices
                if primary_connection_index is not None else False
            )
            primary_streams_here = (
                primary_is_streamed
                and layer_index in connection_plans[
                    primary_connection_index
                ]['stream_layer_indices']
            )
            primary_is_resident_connection = (
                primary_connection_index is not None
                and not primary_is_streamed
                and _connection_is_visible(
                    connections[primary_connection_index], layer_index
                )
            )
            workspace = rule_workspaces[layer_index]
            available_reuse_candidates = []
            for candidate in workspace.get('reuse_candidates', []):
                candidate_connection_index = candidate.get('connection_index')
                if candidate_connection_index is None:
                    available_reuse_candidates.append(candidate)
                elif (
                    candidate_connection_index not in streamed_indices
                    and layer_index in connection_plans[
                        candidate_connection_index
                    ]['resident_layer_indices']
                ):
                    available_reuse_candidates.append(candidate)
            selected_reuse_candidate = (
                max(
                    available_reuse_candidates,
                    key=lambda candidate: candidate.get('shared_banks', 0),
                )
                if available_reuse_candidates else None
            )
            workspace_is_active = selected_reuse_candidate is not None
            shared_banks = (
                selected_reuse_candidate.get('shared_banks', 0)
                if selected_reuse_candidate else 0
            )
            pipeline_boundary_banks = (
                workspace.get('pipeline_boundary_banks', 0)
                if workspace_is_active else 0
            )
            jumpback_banks = (
                workspace.get('jumpback_banks', 0)
                if workspace_is_active else 0
            )
            has_streamed_model_data = bool(layer.get('weight_elems', 0)) or any(
                edge.get('is_model_state')
                for edge in layer.get('input_edges', [])
            )
            optimized_row = {
                'layer': layer.get('name', 'unknown'),
                'x_in': (
                    0 if primary_streams_here or primary_is_resident_connection
                    else _bank_kb(_bank_count(layer.get('input_elems', 0)))
                ),
                'x_out': _bank_kb(_bank_count(layer.get('output_elems', 0))),
                'total_cache_needed_kb': _bank_kb(
                    optimization['total_banks_by_layer'][layer_index]
                ),
                'weight_stream': (
                    _bank_kb(streaming_banks) if has_streamed_model_data else 0
                ),
                'pipeline_boundary': _bank_kb(pipeline_boundary_banks),
                'jumpback': _bank_kb(jumpback_banks),
                'input_output_overlap_kb': _bank_kb(shared_banks),
                'cache_rule': (
                    workspace.get('rule')
                    if workspace_is_active
                    else (
                        'stream_xin_keep_xout'
                        if workspace.get('reuses_input') and primary_streams_here
                        else workspace.get('rule')
                    )
                ),
            }
            optimized_row['x_out'] = _bank_kb(max(
                0,
                _bank_count(layer.get('output_elems', 0)) - shared_banks,
            ))
            for connection_index, connection in enumerate(connections):
                if connection_index in streamed_indices:
                    active = (
                        layer_index
                        in connection_plans[connection_index][
                            'stream_layer_indices'
                        ]
                    )
                    bank_count = streaming_banks if active else 0
                else:
                    active = _connection_is_visible(connection, layer_index)
                    bank_count = connection['bank_count'] if active else 0
                optimized_row[connection['name']] = _bank_kb(bank_count)
            optimized_rows.append(optimized_row)
        rows = optimized_rows
        columns = [
            'x_in', 'x_out', 'total_cache_needed_kb', 'weight_stream',
            'pipeline_boundary', 'jumpback'
        ] + [connection['name'] for connection in connections]

        optimization.update({
            'capacity_banks': capacity_banks,
            'bank_size_elements': int(bank_size),
            'streaming_banks': int(streaming_banks),
            'green_layers': [
                layers[index].get('name', 'unknown')
                for index in optimization['green_layer_indices']
            ],
            'red_layers': [
                layers[index].get('name', 'unknown')
                for index in optimization['red_layer_indices']
            ],
        })

    return {
        'columns': columns,
        'residual_connections': residual_connections,
        'held_connections': held_connections,
        'connections': connections,
        'unit': 'KB',
        'kilobytes_per_element': 0.001,
        'bank_optimized': optimization is not None,
        'optimization': optimization,
        'rows': rows,
    }


def print_cache_map(cache_map: dict):
    """Print the cache map in decimal kilobytes."""
    columns = cache_map['columns']
    rows = cache_map['rows']
    layer_width = max([len('Layer')] + [len(str(row['layer'])) for row in rows])
    widths = {
        column: max(
            len(column),
            max((len(str(row[column])) for row in rows), default=1),
        )
        for column in columns
    }
    header = f"{'Layer':<{layer_width}} | " + " | ".join(
        f"{column:>{widths[column]}}" for column in columns
    )
    separator = '-' * len(header)
    print(f"\nCache map ({cache_map.get('unit', 'KB')})\n{header}\n{separator}")
    for row in rows:
        values = " | ".join(
            f"{row[column]:>{widths[column]}}" for column in columns
        )
        print(f"{row['layer']:<{layer_width}} | {values}")
    print(separator)
    print_cache_map_optimization(cache_map)


def print_cache_map_optimization(cache_map: dict):
    """Print the greedy bank-placement decisions without the full matrix."""
    optimization = cache_map.get('optimization')
    if optimization:
        print(
            "Cache-map bank optimization: "
            f"{len(optimization['streamed_connection_indices'])} streamed, "
            f"{len(optimization['green_layer_indices'])} green, "
            f"{len(optimization['red_layer_indices'])} red"
        )
        for choice in optimization.get('choices', []):
            print(
                f"  stream {choice['connection_name']}: "
                f"{choice['bank_count']} banks, solved "
                f"{choice['solved_layer_count']} layer(s)"
            )


def save_cache_map_csv(cache_map: dict, path: str):
    """Write the numeric cache-map matrix to CSV."""
    fieldnames = ['layer'] + cache_map['columns']
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction='ignore',
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(cache_map['rows'])


def run_simulation(args):
    if getattr(args, 'model_config', None):
        args.model_name = args.model_config

    cache_elements = int(args.cache_size * 1_000_000)
    bank_size      = cache_elements // args.num_banks   # elements per bank

    adapter_override = {}
    if args.fold_layers is not None:
        adapter_override['fold_layers'] = args.fold_layers
    if args.fold_input_norm is not None:
        adapter_override['fold_input_norm'] = args.fold_input_norm
    if args.quantize_first_layer is not None:
        adapter_override['quantize_first_layer'] = args.quantize_first_layer

    # Resolve and load models to run
    models_to_run = []
    is_yaml = args.model_name.endswith('.yaml') or args.model_name.endswith('.yml') or os.path.isfile(args.model_name)
    if is_yaml:
        import yaml
        try:
            with open(args.model_name, 'r') as f:
                cfg = yaml.safe_load(f)
            
            if isinstance(cfg, list):
                for item in cfg:
                    if isinstance(item, dict):
                        models_to_run.append({
                            'config_path': args.model_name,
                            'name': item.get('name'),
                            'model_cfg': item,
                            'adapter_cfg': {}
                        })
                    elif isinstance(item, str):
                        models_to_run.append({
                            'config_path': args.model_name,
                            'name': item,
                            'model_cfg': {'name': item, 'weights': None},
                            'adapter_cfg': {}
                        })
            elif isinstance(cfg, dict):
                if 'model' in cfg and isinstance(cfg['model'], dict):
                    model_cfg = cfg['model']
                    name = model_cfg.get('name', args.model_name)
                    adapter_cfg = dict(cfg.get('adapter', {}), **adapter_override)
                else:
                    model_cfg = cfg
                    name = cfg.get('name', args.model_name)
                    adapter_cfg = dict({}, **adapter_override)
                
                models_to_run.append({
                    'config_path': args.model_name,
                    'name': name,
                    'model_cfg': model_cfg,
                    'adapter_cfg': adapter_cfg
                })
        except Exception as e:
            print(f"Warning: Failed to parse YAML file {args.model_name}: {e}")
            models_to_run.append({
                'config_path': None,
                'name': args.model_name,
                'model_cfg': {'name': args.model_name, 'weights': None},
                'adapter_cfg': dict({}, **adapter_override)
            })
    else:
        models_to_run.append({
            'config_path': None,
            'name': args.model_name,
            'model_cfg': {'name': args.model_name, 'weights': None},
            'adapter_cfg': dict({'type': 'generic', 'build_quantized': True}, **adapter_override)
        })

    total_models = len(models_to_run)
    cache_map_only = getattr(args, 'cache_map_only', False)
    if cache_map_only:
        print("--- ASIC Cache Map Trace ---")
        print(f"Loaded {total_models} model(s) to trace.")
        print(f"Batch Size: {args.batch_size}")
        print("----------------------------")
    else:
        print("--- ASIC Cache Simulation ---")
        print(f"Loaded {total_models} model(s) to simulate.")
        print(f"Cache Size:    {fmt_elems(cache_elements)} elements  ({args.num_banks} banks × {fmt_elems(bank_size)} elements)")
        print(f"Metadata Bits: {args.metadata_bits} per 128-bit chunk")
        print(f"Batch Size:    {args.batch_size}")
        print("-----------------------------")

    for idx, model_info in enumerate(models_to_run, 1):
        model_display_name = model_info['name']
        config_path = model_info['config_path']
        model_cfg = model_info['model_cfg']
        adapter_cfg = model_info['adapter_cfg']

        action = "Tracing" if cache_map_only else "Simulating"
        print(f"\n[{idx}/{total_models}] {action} model: {model_display_name}" + (f" (from config: {config_path})" if config_path else ""))

        try:
            layers, cache_map_layers = analyze_model(
                model_cfg, args.batch_size, args.device, adapter_cfg,
                cache_elements, bank_size, args.metadata_bits,
                return_cache_map_layers=True,
            )
        except Exception as e:
            print(f"Error: Failed to analyze model {model_display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if not layers:
            print(f"No layers found to analyze for model {model_display_name}.")
            continue

        if cache_map_only:
            cache_map = build_cache_map(
                cache_map_layers,
                cache_elements=cache_elements,
                bank_size=bank_size,
                metadata_bits=args.metadata_bits,
            )
            out_dir = os.path.dirname(os.path.abspath(__file__))
            sanitized_model_name = "".join(
                c if c.isalnum() else "_" for c in model_display_name
            )
            cache_map_path = os.path.join(
                out_dir, f"cache_map_{sanitized_model_name}.csv"
            )
            save_cache_map_csv(cache_map, cache_map_path)
            print_cache_map_optimization(cache_map)
            print(f"Cache map saved to {cache_map_path}")
            continue

        # The simulation schedule must be the same explicit runtime schedule
        # used by the cache map and architecture graph.  The older collapsed
        # list is retained as analyze_model's compatibility return value, but
        # it cannot express Q/K/V fan-out or arbitrary tensor lifetimes.
        layers = cache_map_layers
        cache_plan = _producer_consumer_cache_plan(
            layers, cache_elements, bank_size, args.metadata_bits
        )
        results = []

        for i, layer in enumerate(layers):
            cache_step = cache_plan['steps'][i]
            next_layer = layers[i + 1] if i + 1 < len(layers) else None

            weight_elems = get_footprint_elements(layer['weight_elems'], args.metadata_bits)
            input_elems  = get_footprint_elements(layer['input_elems'],  args.metadata_bits)
            output_elems = get_footprint_elements(layer['output_elems'], args.metadata_bits)

            next_xin_elems = (
                get_footprint_elements(next_layer['input_elems'], args.metadata_bits)
                if next_layer else 0
            )

            output_banked   = round_to_banks(output_elems,   bank_size)
            input_banked    = round_to_banks(input_elems,    bank_size)
            weight_banked   = round_to_banks(weight_elems,   bank_size)
            next_xin_banked = round_to_banks(next_xin_elems, bank_size)

            stay_on_chip = cache_step['stay_on_chip']
            perm_elems = cache_step['resident_after_elems']
            possible = cache_step['logical_cache_fits']
            rule_name = cache_step.get('cache_rule') or (
                'producer_consumer_resident'
                if stay_on_chip else 'producer_consumer_spill'
            )
            input_transfer_elems = cache_step['input_transfer_elems']
            residual_input_transfer_elems = cache_step[
                'residual_input_transfer_elems'
            ]
            main_input_transfer_elems = max(
                0, input_transfer_elems - residual_input_transfer_elems
            )
            model_state_transfer_elems = cache_step[
                'model_state_transfer_elems'
            ]
            need_input = main_input_transfer_elems > 0
            need_weight = int(layer.get('weight_elems', 0) or 0) > 0
            need_any_input = (
                input_transfer_elems > 0 or model_state_transfer_elems > 0
            )
            need_output = cache_step['output_spilled']
            residual_bits = int(args.min_bits)
            residual_input_stream_elems = layer.get('residual_input_stream_elems', 0)
            residual_output_elems = layer.get('residual_output_elems', 0)
            fixed_transfers = []
            forced_bits = {}
            residual_output_uses_main_stream = (
                residual_output_elems > 0
                and need_output
                and residual_output_elems == layer.get('output_elems', 0)
            )

            if residual_output_uses_main_stream:
                forced_bits['output'] = residual_bits
            if residual_input_transfer_elems > 0:
                fixed_transfers.append({
                    'name': 'residual_input',
                    'elems': residual_input_transfer_elems,
                    'bits': residual_bits,
                })
            if model_state_transfer_elems > 0:
                fixed_transfers.append({
                    'name': 'model_state_input',
                    'elems': model_state_transfer_elems,
                    'bits': 8,
                })

            in_b, w_b, out_b, cycle_count = optimize_layer_bits(
                layer, args.bandwidth, need_input, need_weight, need_output,
                min_bits=args.min_bits,
                fixed_transfers=fixed_transfers, forced_bits=forced_bits,
                input_transfer_elems=main_input_transfer_elems,
            )
            compute_cycles = _compute_layer_cycles(layer)

            input_bw_limited  = need_input  and in_b < 8
            weight_bw_limited = need_weight and w_b < 8
            output_bw_limited = need_output and out_b < 8

            results.append({
                'name':             layer['name'],
                'type':             layer['type'],
                'residual_connections': len(layer.get('residual_inputs', [])),
                'input_elems':      input_elems,
                'input_transfer_elems': get_footprint_elements(
                    input_transfer_elems, args.metadata_bits
                ),
                'model_state_transfer_elems': get_footprint_elements(
                    model_state_transfer_elems, args.metadata_bits
                ),
                'weight_elems':     weight_elems,
                'output_elems':     output_elems,
                'residual_input_elems': (
                    get_footprint_elements(
                        residual_input_transfer_elems, args.metadata_bits
                    )
                    if residual_input_transfer_elems else 0
                ),
                'residual_output_elems': (
                    get_footprint_elements(residual_output_elems, args.metadata_bits)
                    if residual_output_elems else 0
                ),
                'output_banked':    output_banked,
                'perm_elems':       perm_elems,
                'cache_required_elems': cache_step[
                    'execution_cache_required_elems'
                ],
                'cache_resident_before_elems': sum(
                    cache_plan['output_banked_elems'][producer_index]
                    for producer_index in cache_step[
                        'resident_before_producer_indices'
                    ]
                ),
                'cache_resident_after_elems': cache_step[
                    'resident_after_elems'
                ],
                'live_producer_layer_indices': cache_step[
                    'logical_live_producer_indices'
                ],
                'evicted_producer_layer_indices': cache_step[
                    'evicted_producer_indices'
                ],
                'input_transfer_producer_layer_indices': cache_step[
                    'input_transfer_producer_indices'
                ],
                'output_evicted_at_layer_index': cache_step[
                    'output_evicted_at'
                ],
                'next_xin_banked':  next_xin_banked,
                'footprint_banks':  output_banked // bank_size,
                'next_xin_banks':   next_xin_banked // bank_size,
                'next_layer_name':  next_layer['name'] if next_layer else None,
                'total_required':   cache_step['execution_cache_required_elems'],
                'filter_height':    layer.get('filter_height', 0),
                'filter_width':     layer.get('filter_width', 0),
                'in_channels':      layer.get('in_channels', 0),
                'out_channels':     layer.get('out_channels', 0),
                'input_channel_height':  layer.get('input_channel_height', 0),
                'input_channel_width':   layer.get('input_channel_width', 0),
                'output_channel_height': layer.get('output_channel_height', 0),
                'output_channel_width':  layer.get('output_channel_width', 0),
                'stay_on_chip':     stay_on_chip,
                'xin_from_cache':   not need_any_input,
                'need_input_transfer': need_any_input,
                'rule':             rule_name,
                'placement_policy': (
                    'producer_consumer_resident'
                    if stay_on_chip else 'producer_consumer_spill'
                ),
                'reason': (
                    'producer-consumer lifetime retained'
                    if stay_on_chip else
                    'producer-consumer lifetime spilled or evicted'
                ),
                'residual_producer_name': layer.get('residual_producer_name'),
                'residual_output_consumers': layer.get('residual_output_consumers', []),
                'collapsed_layers': layer.get('collapsed_layers', []),
                'input_bits':       in_b,
                'weight_bits':      w_b,
                'output_bits':      out_b,
                'residual_input_bits': (
                    residual_bits if residual_input_transfer_elems else None
                ),
                'residual_output_bits': (
                    residual_bits
                    if residual_output_elems and need_output else None
                ),
                'input_bw_limited':   input_bw_limited,
                'weight_bw_limited':  weight_bw_limited,
                'output_bw_limited':  output_bw_limited,
                'residual_input_bw_limited': residual_input_transfer_elems > 0,
                'residual_output_bw_limited': (
                    residual_output_elems > 0 and need_output
                ),
                'compute_cycles':   compute_cycles,
                'total_cycles':     cycle_count,
            })

        # --- Console output ---
        COL = 11
        BWCOL = 6
        header = (
            f"{'Layer Name':<45} | {'Type':<14}"
            f" | {'Input':>{COL}} | {'Weights':>{COL}}"
            f" | {'Output':>{COL}} | {'Banked':>{COL}}"
            f" | {'Required':>{COL}} | {'Resident':>{COL}} | {'OnChip':<7}"
            f" | {'inB':>{BWCOL}} | {'wB':>{BWCOL}} | {'outB':>{BWCOL}}"
            f" | Reason"
        )
        sep = "-" * len(header)
        print(f"\n{header}\n{sep}")

        quantize_count = flagged_count = 0
        for res in results:
            on_chip_str = "yes" if res['stay_on_chip'] else "no"
            print(
                f"{res['name']:<45} | {res['type']:<14}"
                f" | {fmt_elems(res['input_elems']):>{COL}}"
                f" | {fmt_elems(res['weight_elems']):>{COL}}"
                f" | {fmt_elems(res['output_elems']):>{COL}}"
                f" | {fmt_elems(res['output_banked']):>{COL}}"
                f" | {fmt_elems(res['cache_required_elems']):>{COL}}"
                f" | {fmt_elems(res['cache_resident_after_elems']):>{COL}}"
                f" | {on_chip_str:<7}"
                f" | {res['input_bits']:>{BWCOL}} | {res['weight_bits']:>{BWCOL}} | {res['output_bits']:>{BWCOL}}"
                f" | {res['reason']}"
            )
            if not res['stay_on_chip'] and res['rule'] != 'FLAGGED':
                quantize_count += 1
            elif res['rule'] == 'FLAGGED':
                flagged_count += 1

        print(sep)
        print(f"Total layers:              {len(results)}")
        print(f"Layers marked QUANTIZE:    {quantize_count}")
        print(f"Layers FLAGGED (no rule):  {flagged_count}")

        cache_map = build_cache_map(
            cache_map_layers,
            cache_elements=cache_elements,
            bank_size=bank_size,
            metadata_bits=args.metadata_bits,
        )
        print_cache_map(cache_map)

        # --- off_chip_layers: names only ---
        off_chip_layers = [res['name'] for res in results if not res['stay_on_chip']]

        # --- Structured JSON output ---
        output = {
            'metadata': {
                'model':          model_display_name,
                'model_config':   config_path,
                'cache_elements': cache_elements,
                'cache_size_M':   args.cache_size,
                'num_banks':      args.num_banks,
                'bank_size':      bank_size,
                'metadata_bits':  args.metadata_bits,
                'batch_size':     args.batch_size,
                'bandwidth':      args.bandwidth,
                'min_bits':       args.min_bits,
                'cache_policy':   cache_plan['policy'],
                'timestamp':      datetime.utcnow().isoformat() + 'Z',
            },
            'summary': {
                'total_layers':   len(results),
                'quantize_count': quantize_count,
                'flagged_count':  flagged_count,
            },
            'layers': results,
            'cache_map': cache_map,
            'off_chip_layers': off_chip_layers,
            'rules': serialize_producer_consumer_policy(),
        }

        out_dir  = os.path.dirname(os.path.abspath(__file__))
        
        # Save standard output file
        out_path_std = os.path.join(out_dir, "simulation_results.json")
        with open(out_path_std, 'w') as f:
            json.dump(output, f, indent=2)

        # Save model-specific output file
        sanitized_model_name = "".join([c if c.isalnum() else "_" for c in model_display_name])
        out_path_model = os.path.join(out_dir, f"simulation_results_{sanitized_model_name}.json")
        with open(out_path_model, 'w') as f:
            json.dump(output, f, indent=2)

        cache_map_path_std = os.path.join(out_dir, "cache_map.csv")
        cache_map_path_model = os.path.join(out_dir, f"cache_map_{sanitized_model_name}.csv")
        save_cache_map_csv(cache_map, cache_map_path_std)
        save_cache_map_csv(cache_map, cache_map_path_model)
        
        print(f"\nResults saved to {out_path_model} and {out_path_std}")
        print(f"Cache map saved to {cache_map_path_model} and {cache_map_path_std}")

        # Upload to DB
        try:
            from runspace.src.database.handler import RunDatabase
            RunDatabase().store_cache_simulation(output)
            print(f"[CacheSim] Successfully stored simulation for {model_display_name} to DB.")
        except Exception as e:
            print(f"[CacheSim] Warning: could not upload to DB for {model_display_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    type=str,   default="resnet18",
                        help="Model name (e.g., resnet18) or path to a model config .yaml file")
    parser.add_argument("--model_config",  type=str,   default=None,
                        help="Path to a model config .yaml file (overrides --model_name)")
    parser.add_argument("--cache_size",    type=float, default=2.0,
                        help="Cache size in millions of elements (e.g. 2.0 = 2,000,000 elements)")
    parser.add_argument("--num_banks",     type=int,   default=16)
    parser.add_argument("--metadata_bits", type=int,   default=0)
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--bandwidth",     type=float, default=1.0,
                        help="Memory bandwidth in bytes/cycle for BW-limitation analysis")
    parser.add_argument("--min_bits",      type=int,   default=3,
                        help="Minimum transfer bit width used by the bandwidth optimizer")
    parser.add_argument("--cache_map_only", action="store_true",
                        help="Only trace tensor sizes and write cache_map_<model>.csv; skip simulation JSON and database upload")
    parser.add_argument("--fold_layers", action="store_true", dest="fold_layers",
                        default=True,
                        help="Fold batchnorm/conv layers during model build")
    parser.add_argument("--no_fold_layers", action="store_false", dest="fold_layers",
                        help="Disable layer folding during model build")
    parser.add_argument("--fold_input_norm", action="store_true", dest="fold_input_norm",
                        default=True,
                        help="Fold input normalization into the first layer")
    parser.add_argument("--no_fold_input_norm", action="store_false", dest="fold_input_norm",
                        help="Disable folding of input normalization")
    parser.add_argument("--quantize_first_layer", action="store_true", dest="quantize_first_layer",
                        default=True,
                        help="Quantize the first layer's input/weights")
    parser.add_argument("--no_quantize_first_layer", action="store_false", dest="quantize_first_layer",
                        help="Disable quantization of the first layer")
    args = parser.parse_args()

    run_simulation(args)
