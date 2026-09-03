"""Backend services for the interactive model-quantization workbench.

The workbench deliberately separates inspection, planning, conversion and
validation.  In particular, inspection is operation based: an ``nn.Linear``
subclass that overrides ``forward`` is expanded by FX and is *not* silently
treated as an ordinary Linear.  The resulting graph keeps the owning module on
every operation node, which lets a GUI draw one-to-many mappings from a custom
module to the proposed quantized operations.

All analysis and recipe objects expose :meth:`to_dict` and only contain JSON
friendly values.  The converted model itself is intentionally kept out of the
serialized conversion recipe.
"""

from __future__ import annotations

import copy
import contextlib
import importlib
import inspect
import io
import json
import math
import operator
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from ..adapters.generic_adapter import GenericAdapter
from ..registry.op_registry import OpRegistry
from ..utils.fx_trace_utils import QuantAwareTracer, find_non_tensor_nodes


STATUS_EXACT = "exact_native_support"
STATUS_TRANSPARENT_SUBCLASS = "transparent_subclass"
STATUS_CUSTOM_EXPANDED = "custom_expanded"
STATUS_FUNCTIONAL = "functional_support"
STATUS_PASSTHROUGH = "structural_passthrough"
STATUS_FP32 = "fp32_fallback"
STATUS_UNSUPPORTED = "unsupported"
WORKBENCH_ANALYSIS_SCHEMA_VERSION = 2
WORKBENCH_DATASET_BENCHMARK_API_VERSION = 1
WORKBENCH_REPLACEMENT_API_VERSION = 1

_REPLACEMENT_PLAN_VERSION = 2
_REPLACEMENT_MAX_STATE_ELEMENTS = 100_000_000


DEFAULT_QUANT_OPTIONS: dict[str, Any] = {
    # These defaults make a conversion preview deterministic and safe on CPU.
    # Users can enable actual weight/activation quantization in a later plan.
    "input_quantization": False,
    "weight_quantization": False,
    "output_quantization": False,
    "fold_layers": False,
    "fold_input_norm": False,
    "skip_calibration": True,
    "enable_fx_quantization": True,
    "quantized_ops": ["all"],
}


class _DictLikeRecord:
    """Small convenience layer for Streamlit code that prefers dict access."""

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - implemented below
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()


@dataclass
class WorkbenchAnalysis(_DictLikeRecord):
    model_name: str
    source: str
    source_graph: dict[str, list[dict[str, Any]]]
    target_graph: dict[str, list[dict[str, Any]]]
    mappings: list[dict[str, Any]]
    summary: dict[str, Any]
    warnings: list[str]
    module_rows: list[dict[str, Any]]
    trace_succeeded: bool = True
    capture_kind: str = "fx"
    capture_details: dict[str, Any] = field(default_factory=dict)
    schema_version: int = WORKBENCH_ANALYSIS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "source": self.source,
            "source_graph": copy.deepcopy(self.source_graph),
            "target_graph": copy.deepcopy(self.target_graph),
            "mappings": copy.deepcopy(self.mappings),
            "summary": copy.deepcopy(self.summary),
            "warnings": list(self.warnings),
            "module_rows": copy.deepcopy(self.module_rows),
            "trace_succeeded": bool(self.trace_succeeded),
            "capture_kind": self.capture_kind,
            "capture_details": copy.deepcopy(self.capture_details),
            "schema_version": int(self.schema_version),
        }


@dataclass
class ConversionPlan(_DictLikeRecord):
    model_name: str
    source: str
    decisions: dict[str, str]
    quant_options: dict[str, Any]
    replacement_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    capture_kind: str = "fx"

    @property
    def fp32_paths(self) -> list[str]:
        return [path for path, choice in self.decisions.items() if choice == "fp32"]

    @property
    def expand_paths(self) -> list[str]:
        return [path for path, choice in self.decisions.items() if choice == "expand"]

    @property
    def alias_choices(self) -> dict[str, str]:
        return {
            path: choice.split(":", 1)[1]
            for path, choice in self.decisions.items()
            if choice.startswith("alias:")
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _REPLACEMENT_PLAN_VERSION,
            "model_name": self.model_name,
            "source": self.source,
            "decisions": dict(self.decisions),
            "quant_options": _json_safe(self.quant_options),
            "replacement_specs": copy.deepcopy(self.replacement_specs),
            "warnings": list(self.warnings),
            "capture_kind": self.capture_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConversionPlan":
        if not isinstance(value, Mapping):
            raise ModelWorkbenchError("Conversion plan must be a mapping.")
        raw_version = value.get("version", 1)
        if isinstance(raw_version, bool) or not isinstance(raw_version, int):
            raise ModelWorkbenchError("Conversion plan version must be an integer.")
        version = raw_version
        if version not in {1, _REPLACEMENT_PLAN_VERSION}:
            raise ModelWorkbenchError(
                f"Unsupported conversion plan version {version}; supported versions are 1 "
                f"and {_REPLACEMENT_PLAN_VERSION}."
            )
        raw_decisions = value.get("decisions", {})
        if not isinstance(raw_decisions, Mapping):
            raise ModelWorkbenchError("Conversion plan decisions must be a mapping.")
        decisions = {
            str(key): _normalize_choice(choice)
            for key, choice in raw_decisions.items()
        }
        replacement_specs = _normalize_replacement_specs_schema(
            value.get("replacement_specs", {}), require_confirmed=True
        )
        _validate_replacement_plan_relationships(decisions, replacement_specs)
        raw_quant_options = value.get("quant_options", {})
        if not isinstance(raw_quant_options, Mapping):
            raise ModelWorkbenchError("Conversion plan quant_options must be a mapping.")
        raw_warnings = value.get("warnings", [])
        if isinstance(raw_warnings, (str, bytes)) or not isinstance(raw_warnings, Sequence):
            raise ModelWorkbenchError("Conversion plan warnings must be a list.")
        return cls(
            model_name=str(value.get("model_name", "model")),
            source=str(value.get("source", "custom")),
            decisions=decisions,
            quant_options=dict(raw_quant_options),
            replacement_specs=replacement_specs,
            warnings=[str(warning) for warning in raw_warnings],
            capture_kind=str(value.get("capture_kind", "fx")),
        )


@dataclass
class ConversionResult(_DictLikeRecord):
    model: nn.Module
    adapter: GenericAdapter | None
    warnings: list[str]
    recipe: dict[str, Any]
    realization: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": _qualified_type_name(type(self.model)),
            "adapter_type": (
                _qualified_type_name(type(self.adapter)) if self.adapter is not None else None
            ),
            "warnings": list(self.warnings),
            "recipe": copy.deepcopy(self.recipe),
            "realization": copy.deepcopy(self.realization),
        }


class ModelWorkbenchError(RuntimeError):
    """Clear, user-facing error raised by model-provider/workbench operations."""


def list_replacement_targets() -> list[dict[str, Any]]:
    """Return the safe, registry-backed catalog for explicit replacements.

    Catalog IDs are matched against this in-memory registry only.  They are
    deliberately not import paths and are never evaluated or imported from a
    user-provided string.
    """

    return [
        {
            key: copy.deepcopy(value)
            for key, value in entry.items()
            if not key.startswith("_")
        }
        for entry in _replacement_registry_entries()
    ]


def inspect_replacement_target(
    model: nn.Module,
    source_path: str,
    target_id: str,
    *,
    constructor_args: Sequence[Any] | None = None,
    constructor_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect a source module and one safely registered replacement target."""

    source = _replacement_source_module(model, source_path)
    raw_args: Sequence[Any] = [] if constructor_args is None else constructor_args
    if isinstance(raw_args, (str, bytes)) or not isinstance(raw_args, Sequence):
        raise ModelWorkbenchError("constructor_args must be a JSON array.")
    raw_kwargs: Mapping[str, Any] = {} if constructor_kwargs is None else constructor_kwargs
    if not isinstance(raw_kwargs, Mapping):
        raise ModelWorkbenchError("constructor_kwargs must be a mapping.")
    args = _json_only(list(raw_args), "constructor_args")
    kwargs = _json_only(dict(raw_kwargs), "constructor_kwargs")
    entry = _resolve_replacement_catalog_entry(target_id)
    target = _instantiate_replacement_native(entry, args, kwargs, initializer_seed=0)
    source_fields = _module_state_fields(source, source_path, role="Source")
    target_fields = _module_state_fields(target, source_path, role="Replacement target")
    source_by_key = {field["local_key"]: field for field in source_fields}

    suggestions: dict[str, str] = {}
    suggestion_rows: list[dict[str, Any]] = []
    compatible: dict[str, list[str]] = {}
    for target_field in target_fields:
        target_key = target_field["local_key"]
        shape_matches = [
            source_field["local_key"]
            for source_field in source_fields
            if source_field["shape"] == target_field["shape"]
        ]
        compatible[target_key] = shape_matches
        source_field = source_by_key.get(target_key)
        if (
            source_field is not None
            and source_field["shape"] == target_field["shape"]
            and source_field["dtype"] == target_field["dtype"]
        ):
            suggestions[target_key] = target_key
            suggestion_rows.append(
                {
                    "target_key": target_key,
                    "source_key": target_key,
                    "match": "exact_name_shape_and_dtype",
                    "shape": list(target_field["shape"]),
                    "dtype": target_field["dtype"],
                }
            )

    suggested_sources = set(suggestions.values())
    initializers = {
        field["local_key"]: "target_default"
        for field in target_fields
        if field["local_key"] not in suggestions
    }
    template = {
        "target_id": entry["id"],
        "constructor_args": copy.deepcopy(args),
        "constructor_kwargs": copy.deepcopy(kwargs),
        "state_mapping": suggestions,
        "state_initializers": initializers,
        "initializer_seed": 0,
        "confirmed": False,
    }
    return {
        "api_version": WORKBENCH_REPLACEMENT_API_VERSION,
        "source": {
            "path": str(source_path),
            "name": type(source).__name__,
            "type": _qualified_type_name(type(source)),
            "state_fields": source_fields,
        },
        "target": {
            "id": entry["id"],
            "target_name": entry["target_name"],
            "target_type": entry["target_type"],
            "native_name": entry["native_name"],
            "native_type": entry["native_type"],
            "state_fields": target_fields,
            "state_elements": sum(field["numel"] for field in target_fields),
        },
        "suggested_state_mapping": copy.deepcopy(suggestions),
        "exact_name_shape_suggestions": suggestion_rows,
        "shape_compatible_source_keys": compatible,
        "unused_source_fields": [
            copy.deepcopy(field)
            for field in source_fields
            if field["local_key"] not in suggested_sources
        ],
        "spec_template": template,
        "warnings": [
            "The exported converted state_dict is canonical. target_default values are "
            "created with initializer_seed=0 for deterministic recipe replay.",
            *[
                f"Mapping {target_field['local_key']!r} from {source_field['local_key']!r} "
                f"would cast {source_field['dtype']} to {target_field['dtype']}."
                for target_field in target_fields
                for source_field in source_fields
                if source_field["shape"] == target_field["shape"]
                and source_field["dtype"] != target_field["dtype"]
            ],
        ],
    }


def validate_replacement_spec(
    model: nn.Module,
    source_path: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one replacement spec without mutating ``model``."""

    normalized, _entry, _target = _validate_and_materialize_replacement(
        model, source_path, spec, load_state=False
    )
    return normalized


def _replacement_registry_entries() -> list[dict[str, Any]]:
    # Importing ops populates OpRegistry in minimal dashboard/test processes.
    try:
        from .. import ops as _ops  # noqa: F401
    except ImportError:
        import runspace.src.ops as _ops  # type: ignore[no-redef]  # noqa: F401

    entries: dict[str, dict[str, Any]] = {}
    noop_types = tuple(GenericAdapter._noop_layer_types())
    for native_cls, quantized_cls in OpRegistry.get_supported_ops().items():
        if not (
            isinstance(native_cls, type)
            and issubclass(native_cls, nn.Module)
            and isinstance(quantized_cls, type)
            and issubclass(quantized_cls, nn.Module)
        ):
            continue
        # GenericAdapter deliberately removes Dropout/Identity modules, so
        # promising those as realized replacements would be misleading.
        if any(issubclass(native_cls, noop_type) for noop_type in noop_types):
            continue
        target_name = quantized_cls.__name__
        if not OpRegistry.is_quantized(target_name) or OpRegistry.is_under_construction(
            target_name
        ):
            continue
        native_type = _qualified_type_name(native_cls)
        catalog_id = f"{native_type}::{target_name}"
        entry = {
            "id": catalog_id,
            "target_name": target_name,
            "target_type": _qualified_type_name(quantized_cls),
            "native_name": native_cls.__name__,
            "native_type": native_type,
            "constructor_parameters": _constructor_parameter_metadata(native_cls),
            "_native_cls": native_cls,
            "_quantized_cls": quantized_cls,
        }
        previous = entries.get(catalog_id)
        if previous is not None and previous["_quantized_cls"] is not quantized_cls:
            raise ModelWorkbenchError(
                f"Ambiguous replacement catalog ID {catalog_id!r}; registry entries conflict."
            )
        entries[catalog_id] = entry
    return sorted(
        entries.values(), key=lambda item: (item["target_name"].casefold(), item["native_type"])
    )


def _constructor_parameter_metadata(native_cls: type[nn.Module]) -> list[dict[str, Any]]:
    try:
        signature = inspect.signature(native_cls)
    except (TypeError, ValueError):
        return []
    result = []
    for parameter in signature.parameters.values():
        default = parameter.default
        result.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name.lower(),
                "required": default is inspect.Parameter.empty,
                "default": (
                    None
                    if default is inspect.Parameter.empty
                    else _json_safe(default)
                ),
                "annotation": (
                    None
                    if parameter.annotation is inspect.Parameter.empty
                    else _json_safe(parameter.annotation)
                ),
            }
        )
    return result


def _resolve_replacement_catalog_entry(target_id: Any) -> dict[str, Any]:
    if not isinstance(target_id, str) or not target_id.strip():
        raise ModelWorkbenchError("replacement target_id must be a non-empty catalog ID.")
    matches = [entry for entry in _replacement_registry_entries() if entry["id"] == target_id]
    if not matches:
        raise ModelWorkbenchError(
            f"Unknown replacement target_id {target_id!r}. Select an ID returned by "
            "list_replacement_targets(); arbitrary import paths are not allowed."
        )
    return matches[0]


def _json_only(value: Any, field_path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelWorkbenchError(f"{field_path} must not contain NaN or infinity.")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _json_only(child, f"{field_path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ModelWorkbenchError(f"{field_path} keys must be strings.")
            normalized[key] = _json_only(child, f"{field_path}.{key}")
        return normalized
    raise ModelWorkbenchError(
        f"{field_path} must contain JSON values only, got {type(value).__name__}."
    )


def _normalize_state_initializer(value: Any, field_path: str) -> str | dict[str, Any]:
    if isinstance(value, str):
        kind = value.strip().lower()
        if kind not in {"target_default", "zeros", "ones"}:
            raise ModelWorkbenchError(
                f"{field_path} must be target_default, zeros, ones, or a constant initializer."
            )
        return kind
    if not isinstance(value, Mapping):
        raise ModelWorkbenchError(f"{field_path} must describe an explicit initializer.")
    unknown = set(value) - {"kind", "value"}
    if unknown:
        raise ModelWorkbenchError(
            f"{field_path} contains unknown keys: {', '.join(sorted(str(key) for key in unknown))}."
        )
    if str(value.get("kind", "")).strip().lower() != "constant":
        raise ModelWorkbenchError(f"{field_path}.kind must be 'constant'.")
    constant = value.get("value")
    if isinstance(constant, bool) or not isinstance(constant, (int, float)):
        raise ModelWorkbenchError(f"{field_path}.value must be a finite number.")
    if isinstance(constant, float) and not math.isfinite(constant):
        raise ModelWorkbenchError(f"{field_path}.value must be a finite number.")
    return {"kind": "constant", "value": constant}


def _normalize_replacement_spec_schema(
    spec: Mapping[str, Any], *, require_confirmed: bool
) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ModelWorkbenchError("Each replacement spec must be a mapping.")
    allowed = {
        "target_id",
        "constructor_args",
        "constructor_kwargs",
        "state_mapping",
        "state_initializers",
        "initializer_seed",
        "confirmed",
    }
    unknown = set(spec) - allowed
    if unknown:
        raise ModelWorkbenchError(
            "Replacement spec contains unknown keys: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    target_id = spec.get("target_id")
    entry = _resolve_replacement_catalog_entry(target_id)
    raw_args = spec.get("constructor_args", [])
    if isinstance(raw_args, (str, bytes)) or not isinstance(raw_args, Sequence):
        raise ModelWorkbenchError("replacement constructor_args must be a JSON array.")
    args = _json_only(list(raw_args), "replacement.constructor_args")
    raw_kwargs = spec.get("constructor_kwargs", {})
    if not isinstance(raw_kwargs, Mapping):
        raise ModelWorkbenchError("replacement constructor_kwargs must be a mapping.")
    kwargs = _json_only(dict(raw_kwargs), "replacement.constructor_kwargs")

    raw_mapping = spec.get("state_mapping", {})
    if not isinstance(raw_mapping, Mapping):
        raise ModelWorkbenchError("replacement state_mapping must be a mapping.")
    state_mapping: dict[str, str] = {}
    for target_key, source_key in raw_mapping.items():
        if not isinstance(target_key, str) or not target_key:
            raise ModelWorkbenchError("replacement state_mapping target keys must be non-empty strings.")
        if not isinstance(source_key, str) or not source_key:
            raise ModelWorkbenchError("replacement state_mapping source keys must be non-empty strings.")
        state_mapping[target_key] = source_key

    raw_initializers = spec.get("state_initializers", {})
    if not isinstance(raw_initializers, Mapping):
        raise ModelWorkbenchError("replacement state_initializers must be a mapping.")
    state_initializers: dict[str, Any] = {}
    for target_key, initializer in raw_initializers.items():
        if not isinstance(target_key, str) or not target_key:
            raise ModelWorkbenchError(
                "replacement state_initializers target keys must be non-empty strings."
            )
        state_initializers[target_key] = _normalize_state_initializer(
            initializer, f"replacement.state_initializers.{target_key}"
        )
    overlap = set(state_mapping) & set(state_initializers)
    if overlap:
        raise ModelWorkbenchError(
            "Replacement target fields cannot be both mapped and initialized: "
            + ", ".join(sorted(overlap))
        )

    seed = spec.get("initializer_seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**63:
        raise ModelWorkbenchError("replacement initializer_seed must be an integer in [0, 2^63).")
    confirmed = spec.get("confirmed", False)
    if not isinstance(confirmed, bool):
        raise ModelWorkbenchError("replacement confirmed must be a JSON boolean.")
    if require_confirmed and confirmed is not True:
        raise ModelWorkbenchError("Replacement spec requires confirmed=true before it can be used.")
    return {
        "target_id": entry["id"],
        "constructor_args": args,
        "constructor_kwargs": kwargs,
        "state_mapping": state_mapping,
        "state_initializers": state_initializers,
        "initializer_seed": seed,
        "confirmed": confirmed,
    }


def _normalize_replacement_specs_schema(
    specs: Any, *, require_confirmed: bool
) -> dict[str, dict[str, Any]]:
    if specs is None:
        return {}
    if not isinstance(specs, Mapping):
        raise ModelWorkbenchError("replacement_specs must be keyed by source module path.")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_path, spec in specs.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ModelWorkbenchError("replacement_specs keys must be non-empty module paths.")
        path = raw_path.strip()
        normalized[path] = _normalize_replacement_spec_schema(
            spec, require_confirmed=require_confirmed
        )
    return normalized


def _validate_replacement_plan_relationships(
    decisions: Mapping[str, str], replacement_specs: Mapping[str, Mapping[str, Any]]
) -> None:
    replace_paths = {
        str(path) for path, choice in decisions.items() if _normalize_choice(choice) == "replace"
    }
    spec_paths = set(replacement_specs)
    missing_specs = sorted(replace_paths - spec_paths)
    unused_specs = sorted(spec_paths - replace_paths)
    if missing_specs:
        raise ModelWorkbenchError(
            "Decision 'replace' requires a replacement spec for: " + ", ".join(missing_specs)
        )
    if unused_specs:
        raise ModelWorkbenchError(
            "Replacement specs require decision 'replace' for: " + ", ".join(unused_specs)
        )
    ordered = sorted(spec_paths, key=lambda path: (path.count("."), path))
    for index, ancestor in enumerate(ordered):
        for descendant in ordered[index + 1 :]:
            if descendant.startswith(ancestor + "."):
                raise ModelWorkbenchError(
                    "Replacement paths cannot overlap: "
                    f"{ancestor!r} is an ancestor of {descendant!r}."
                )
    for path in ordered:
        parts = path.split(".")
        for depth in range(1, len(parts)):
            ancestor = ".".join(parts[:depth])
            ancestor_choice = _normalize_choice(decisions.get(ancestor, ""))
            if (
                ancestor_choice in {"fp32", "expand"}
                or ancestor_choice.startswith("alias:")
            ):
                raise ModelWorkbenchError(
                    f"Replacement {path!r} is hidden by ancestor decision "
                    f"{ancestor!r}={ancestor_choice!r}; resolve the ancestor first."
                )


def _replacement_source_module(model: nn.Module, source_path: str) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise ModelWorkbenchError("replacement inspection requires a torch.nn.Module model.")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ModelWorkbenchError("source_path must be a non-empty module path; root replacement is unsupported.")
    path = source_path.strip()
    try:
        return model.get_submodule(path)
    except Exception as exc:
        raise ModelWorkbenchError(f"Source module path {path!r} does not exist.") from exc


def _constructor_size_hint(args: Sequence[Any], kwargs: Mapping[str, Any]) -> int:
    values: list[int] = []

    def collect(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int) and value > 0:
            values.append(value)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        elif isinstance(value, Mapping):
            for child in value.values():
                collect(child)

    collect(args)
    collect(kwargs)
    estimate = 1
    for value in values:
        estimate *= value
        if estimate > _REPLACEMENT_MAX_STATE_ELEMENTS:
            return estimate
    return estimate


def _instantiate_replacement_native(
    entry: Mapping[str, Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    initializer_seed: int,
) -> nn.Module:
    if _constructor_size_hint(args, kwargs) > _REPLACEMENT_MAX_STATE_ELEMENTS:
        raise ModelWorkbenchError(
            "Replacement constructor dimensions exceed the workbench safety limit of "
            f"{_REPLACEMENT_MAX_STATE_ELEMENTS:,} state elements."
        )
    device = kwargs.get("device")
    if device is not None and str(device) not in {"cpu", "cpu:0"}:
        raise ModelWorkbenchError("Replacement constructors may only target CPU device state.")
    native_cls = entry["_native_cls"]
    try:
        signature = inspect.signature(native_cls)
        signature.bind(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ModelWorkbenchError(
            f"Invalid constructor arguments for {entry['native_name']}: {exc}"
        ) from exc
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initializer_seed))
            target = native_cls(*copy.deepcopy(list(args)), **copy.deepcopy(dict(kwargs)))
    except Exception as exc:
        raise ModelWorkbenchError(
            f"Could not instantiate registered native target {entry['native_name']}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    target.cpu()
    target_state = _tensor_state_dict(target, role="Replacement target")
    state_elements = sum(int(value.numel()) for value in target_state.values())
    if state_elements > _REPLACEMENT_MAX_STATE_ELEMENTS:
        raise ModelWorkbenchError(
            f"Replacement target has {state_elements:,} state elements, exceeding the "
            f"safety limit of {_REPLACEMENT_MAX_STATE_ELEMENTS:,}."
        )
    return target


def _tensor_state_dict(module: nn.Module, *, role: str) -> Mapping[str, torch.Tensor]:
    state = module.state_dict()
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise ModelWorkbenchError(
                f"{role} state field {key!r} is {type(value).__name__}, not a tensor; "
                "explicit replacement supports tensor parameters and buffers only."
            )
    return state


def _module_state_fields(
    module: nn.Module, module_path: str, *, role: str = "Module"
) -> list[dict[str, Any]]:
    try:
        parameters = dict(module.named_parameters(recurse=True, remove_duplicate=False))
        buffers = dict(module.named_buffers(recurse=True, remove_duplicate=False))
    except TypeError:  # older torch releases
        parameters = dict(module.named_parameters(recurse=True))
        buffers = dict(module.named_buffers(recurse=True))
    fields = []
    for local_key, tensor in _tensor_state_dict(module, role=role).items():
        kind = "parameter" if local_key in parameters else "buffer"
        owner = parameters.get(local_key, buffers.get(local_key))
        qualified_key = f"{module_path}.{local_key}" if module_path else local_key
        fields.append(
            {
                "key": local_key,
                "local_key": local_key,
                "qualified_key": qualified_key,
                "kind": kind,
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": int(tensor.numel()),
                "requires_grad": bool(getattr(owner, "requires_grad", False)),
            }
        )
    return fields


def _validate_replacement_state_coverage(
    source: nn.Module,
    target: nn.Module,
    normalized: Mapping[str, Any],
) -> None:
    source_state = _tensor_state_dict(source, role="Source module")
    target_state = _tensor_state_dict(target, role="Replacement target")
    target_keys = set(target_state)
    mapped_keys = set(normalized["state_mapping"])
    initialized_keys = set(normalized["state_initializers"])
    provided_keys = mapped_keys | initialized_keys
    missing = sorted(target_keys - provided_keys)
    extra = sorted(provided_keys - target_keys)
    if missing:
        raise ModelWorkbenchError(
            "Replacement spec must explicitly map or initialize every target state field; "
            "missing: " + ", ".join(missing)
        )
    if extra:
        raise ModelWorkbenchError(
            "Replacement spec references unknown target state fields: " + ", ".join(extra)
        )
    for target_key, source_key in normalized["state_mapping"].items():
        if source_key not in source_state:
            raise ModelWorkbenchError(
                f"Replacement state_mapping references unknown source field {source_key!r}."
            )
        source_shape = tuple(source_state[source_key].shape)
        target_shape = tuple(target_state[target_key].shape)
        if source_shape != target_shape:
            raise ModelWorkbenchError(
                f"Replacement state shape mismatch for {target_key!r} <- {source_key!r}: "
                f"target {list(target_shape)} versus source {list(source_shape)}."
            )


def _replacement_state_dict(
    source: nn.Module,
    target: nn.Module,
    normalized: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    source_state = _tensor_state_dict(source, role="Source module")
    target_state = _tensor_state_dict(target, role="Replacement target")
    result: dict[str, torch.Tensor] = {}
    for target_key, target_value in target_state.items():
        if target_key in normalized["state_mapping"]:
            source_value = source_state[normalized["state_mapping"][target_key]]
            result[target_key] = source_value.detach().to(
                device=target_value.device, dtype=target_value.dtype
            ).clone()
            continue
        initializer = normalized["state_initializers"][target_key]
        if initializer == "target_default":
            result[target_key] = target_value.detach().clone()
        elif initializer == "zeros":
            result[target_key] = torch.zeros_like(target_value)
        elif initializer == "ones":
            result[target_key] = torch.ones_like(target_value)
        else:
            result[target_key] = torch.full_like(target_value, initializer["value"])
    return result


def _validate_and_materialize_replacement(
    model: nn.Module,
    source_path: str,
    spec: Mapping[str, Any],
    *,
    load_state: bool,
) -> tuple[dict[str, Any], dict[str, Any], nn.Module]:
    source = _replacement_source_module(model, source_path)
    normalized = _normalize_replacement_spec_schema(spec, require_confirmed=True)
    entry = _resolve_replacement_catalog_entry(normalized["target_id"])
    target = _instantiate_replacement_native(
        entry,
        normalized["constructor_args"],
        normalized["constructor_kwargs"],
        initializer_seed=normalized["initializer_seed"],
    )
    _validate_replacement_state_coverage(source, target, normalized)
    if load_state:
        state = _replacement_state_dict(source, target, normalized)
        try:
            target.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ModelWorkbenchError(
                f"Could not load explicit replacement state for {source_path!r}: {exc}"
            ) from exc
        target.train(source.training)
    return normalized, entry, target


class _OwnershipTracer(QuantAwareTracer):
    """FX tracer that records the module whose ``forward`` owns each op."""

    def __init__(self, leaf_paths: Iterable[str] = ()):
        super().__init__()
        self._owner_stack: list[tuple[str, type[nn.Module]]] = []
        self._leaf_paths = {path for path in leaf_paths if path}

    def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
        if module_qualified_name in self._leaf_paths:
            return True
        return super().is_leaf_module(module, module_qualified_name)

    def call_module(self, module, forward, args, kwargs):
        try:
            path = self.path_of_module(module)
        except Exception:
            path = ""
        self._owner_stack.append((path, type(module)))
        try:
            return super().call_module(module, forward, args, kwargs)
        finally:
            self._owner_stack.pop()

    def create_node(self, *args, **kwargs):
        node = super().create_node(*args, **kwargs)
        if self._owner_stack:
            path, module_type = self._owner_stack[-1]
            node.meta["qbench_owner_path"] = path
            node.meta["qbench_owner_type"] = _qualified_type_name(module_type)
        return node


class _WorkbenchFP32Island(nn.Module):
    """Opaque FX leaf that keeps a selected custom subtree entirely in FP32."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def named_children(self):
        # GenericAdapter's recursive module and submodule-FX passes both walk
        # named_children().  Hiding the registered inner module from those two
        # conversion traversals makes the FP32 decision a real subtree
        # boundary.  The module remains registered in _modules, so device
        # moves, state_dict, train/eval and ordinary modules() traversal retain
        # normal PyTorch semantics.
        return iter(())

    def train(self, mode: bool = True):
        self.training = bool(mode)
        self.inner.train(mode)
        return self

    def _apply(self, fn, recurse: bool = True):
        # nn.Module._apply normally reaches children through children(), which
        # is intentionally opaque above.  Forward device/dtype transforms to
        # the registered inner subtree explicitly.
        try:
            self.inner._apply(fn, recurse=recurse)
        except TypeError:  # compatibility with older torch releases
            self.inner._apply(fn)
        return self


# torch.fx's default leaf policy intentionally treats torch.nn-owned modules as
# opaque.  Registering this small wrapper on the real torch.nn.modules package
# also keeps full-model pickling functional (state_dict export does not require
# this, but it is a useful property for callers outside the dashboard).
_WorkbenchFP32Island.__module__ = nn.modules.__name__
setattr(nn.modules, "_WorkbenchFP32Island", _WorkbenchFP32Island)


def list_model_names(source: str) -> list[str]:
    """Return model names for a provider without constructing any model.

    ``timm`` is optional; an absent installation simply yields an empty list so
    a dashboard can hide/disable that provider without an exception dialog.
    """

    normalized = _normalize_source(source)
    if normalized == "torchvision":
        try:
            from torchvision import models

            if hasattr(models, "list_models"):
                return sorted(set(models.list_models()))
            names = []
            for name in dir(models):
                value = getattr(models, name)
                if name.startswith("_") or not callable(value):
                    continue
                if name.lower().endswith(("_weights", "weights")):
                    continue
                names.append(name)
            return sorted(set(names))
        except Exception as exc:
            raise ModelWorkbenchError(f"Unable to inspect torchvision models: {exc}") from exc

    if normalized == "timm":
        try:
            import timm
        except ImportError:
            return []
        return sorted(set(timm.list_models()))

    if normalized == "custom":
        return []

    if normalized == "all":
        return sorted(set(list_model_names("torchvision") + list_model_names("timm")))

    raise ValueError("source must be 'torchvision', 'timm', 'custom', or 'all'")


def load_model(
    source: str,
    model_name: str,
    pretrained: bool = False,
    custom_factory: str | Callable[..., nn.Module] | None = None,
) -> nn.Module:
    """Load a model from torchvision, timm, or an importable custom factory."""

    normalized = _normalize_source(source)
    if normalized == "torchvision":
        try:
            from torchvision import models

            if hasattr(models, "get_model"):
                weights = "DEFAULT" if pretrained else None
                return models.get_model(model_name, weights=weights)
            if not hasattr(models, model_name):
                raise ValueError(f"unknown torchvision model {model_name!r}")
            factory = getattr(models, model_name)
            try:
                return factory(weights="DEFAULT" if pretrained else None)
            except TypeError:
                return factory(pretrained=bool(pretrained))
        except Exception as exc:
            raise ModelWorkbenchError(
                f"Could not load torchvision model {model_name!r}: {exc}"
            ) from exc

    if normalized == "timm":
        try:
            import timm
        except ImportError as exc:
            raise ModelWorkbenchError(
                "The optional 'timm' package is not installed. Install timm to use this provider."
            ) from exc
        try:
            return timm.create_model(model_name, pretrained=bool(pretrained))
        except Exception as exc:
            raise ModelWorkbenchError(f"Could not load timm model {model_name!r}: {exc}") from exc

    if normalized == "custom":
        if custom_factory is None:
            raise ModelWorkbenchError(
                "A custom model requires custom_factory='package.module:callable' (or a callable)."
            )
        factory = _resolve_custom_factory(custom_factory)
        model = _call_custom_factory(factory, model_name, bool(pretrained))
        if not isinstance(model, nn.Module):
            raise ModelWorkbenchError(
                f"Custom factory returned {type(model).__name__}; expected torch.nn.Module."
            )
        return model

    raise ValueError("source must be 'torchvision', 'timm', or 'custom'")


def analyze_model(
    model: nn.Module,
    model_name: str = "model",
    source: str = "custom",
    sample_input: Any = None,
) -> WorkbenchAnalysis:
    """Inspect modules and executed operations and build a conversion preview.

    Quantization-aware FX is attempted first.  When ``sample_input`` is
    supplied, the symbolic graph must execute for that exact shape; otherwise
    an input-specialized ``torch.export`` graph is used for analysis while the
    original eager model remains the conversion source.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")

    # Importing ops is what populates OpRegistry in minimal/test processes.
    try:
        from .. import ops as _ops  # noqa: F401
    except ImportError:
        import runspace.src.ops as _ops  # type: ignore[no-redef]  # noqa: F401

    warnings: list[str] = []
    named_modules = dict(model.named_modules())
    supported_ops = OpRegistry.get_supported_ops()

    graph_module: GraphModule | None = None
    trace_error: Exception | None = None
    shape_error: Exception | None = None
    export_error: Exception | None = None
    capture_kind = "module_hierarchy"
    capture_details: dict[str, Any] = {
        "input_specialized": False,
        "full_fx_succeeded": False,
    }
    if torch.is_tensor(sample_input):
        capture_details["sample_input_shape"] = [int(dim) for dim in sample_input.shape]
    try:
        tracer = _OwnershipTracer()
        graph = tracer.trace(model)
        graph_module = GraphModule(model, graph)
    except Exception as exc:
        trace_error = exc
    else:
        capture_kind = "fx"
        capture_details["full_fx_succeeded"] = True

    if graph_module is not None and sample_input is not None:
        try:
            _propagate_shapes(graph_module, sample_input)
        except Exception as exc:
            # A symbolic trace can select the wrong side of an input-shape
            # guard (MobileViT is a common example).  Such a graph is useful
            # neither for the UI nor for conversion planning, so prefer an
            # input-specialized export instead of merely dropping shapes.
            shape_error = exc
            graph_module = None
            capture_kind = "module_hierarchy"
            capture_details["full_fx_succeeded"] = False

    if graph_module is None and sample_input is not None:
        try:
            graph_module, export_mode = _export_model_graph(model, sample_input)
        except Exception as exc:
            export_error = exc
        else:
            capture_kind = "torch_export"
            capture_details.update(
                input_specialized=True,
                exported_node_count=len(list(graph_module.graph.nodes)),
                export_mode=export_mode,
            )
            if shape_error is not None:
                warnings.append(
                    "Symbolic FX selected a branch that was invalid for the sample input; "
                    "the operation graph was captured with torch.export for this exact input shape. "
                    f"FX validation error: {_brief_exception(shape_error)}"
                )
            elif trace_error is not None:
                warnings.append(
                    "Full-model symbolic FX was unavailable; the operation graph was captured "
                    "with torch.export for this sample input. "
                    f"FX error: {_brief_exception(trace_error)}"
                )

    if graph_module is None:
        if sample_input is not None:
            try:
                _validate_eager_sample(model, sample_input)
            except Exception as exc:
                shape_hint = capture_details.get("sample_input_shape")
                raise ModelWorkbenchError(
                    "The sample input cannot execute the reference model"
                    + (f" at shape {tuple(shape_hint)}" if shape_hint else "")
                    + f": {_brief_exception(exc)}. Choose the model's preferred input size "
                    "or provide a valid custom shape before analyzing support."
                ) from exc
        errors = []
        if trace_error is not None:
            errors.append(f"FX: {_brief_exception(trace_error)}")
        if shape_error is not None:
            errors.append(f"FX sample validation: {_brief_exception(shape_error)}")
        if export_error is not None:
            errors.append(f"torch.export: {_brief_exception(export_error)}")
        detail = "; ".join(errors) or "no operation-capture backend was available"
        warnings.append(
            "Operation capture was unavailable; showing the complete module hierarchy only. "
            + detail
        )

    module_info: dict[str, dict[str, Any]] = {}
    for path, module in named_modules.items():
        module_info[path] = _initial_module_info(path, module, supported_ops)

    op_records: list[tuple[Node, dict[str, Any]]] = []
    owned_op_records: dict[str, list[dict[str, Any]]] = {}
    if graph_module is not None:
        non_tensor_nodes = find_non_tensor_nodes(graph_module.graph)
        for node in graph_module.graph.nodes:
            record = _operation_record(
                node,
                graph_module,
                module_info,
                non_tensor_nodes=non_tensor_nodes,
                capture_kind=capture_kind,
            )
            op_records.append((node, record))
            owner = record.get("module_path", "")
            if node.op in ("call_function", "call_method"):
                owned_op_records.setdefault(owner, []).append(record)

        # Refine custom module classifications from the actual operations in
        # their forward.  This is the safeguard against treating MyLinear as a
        # plain Linear when it also performs relu/multiply/etc.
        for path, info in module_info.items():
            if capture_kind != "fx":
                break
            if not path or not info.get("custom_type"):
                continue
            if info.get("quant_target") in {
                "QuantBatchNormAct2d",
                "DecomposedQkvAttention",
                "DecomposedMlpBlock",
            }:
                continue
            # An inherited forward is intentionally classified from method
            # identity, not from the equivalent functional nodes FX exposes.
            if info.get("status") == STATUS_TRANSPARENT_SUBCLASS:
                continue
            direct_ops = owned_op_records.get(path, [])
            if not direct_ops:
                continue
            unsupported = [r for r in direct_ops if r["status"] in (STATUS_UNSUPPORTED, STATUS_FP32)]
            supported = [r for r in direct_ops if r["status"] == STATUS_FUNCTIONAL]
            if unsupported:
                info.update(
                    status=STATUS_FP32,
                    reason=(
                        f"Custom forward contains {len(unsupported)} unsupported operation(s); "
                        "keep the unresolved portion in FP32 or register a converter."
                    ),
                    recommended="fp32",
                )
                if info.get("native_base"):
                    info["candidates"] = [
                        "expand",
                        f"alias:{info['native_base']}",
                        "fp32",
                    ]
                else:
                    info["candidates"] = ["expand", "fp32"]
            elif supported:
                info.update(
                    status=STATUS_CUSTOM_EXPANDED,
                    reason=(
                        f"Custom forward expands to {len(direct_ops)} operation(s); "
                        "preserve and convert them independently."
                    ),
                    recommended="expand",
                )
                candidates = ["expand"]
                if info.get("native_base"):
                    # This is intentionally explicit: selecting it opts in to
                    # discarding the extra forward behavior.
                    candidates.append(f"alias:{info['native_base']}")
                candidates.append("fp32")
                info["candidates"] = candidates

    source_nodes: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for path, module in named_modules.items():
        info = module_info[path]
        module_id = _module_node_id(path)
        parent_path = path.rpartition(".")[0] if path else ""
        parent_id = _module_node_id(parent_path) if path else None
        node = {
            "id": module_id,
            "label": model_name if not path else path.rsplit(".", 1)[-1],
            "op": "module_group",
            "target": _qualified_type_name(type(module)),
            "module_path": path,
            "module_type": type(module).__name__,
            "status": info["status"],
            "reason": info["reason"],
            "parent": parent_id,
            "candidates": list(info["candidates"]),
            "recommended": info["recommended"],
            "kind": "module_group",
            "type": "module_group",
            "is_group": True,
        }
        source_nodes.append(node)
        if path:
            module_rows.append(
                {
                    "node_id": module_id,
                    "path": path,
                    "type": type(module).__name__,
                    "status": info["status"],
                    "reason": info["reason"],
                    "candidates": list(info["candidates"]),
                    "recommended": info["recommended"],
                }
            )

    source_edges: list[dict[str, Any]] = []
    fx_id_by_node: dict[Node, str] = {}
    if graph_module is not None:
        for node, record in op_records:
            op_id = f"op:{node.name}"
            fx_id_by_node[node] = op_id
            record["id"] = op_id
            record["parent"] = _module_node_id(record.get("module_path", ""))
            source_nodes.append(record)

        seen_edges: set[tuple[str, str]] = set()
        for node, _record in op_records:
            target_id = fx_id_by_node[node]
            for input_node in _iter_fx_nodes((node.args, node.kwargs)):
                source_id = fx_id_by_node.get(input_node)
                if source_id is None or (source_id, target_id) in seen_edges:
                    continue
                seen_edges.add((source_id, target_id))
                source_edges.append(
                    {"source": source_id, "target": target_id, "kind": "dataflow"}
                )
    else:
        # A module-only graph is still useful for unsupported dynamic models.
        for path in named_modules:
            if not path:
                continue
            parent_path = path.rpartition(".")[0]
            source_edges.append(
                {
                    "source": _module_node_id(parent_path),
                    "target": _module_node_id(path),
                    "kind": "contains",
                }
            )

    target_graph, mappings = _build_target_preview(
        source_nodes, source_edges, module_info
    )

    status_counts = Counter(row["status"] for row in module_rows)
    operation_status_counts = Counter(record["status"] for _node, record in op_records)
    supported_statuses = {STATUS_EXACT, STATUS_TRANSPARENT_SUBCLASS, STATUS_CUSTOM_EXPANDED}
    fallback_statuses = {STATUS_FP32, STATUS_UNSUPPORTED}
    summary = {
        "total_modules": len(module_rows),
        "total_operations": len(op_records),
        "supported_modules": sum(status_counts[s] for s in supported_statuses),
        "unsupported_modules": sum(status_counts[s] for s in fallback_statuses),
        "passthrough_modules": status_counts[STATUS_PASSTHROUGH],
        "functional_supported_operations": operation_status_counts[STATUS_FUNCTIONAL],
        "unsupported_operations": (
            operation_status_counts[STATUS_UNSUPPORTED] + operation_status_counts[STATUS_FP32]
        ),
        "operation_status_counts": dict(operation_status_counts),
        # Short aliases are convenient for dashboard metric cards.
        "supported": sum(status_counts[s] for s in supported_statuses),
        "unsupported": sum(status_counts[s] for s in fallback_statuses),
        "status_counts": dict(status_counts),
        "source_node_count": len(source_nodes),
        "target_node_count": len(target_graph["nodes"]),
        "mapping_count": len(mappings),
        "trace_succeeded": graph_module is not None,
        "capture_kind": capture_kind,
        "capture_details": copy.deepcopy(capture_details),
        "schema_version": WORKBENCH_ANALYSIS_SCHEMA_VERSION,
    }

    if trace_error is not None and not module_rows:
        summary["unsupported"] = 1

    return WorkbenchAnalysis(
        model_name=model_name,
        source=source,
        source_graph={"nodes": source_nodes, "edges": source_edges},
        target_graph=target_graph,
        mappings=mappings,
        summary=summary,
        warnings=warnings,
        module_rows=module_rows,
        trace_succeeded=graph_module is not None,
        capture_kind=capture_kind,
        capture_details=capture_details,
    )


def build_conversion_plan(
    analysis: WorkbenchAnalysis | Mapping[str, Any],
    decisions: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    quant_options: Mapping[str, Any] | None = None,
    replacement_specs: Mapping[str, Mapping[str, Any]] | None = None,
    **option_overrides: Any,
) -> ConversionPlan:
    """Create an editable, exportable conversion plan from an analysis."""

    analysis_dict = analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis)
    rows = list(analysis_dict.get("module_rows", []))

    supplied: dict[str, Any] = {}
    if decisions is None:
        supplied = {}
    elif isinstance(decisions, Mapping):
        supplied = dict(decisions)
    else:
        for decision in decisions:
            key = decision.get("path") or decision.get("node_id")
            if key:
                supplied[str(key)] = decision

    normalized_decisions: dict[str, str] = {}
    normalized_replacement_specs = _normalize_replacement_specs_schema(
        replacement_specs, require_confirmed=True
    )
    warnings = list(analysis_dict.get("warnings", []))
    known_keys: set[str] = set()
    for row in rows:
        path = str(row["path"])
        node_id = str(row.get("node_id", ""))
        known_keys.update((path, node_id))
        raw_choice = supplied.get(path, supplied.get(node_id, row.get("recommended", "fp32")))
        choice = _normalize_choice(raw_choice)
        candidates = list(row.get("candidates", []))
        if choice == "replace" and path in normalized_replacement_specs:
            # A confirmed explicit replacement is intentionally available even
            # when an older analysis payload did not advertise that UI action.
            pass
        elif candidates and not _choice_matches_candidates(choice, candidates):
            warnings.append(
                f"Decision {choice!r} is not advertised for {path}; using {row.get('recommended', 'fp32')!r}."
            )
            choice = _normalize_choice(row.get("recommended", "fp32"))
        normalized_decisions[path] = choice

    unknown = sorted(str(key) for key in supplied if str(key) not in known_keys)
    if unknown:
        warnings.append("Ignored decisions for unknown graph nodes: " + ", ".join(unknown))

    known_paths = {str(row.get("path", "")) for row in rows if row.get("path")}
    unknown_replacements = sorted(set(normalized_replacement_specs) - known_paths)
    if unknown_replacements:
        raise ModelWorkbenchError(
            "Replacement specs reference unknown source module paths: "
            + ", ".join(unknown_replacements)
        )
    _validate_replacement_plan_relationships(
        normalized_decisions, normalized_replacement_specs
    )

    options = copy.deepcopy(DEFAULT_QUANT_OPTIONS)
    if quant_options:
        options.update(dict(quant_options))
    options.update(option_overrides)
    if "format" in options and "quantization_type" not in options:
        options["quantization_type"] = options.pop("format")
    capture_details = dict(analysis_dict.get("capture_details", {}) or {})
    sample_shape = capture_details.get("sample_input_shape")
    if (
        "input_size" not in options
        and isinstance(sample_shape, (list, tuple))
        and sample_shape
        and all(isinstance(dim, int) and dim > 0 for dim in sample_shape)
    ):
        options["input_size"] = tuple(sample_shape)

    return ConversionPlan(
        model_name=str(analysis_dict.get("model_name", "model")),
        source=str(analysis_dict.get("source", "custom")),
        decisions=normalized_decisions,
        quant_options=options,
        replacement_specs=normalized_replacement_specs,
        warnings=warnings,
        capture_kind=str(analysis_dict.get("capture_kind", "fx")),
    )


def preview_conversion_plan(
    analysis: WorkbenchAnalysis | Mapping[str, Any],
    plan: ConversionPlan | Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Return a target graph/mapping preview resolved for ``plan``.

    The analysis-time preview represents recommended decisions.  This helper
    applies edited module decisions without mutating that reusable analysis:

    * an FP32 module turns its complete target subtree into a fallback island;
    * a native alias collapses the module's operation subtree into one target;
    * a confirmed user replacement collapses it into the selected registry target;
    * expansion retains the operation-level one-to-many view; and
    * ordinary Quant*/passthrough selections update their target module node.
    """

    analysis_dict = (
        analysis.to_dict() if hasattr(analysis, "to_dict") else copy.deepcopy(dict(analysis))
    )
    if not isinstance(plan, ConversionPlan):
        plan = ConversionPlan.from_dict(copy.deepcopy(dict(plan)))
    _validate_replacement_plan_relationships(plan.decisions, plan.replacement_specs)

    source_graph = copy.deepcopy(analysis_dict.get("source_graph", {}))
    source_nodes = list(source_graph.get("nodes", []) or [])
    target_graph = copy.deepcopy(analysis_dict.get("target_graph", {}))
    target_graph.setdefault("nodes", [])
    target_graph.setdefault("edges", [])
    mappings = copy.deepcopy(list(analysis_dict.get("mappings", []) or []))
    rows_by_path = {
        str(row.get("path", "")): copy.deepcopy(dict(row))
        for row in analysis_dict.get("module_rows", []) or []
        if row.get("path")
    }

    replacement_paths = set(plan.replacement_specs)

    def inside_replacement(path: str) -> bool:
        return any(path == root or path.startswith(root + ".") for root in replacement_paths)

    fp32_roots = _minimal_module_paths(
        path
        for path, choice in plan.decisions.items()
        if _normalize_choice(choice) == "fp32" and not inside_replacement(path)
    )

    # FP32 is an ancestor decision: all operation and module targets below it
    # stay present for visual comparison, but every one is marked fallback.
    for path in fp32_roots:
        source_ids = _preview_source_subtree_ids(source_nodes, path)
        target_ids = _preview_target_subtree_ids(target_graph["nodes"], path)
        for node in target_graph["nodes"]:
            if str(node.get("id")) not in target_ids:
                continue
            original_label = str(node.get("label", node.get("module_type", "operation")))
            if not original_label.startswith("FP32 · "):
                node["label"] = f"FP32 · {original_label}"
            node["status"] = STATUS_FP32
            node["recommended"] = "fp32"
            node["reason"] = f"Plan keeps the complete {path!r} subtree in FP32."

        for mapping in mappings:
            mapped_sources = {str(value) for value in mapping.get("source_node_ids", [])}
            mapped_targets = {str(value) for value in mapping.get("target_node_ids", [])}
            if mapped_sources & source_ids or mapped_targets & target_ids:
                mapping["kind"] = "fp32_fallback"
                mapping["reason"] = f"Plan keeps the complete {path!r} subtree in FP32."

    def inside_fp32(path: str) -> bool:
        return any(path == root or path.startswith(root + ".") for root in fp32_roots)

    # A user replacement is a deliberate whole-module substitution.  Collapse
    # the old implementation subtree to the exact safe catalog target rather
    # than presenting it as a class alias or inferred equivalence.
    for path in sorted(replacement_paths, key=lambda value: (value.count("."), value)):
        spec = plan.replacement_specs[path]
        entry = _resolve_replacement_catalog_entry(spec["target_id"])
        source_ids = _preview_source_subtree_ids(source_nodes, path)
        source_group_id = _module_node_id(path)
        replacement_target_id = f"target:{source_group_id}"
        target_ids = _preview_target_subtree_ids(target_graph["nodes"], path)
        if replacement_target_id not in target_ids:
            raise ModelWorkbenchError(
                f"Replacement preview target for {path!r} is missing from the analysis graph."
            )
        removed_target_ids = target_ids - {replacement_target_id}
        replacement_node = next(
            node
            for node in target_graph["nodes"]
            if str(node.get("id")) == replacement_target_id
        )
        replacement_node.update(
            label=entry["target_name"],
            target=entry["target_name"],
            module_type=entry["target_name"],
            target_id=entry["id"],
            status="user_replacement",
            recommended="replace",
            reason=(
                f"User-confirmed replacement of {path!r} with registered target "
                f"{entry['target_name']}."
            ),
        )
        target_graph["nodes"] = [
            node
            for node in target_graph["nodes"]
            if str(node.get("id")) not in removed_target_ids
        ]
        target_graph["edges"] = _rewire_collapsed_preview_edges(
            target_graph.get("edges", []), removed_target_ids, replacement_target_id
        )
        mappings = [
            mapping
            for mapping in mappings
            if not (
                {str(value) for value in mapping.get("source_node_ids", [])} & source_ids
                or {str(value) for value in mapping.get("target_node_ids", [])} & target_ids
            )
        ]
        ordered_source_ids = [
            str(node.get("id"))
            for node in source_nodes
            if str(node.get("id")) in source_ids
        ]
        mappings.append(
            {
                "source_node_ids": ordered_source_ids,
                "target_node_ids": [replacement_target_id],
                "kind": "user_replacement",
                "reason": (
                    f"User explicitly mapped {path!r} to registered target "
                    f"{entry['target_name']}."
                ),
            }
        )

    # Alias roots replace a complete custom forward.  Remove all proposed
    # internal target operations/modules, rewire boundary dataflow through the
    # retained target module node, and express the result as a many-to-one map.
    alias_choices = {
        path: _normalize_choice(choice).split(":", 1)[1]
        for path, choice in plan.decisions.items()
        if _normalize_choice(choice).startswith("alias:")
        and not inside_fp32(path)
        and not inside_replacement(path)
    }
    for path, alias_target in sorted(
        alias_choices.items(), key=lambda item: (item[0].count("."), item[0])
    ):
        source_ids = _preview_source_subtree_ids(source_nodes, path)
        source_group_id = _module_node_id(path)
        alias_target_id = f"target:{source_group_id}"
        target_ids = _preview_target_subtree_ids(target_graph["nodes"], path)
        if alias_target_id not in target_ids:
            # A caller may provide a hand-built analysis graph.  In that case
            # keep the preview valid and simply leave this decision unrendered.
            continue
        removed_target_ids = target_ids - {alias_target_id}

        alias_node = next(
            node for node in target_graph["nodes"] if str(node.get("id")) == alias_target_id
        )
        alias_short_name = alias_target.rsplit(".", 1)[-1]
        alias_node.update(
            label=f"Alias · {alias_short_name}",
            target=alias_target,
            module_type=alias_short_name,
            status="native_alias",
            recommended=f"alias:{alias_target}",
            reason=(
                f"Plan explicitly treats {path!r} as {alias_target}; its internal "
                "operations collapse into this single target."
            ),
        )

        target_graph["nodes"] = [
            node
            for node in target_graph["nodes"]
            if str(node.get("id")) not in removed_target_ids
        ]
        target_graph["edges"] = _rewire_collapsed_preview_edges(
            target_graph.get("edges", []), removed_target_ids, alias_target_id
        )

        mappings = [
            mapping
            for mapping in mappings
            if not (
                {str(value) for value in mapping.get("source_node_ids", [])} & source_ids
                or {str(value) for value in mapping.get("target_node_ids", [])} & target_ids
            )
        ]
        ordered_source_ids = [
            str(node.get("id"))
            for node in source_nodes
            if str(node.get("id")) in source_ids
        ]
        mappings.append(
            {
                "source_node_ids": ordered_source_ids,
                "target_node_ids": [alias_target_id],
                "kind": "many_to_one_alias",
                "reason": (
                    f"The source group and its internal operations are explicitly aliased "
                    f"to {alias_target}."
                ),
            }
        )

    # Apply non-collapsing module choices.  Descendant choices underneath an
    # FP32/alias root cannot affect the actual conversion and are ignored here.
    for path, raw_choice in plan.decisions.items():
        choice = _normalize_choice(raw_choice)
        if inside_fp32(path) or inside_replacement(path) or path in alias_choices:
            continue
        if any(path.startswith(root + ".") for root in alias_choices):
            continue
        target_id = f"target:{_module_node_id(path)}"
        target_node = next(
            (node for node in target_graph["nodes"] if str(node.get("id")) == target_id),
            None,
        )
        if target_node is None:
            continue
        row = rows_by_path.get(path, {})
        if choice == "expand":
            label = str(target_node.get("label", row.get("type", path.rsplit(".", 1)[-1])))
            label = label.removeprefix("FP32 · ").removeprefix("Alias · ")
            if not label.startswith("Expanded · "):
                label = f"Expanded · {label}"
            target_node.update(
                label=label,
                status=STATUS_CUSTOM_EXPANDED,
                recommended="expand",
                reason=f"Plan expands {path!r} into independently converted operations.",
            )
            operation_targets = [
                str(node.get("id"))
                for node in target_graph["nodes"]
                if node.get("kind") == "operation"
                and _path_is_at_or_below(str(node.get("module_path", "")), path)
            ]
            if operation_targets and not any(
                mapping.get("kind") == "decomposed"
                and _module_node_id(path) in mapping.get("source_node_ids", [])
                and len(mapping.get("target_node_ids", [])) > 1
                for mapping in mappings
            ):
                mappings.append(
                    {
                        "source_node_ids": [_module_node_id(path)],
                        "target_node_ids": operation_targets,
                        "kind": "decomposed",
                        "reason": f"Plan expands {path!r} into independently converted operations.",
                    }
                )
        elif choice == "passthrough":
            target_node.update(
                label=str(row.get("type", target_node.get("module_type", target_node.get("label", "")))),
                target=str(target_node.get("target", row.get("type", ""))),
                module_type=str(row.get("type", target_node.get("module_type", ""))),
                status=STATUS_PASSTHROUGH,
                recommended="passthrough",
                reason=f"Plan preserves structural module {path!r}.",
            )
            _update_module_mapping_kind(mappings, path, "passthrough", target_id)
        else:
            quant_target = _resolve_preview_quant_choice(choice, row)
            if quant_target is None:
                continue
            target_node.update(
                label=quant_target,
                target=quant_target,
                module_type=quant_target,
                status="proposed_quantized",
                recommended=quant_target,
                reason=f"Plan converts {path!r} with {quant_target}.",
            )
            _update_module_mapping_kind(mappings, path, "one_to_one", target_id)

    _sanitize_preview_graph_and_mappings(source_nodes, target_graph, mappings)
    return target_graph, mappings


def convert_model(
    model: nn.Module,
    plan: ConversionPlan | Mapping[str, Any],
) -> ConversionResult:
    """Apply a conversion plan through :class:`GenericAdapter`.

    Explicit aliases are applied to a private clone before conversion.  Leaf
    custom modules selected as ``expand`` are FX-lowered in isolation;
    functional Linear nodes backed by parameters are materialized as native
    ``nn.Linear`` modules.  Composite models stay eager so a symbolic tracer
    cannot freeze the wrong side of an input-dependent branch.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")
    if not isinstance(plan, ConversionPlan):
        plan = ConversionPlan.from_dict(plan)
    _validate_replacement_plan_relationships(plan.decisions, plan.replacement_specs)

    warnings = list(plan.warnings)
    prepared = copy.deepcopy(model).cpu()

    replacement_paths = set(plan.replacement_specs)

    def is_in_replacement(path: str) -> bool:
        return any(path == root or path.startswith(root + ".") for root in replacement_paths)

    resolved_replacements: dict[str, dict[str, Any]] = {}
    replacement_entries: dict[str, dict[str, Any]] = {}
    for path in sorted(replacement_paths, key=lambda value: (value.count("."), value)):
        try:
            normalized_spec, entry, replacement = _validate_and_materialize_replacement(
                prepared, path, plan.replacement_specs[path], load_state=True
            )
            _set_submodule(prepared, path, replacement)
        except ModelWorkbenchError:
            raise
        except Exception as exc:
            raise ModelWorkbenchError(
                f"Could not apply user replacement at {path!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        replacement_entries[path] = entry
        resolved_replacements[path] = {
            "target_id": entry["id"],
            "native_type": entry["native_type"],
            "target_name": entry["target_name"],
            "state_mapping": copy.deepcopy(normalized_spec["state_mapping"]),
            "state_initializers": copy.deepcopy(normalized_spec["state_initializers"]),
            "initializer_seed": int(normalized_spec["initializer_seed"]),
        }

    fp32_roots = _minimal_module_paths(
        path for path in plan.fp32_paths if not is_in_replacement(path)
    )

    def is_in_fp32_island(path: str) -> bool:
        return any(path == root or path.startswith(root + ".") for root in fp32_roots)

    for path, alias_name in plan.alias_choices.items():
        if is_in_replacement(path):
            continue
        if is_in_fp32_island(path):
            if path not in fp32_roots:
                warnings.append(
                    f"Ignored native alias for {path!r} because ancestor FP32 island takes precedence."
                )
            continue
        if not path:
            warnings.append("Ignoring an alias decision on the root model.")
            continue
        try:
            module = prepared.get_submodule(path)
            native_cls = _resolve_native_alias(alias_name, module)
            replacement = copy.deepcopy(module)
            replacement.__class__ = native_cls
            _set_submodule(prepared, path, replacement)
        except Exception as exc:
            raise ModelWorkbenchError(
                f"Could not apply native alias {alias_name!r} to {path!r}: {exc}"
            ) from exc

    # Protect selected FP32 subtrees before any workbench or GenericAdapter FX
    # pass.  Merely setting layer_config.skip_quantization on the outer module
    # is insufficient: FX would otherwise re-enter its forward and rewrite
    # supported operations inside it.
    for path in fp32_roots:
        if not path:
            continue
        try:
            inner = prepared.get_submodule(path)
            _set_submodule(prepared, path, _WorkbenchFP32Island(inner))
        except Exception as exc:
            raise ModelWorkbenchError(f"Could not create FP32 island at {path!r}: {exc}") from exc

    effective_expand_paths = [
        path
        for path in plan.expand_paths
        if not is_in_fp32_island(path) and not is_in_replacement(path)
    ]
    locally_lowered_paths: list[str] = []
    for path in sorted(effective_expand_paths, key=lambda value: value.count("."), reverse=True):
        try:
            module = prepared.get_submodule(path)
        except Exception as exc:
            warnings.append(
                f"Could not inspect selected expansion {path!r}; the eager adapter will "
                f"handle it conservatively: {type(exc).__name__}: {exc}"
            )
            continue

        # Containers such as torchvision EncoderBlock and timm Attention are
        # converted recursively by GenericAdapter.  Flattening an entire
        # composite here would specialize dynamic control flow and can produce
        # a graph that is invalid for the user's sample shape.
        has_children = any(True for _ in module.named_children())
        if has_children and not GenericAdapter._overrides_registered_forward(module):
            continue
        try:
            tracer = _OwnershipTracer()
            lowered = GraphModule(module, tracer.trace(module))
            _materialize_functional_linears(lowered)
            lowered.train(module.training)
            _set_submodule(prepared, path, lowered)
            locally_lowered_paths.append(path)
        except Exception as exc:
            warnings.append(
                f"Local expansion of {path!r} was unavailable; its eager implementation "
                f"was preserved for adapter fallback: {type(exc).__name__}: {exc}"
            )

    layer_config = copy.deepcopy(plan.quant_options.get("layer_config", {}))
    skip_paths: set[str] = set()
    for path, _module in prepared.named_modules():
        if any(path == root or path.startswith(root + ".") for root in fp32_roots):
            skip_paths.add(path)
    for path in skip_paths:
        if not path:
            continue
        existing = dict(layer_config.get(path, {}))
        existing["skip_quantization"] = True
        layer_config[path] = existing
    for path in replacement_paths:
        existing = dict(layer_config.get(path, {}))
        existing.pop("skip_quantization", None)
        if str(existing.get("format", "")).lower() == "fp32":
            existing.pop("format", None)
        if str(existing.get("type", "")).lower() == "fp32":
            existing.pop("type", None)
        layer_config[path] = existing

    options = dict(plan.quant_options)
    options["layer_config"] = layer_config
    options.setdefault("input_quantization", False)
    options.setdefault("weight_quantization", False)
    options.setdefault("output_quantization", False)
    options.setdefault("fold_layers", False)
    options.setdefault("fold_input_norm", False)
    options.setdefault("skip_calibration", True)
    options.setdefault("enable_fx_quantization", True)
    options.setdefault("quantized_ops", ["all"])
    requested_ops = options.get("quantized_ops", ["all"])
    if isinstance(requested_ops, str):
        requested_ops = [requested_ops]
    else:
        requested_ops = list(requested_ops or [])
    excluded_ops = options.get("excluded_ops", [])
    if isinstance(excluded_ops, str):
        excluded_ops = [excluded_ops]
    else:
        excluded_ops = list(excluded_ops or [])
    options["quantized_ops"] = requested_ops
    options["excluded_ops"] = excluded_ops
    if plan.capture_kind != "fx" and options.get("enable_fx_quantization"):
        options["enable_fx_quantization"] = False
        warnings.append(
            "Whole-model functional FX rewriting was disabled because analysis used "
            f"{plan.capture_kind!r} capture. Registered module and composite converters "
            "remain enabled; unrewritten functional operations execute in FP32."
        )

    # Workbench-only keys should not leak into GenericAdapter.__init__.
    allowed = set(inspect.signature(GenericAdapter.__init__).parameters) - {
        "self",
        "model_name",
        "model",
        "model_source",
        "build_quantized",
    }
    adapter_options = {key: value for key, value in options.items() if key in allowed}
    ignored_options = sorted(set(options) - set(adapter_options))
    if ignored_options:
        warnings.append("Ignored unknown quantization options: " + ", ".join(ignored_options))

    # Realize confirmed replacements before the general adapter pass.  This is
    # deliberately path-local: adding a target's native name to quantized_ops,
    # or removing it from excluded_ops, would also quantize unrelated siblings
    # of the same type and could enable matching functional FX rewrites.
    if replacement_entries:
        staging_options = dict(adapter_options)
        staging_options.update(
            enable_fx_quantization=False,
            fold_layers=False,
            fold_input_norm=False,
        )
        try:
            staging_adapter = GenericAdapter(
                model_name=plan.model_name,
                model=prepared,
                model_source="custom",
                build_quantized=False,
                **staging_options,
            )
            staged_model = staging_adapter.model
            staging_adapter.base_model_instance = None
            guard = getattr(staged_model, "_qbench_activation_transport_guard_handle", None)
            if guard is not None:
                guard.remove()
            for attribute in (
                "_qbench_activation_transport_guard_handle",
                "_qbench_activation_transport_guarded",
                "_qbench_legacy_activation_boundaries",
            ):
                if hasattr(staged_model, attribute):
                    delattr(staged_model, attribute)

            for path, entry in replacement_entries.items():
                native_target = staged_model.get_submodule(path)
                if type(native_target) is not entry["_native_cls"]:
                    raise ModelWorkbenchError(
                        f"User replacement at {path!r} expected registered native type "
                        f"{entry['native_name']}, got {type(native_target).__name__}."
                    )
                realized = staging_adapter._create_quantized_module(
                    native_target, entry["_quantized_cls"], name=path
                )
                if realized is not native_target:
                    realized.train(native_target.training)
                    _set_submodule(staged_model, path, realized)
                realized.layer_name = path
                realized.run_id = getattr(staging_adapter, "run_id", "default")
            prepared = staged_model
            del staging_adapter
        except ModelWorkbenchError:
            raise
        except Exception as exc:
            raise ModelWorkbenchError(
                "Could not realize path-local user replacements before conversion: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        warnings.append(
            "Explicit user replacement filters were scoped to selected paths only: "
            + ", ".join(sorted(replacement_entries))
        )

    try:
        adapter = GenericAdapter(
            model_name=plan.model_name,
            model=prepared,
            model_source="custom",
            build_quantized=True,
            **adapter_options,
        )
    except Exception as exc:
        raise ModelWorkbenchError(f"QBench conversion failed: {type(exc).__name__}: {exc}") from exc

    for path, entry in replacement_entries.items():
        try:
            realized = adapter.model.get_submodule(path)
        except Exception as exc:
            raise ModelWorkbenchError(
                f"User replacement at {path!r} was not retained by conversion."
            ) from exc
        if type(realized) is not entry["_quantized_cls"]:
            raise ModelWorkbenchError(
                f"User replacement at {path!r} did not realize the selected registry target "
                f"{entry['target_name']}; got {type(realized).__name__}."
            )

    adapter.model.cpu().eval()
    recipe = plan.to_dict()
    recipe["resolved"] = {
        "fp32_paths": plan.fp32_paths,
        "expanded_paths": effective_expand_paths,
        "locally_lowered_paths": locally_lowered_paths,
        "native_aliases": plan.alias_choices,
        "user_replacements": copy.deepcopy(resolved_replacements),
        "adapter_options": _json_safe(adapter_options),
    }
    realization = _quantized_module_inventory(adapter.model)
    recipe["resolved"]["realization"] = copy.deepcopy(realization)
    return ConversionResult(
        model=adapter.model,
        adapter=adapter,
        warnings=warnings,
        recipe=recipe,
        realization=realization,
    )


def run_sample_inference(
    reference: nn.Module,
    quantized: nn.Module | ConversionResult,
    sample_input: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Run both models and compare arbitrarily nested tensor outputs."""

    quantized_model = quantized.model if isinstance(quantized, ConversionResult) else quantized
    if not isinstance(reference, nn.Module) or not isinstance(quantized_model, nn.Module):
        raise TypeError("reference and quantized must be nn.Module instances")

    ref_was_training = reference.training
    quant_was_training = quantized_model.training
    runtime_calls: Counter[str] = Counter()
    runtime_paths: dict[str, str] = {}
    hook_handles = []
    for path, module in quantized_model.named_modules():
        if not path or not _is_qbench_module(module):
            continue
        module_type = type(module).__name__

        def _record_hit(_module, _inputs, _output, *, _path=path, _type=module_type):
            runtime_calls[_type] += 1
            runtime_paths[_path] = _type

        hook_handles.append(module.register_forward_hook(_record_hit))

    reference.eval()
    quantized_model.eval()
    try:
        with torch.no_grad():
            ref_output = _invoke_model(reference, sample_input)
            quant_output = _invoke_model(quantized_model, sample_input)
    finally:
        for handle in hook_handles:
            handle.remove()
        reference.train(ref_was_training)
        quantized_model.train(quant_was_training)

    leaves: list[dict[str, Any]] = []
    structure_match = _compare_nested_outputs(
        ref_output, quant_output, path="output", leaves=leaves, atol=atol, rtol=rtol
    )
    numeric = [leaf for leaf in leaves if leaf.get("kind") == "tensor" and leaf.get("shape_match")]
    allclose = bool(structure_match and all(leaf.get("allclose", False) for leaf in numeric))
    if not numeric and structure_match:
        allclose = all(leaf.get("equal", False) for leaf in leaves)

    max_abs_error = max((leaf.get("max_abs_error", 0.0) for leaf in numeric), default=0.0)
    weighted_error = sum(
        leaf.get("mean_abs_error", 0.0) * max(leaf.get("numel", 0), 1) for leaf in numeric
    )
    total_numel = sum(max(leaf.get("numel", 0), 1) for leaf in numeric)
    mean_abs_error = weighted_error / total_numel if total_numel else 0.0
    inventory = _quantized_module_inventory(quantized_model)
    inventory_paths = set(inventory["paths"])
    executed_paths = set(runtime_paths)

    return {
        "reference_output": ref_output,
        "quantized_output": quant_output,
        "comparison": {
            "structure_match": structure_match,
            "allclose": allclose,
            "atol": float(atol),
            "rtol": float(rtol),
            "max_abs_error": float(max_abs_error),
            "mean_abs_error": float(mean_abs_error),
            "leaves": leaves,
        },
        "reference_summary": _output_summary(ref_output),
        "quantized_summary": _output_summary(quant_output),
        "runtime_audit": {
            "quantized_modules_total": inventory["total"],
            "quantized_modules_by_type": inventory["by_type"],
            "executed_quantized_modules": len(executed_paths),
            "runtime_calls_by_type": dict(runtime_calls),
            "executed_paths": sorted(executed_paths),
            "not_executed_paths": sorted(inventory_paths - executed_paths),
        },
    }


class _ClassificationTargetMap:
    """Pickle-safe ImageFolder target remapping for DataLoader workers."""

    def __init__(self, index_map: Mapping[int, int]):
        self.index_map = {int(key): int(value) for key, value in index_map.items()}

    def __call__(self, target: int) -> int:
        return self.index_map.get(int(target), int(target))


def build_classification_validation_loader(
    *,
    dataset_kind: str,
    model: nn.Module | ConversionResult,
    source: str,
    model_name: str,
    path: str | os.PathLike[str] | None = None,
    custom_factory: str | Callable[..., Any] | None = None,
    batch_size: int = 8,
    max_samples: int | None = 128,
    seed: int = 42,
    num_workers: int = 0,
    factory_kwargs: Mapping[str, Any] | None = None,
    split: str = "val",
    pin_memory: bool | None = None,
) -> tuple[DataLoader, dict[str, Any]]:
    """Build a deterministic validation loader for the workbench dashboard.

    ``image_folder`` uses model-appropriate inference transforms and, when the
    directory contains ImageNet WordNet IDs, converts ImageFolder's local
    alphabetical targets to canonical ImageNet indices.  ``custom_factory``
    accepts either a Dataset or a fully configured DataLoader, which keeps the
    API usable for non-image and application-specific classification data.
    """

    kind = _normalize_classification_dataset_kind(dataset_kind)
    normalized_source = _normalize_source(source)
    batch_size = _positive_int(batch_size, "batch_size")
    num_workers = _nonnegative_int(num_workers, "num_workers")
    seed = _integer_value(seed, "seed")
    if max_samples is not None:
        max_samples = _positive_int(max_samples, "max_samples")
    if factory_kwargs is not None and not isinstance(factory_kwargs, Mapping):
        raise ModelWorkbenchError("factory_kwargs must be a mapping when provided.")
    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    warnings: list[str] = []
    transform: Any = None
    transform_source = "custom_factory"
    transform_config: dict[str, Any] = {}
    # A custom tensor factory should not need torchvision installed.  Image
    # folders always need a transform; torchvision/timm factories commonly
    # accept one as well.
    if kind == "image_folder" or normalized_source in {"torchvision", "timm"}:
        transform, transform_source, transform_config, transform_warnings = (
            _classification_validation_transform(
                model=model,
                source=normalized_source,
                model_name=str(model_name),
            )
        )
        warnings.extend(transform_warnings)

    resolved_path: str | None = None
    full_dataset_size: int | None = None
    canonical_map: dict[int, int] = {}
    factory_controls_loader = False

    if kind == "image_folder":
        if path is None or not str(os.fspath(path)).strip():
            raise ModelWorkbenchError("path is required for an image_folder dataset.")
        requested_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        if not os.path.isdir(requested_path):
            raise ModelWorkbenchError(f"ImageFolder directory does not exist: {requested_path}")
        split_path = os.path.join(requested_path, str(split)) if str(split) else requested_path
        resolved_path = split_path if os.path.isdir(split_path) else requested_path
        try:
            from torchvision.datasets import ImageFolder

            dataset: Dataset[Any] = ImageFolder(resolved_path, transform=transform)
        except Exception as exc:
            raise ModelWorkbenchError(
                f"Could not build ImageFolder dataset at {resolved_path!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        canonical_map, mapping_warnings = _apply_canonical_imagenet_targets(dataset)
        warnings.extend(mapping_warnings)
        full_dataset_size = _safe_len(dataset)
        dataset = _deterministic_dataset_subset(dataset, max_samples=max_samples, seed=seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
            generator=generator,
        )
    else:
        if custom_factory is None:
            raise ModelWorkbenchError(
                "custom_factory is required when dataset_kind='custom_factory'."
            )
        factory = _resolve_dataset_factory(custom_factory)
        resolved_path = (
            os.path.abspath(os.path.expanduser(os.fspath(path))) if path is not None else None
        )
        produced = _call_dataset_factory(
            factory,
            model=model.model if isinstance(model, ConversionResult) else model,
            source=normalized_source,
            model_name=str(model_name),
            path=resolved_path,
            split=str(split),
            transform=transform,
            batch_size=batch_size,
            max_samples=max_samples,
            seed=seed,
            num_workers=num_workers,
            factory_kwargs=dict(factory_kwargs or {}),
        )
        if isinstance(produced, DataLoader):
            loader = produced
            dataset = produced.dataset
            full_dataset_size = _safe_len(dataset)
            factory_controls_loader = True
        elif isinstance(produced, Dataset):
            dataset = produced
            full_dataset_size = _safe_len(dataset)
            if isinstance(dataset, IterableDataset):
                if max_samples is not None:
                    warnings.append(
                        "The custom factory returned an IterableDataset; max_samples will "
                        "be enforced by the benchmark instead of by a random Subset."
                    )
            else:
                dataset = _deterministic_dataset_subset(
                    dataset, max_samples=max_samples, seed=seed
                )
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=bool(pin_memory),
                generator=generator,
            )
        else:
            raise ModelWorkbenchError(
                "Custom dataset factory must return torch.utils.data.Dataset or "
                f"DataLoader, but returned {type(produced).__name__}."
            )

    class_count, class_names = _classification_dataset_classes(dataset)
    subset_size = _safe_len(dataset)
    actual_batch_size = getattr(loader, "batch_size", None)
    metadata: dict[str, Any] = {
        "api_version": WORKBENCH_DATASET_BENCHMARK_API_VERSION,
        "dataset_kind": kind,
        "source": normalized_source,
        "model_name": str(model_name),
        "path": os.fspath(path) if path is not None else None,
        "resolved_path": resolved_path,
        "split": str(split),
        "dataset_size": full_dataset_size,
        "subset_size": subset_size,
        "requested_max_samples": max_samples,
        "class_count": class_count,
        "class_names": class_names,
        "canonical_imagenet_labels_mapped": len(canonical_map),
        "canonical_target_map": {str(key): value for key, value in canonical_map.items()},
        "transform_source": transform_source,
        "transform_config": _json_safe(transform_config),
        "batch_size": int(actual_batch_size) if actual_batch_size is not None else None,
        "num_workers": int(getattr(loader, "num_workers", num_workers)),
        "pin_memory": bool(getattr(loader, "pin_memory", pin_memory)),
        "seed": seed,
        "factory_controls_loader": factory_controls_loader,
        "warnings": warnings,
    }
    # The dashboard can pass only the loader to the benchmark and still retain
    # the exact dataset provenance in its exported result.
    try:
        loader.qbench_metadata = copy.deepcopy(metadata)
    except Exception as exc:  # pragma: no cover - defensive for exotic subclasses
        raise ModelWorkbenchError(
            f"Could not attach qbench_metadata to the validation loader: {exc}"
        ) from exc
    return loader, metadata


def benchmark_classification_models(
    reference: nn.Module,
    quantized: nn.Module | ConversionResult,
    data_loader: Iterable[Any],
    *,
    max_samples: int | None = None,
    device: str | torch.device = "auto",
    progress_callback: Callable[[int, int], Any] | None = None,
) -> dict[str, Any]:
    """Benchmark two classifiers on exactly the same validation batches.

    Accuracy values are percentages and deltas are percentage points
    (quantized minus reference).  Input tensors are cloned for each invocation
    so an in-place operation in one model cannot alter the other model's batch.
    """

    quantized_model = quantized.model if isinstance(quantized, ConversionResult) else quantized
    if not isinstance(reference, nn.Module) or not isinstance(quantized_model, nn.Module):
        raise ModelWorkbenchError("reference and quantized must be nn.Module instances.")
    if not hasattr(data_loader, "__iter__"):
        raise ModelWorkbenchError("data_loader must be an iterable of labeled batches.")
    if max_samples is not None:
        max_samples = _positive_int(max_samples, "max_samples")
    if progress_callback is not None and not callable(progress_callback):
        raise ModelWorkbenchError("progress_callback must be callable when provided.")

    target_device = _resolve_benchmark_device(device)
    loader_metadata = copy.deepcopy(getattr(data_loader, "qbench_metadata", {}))
    if not isinstance(loader_metadata, Mapping):
        loader_metadata = {}
    else:
        loader_metadata = dict(loader_metadata)
    expected_total = _classification_benchmark_total(
        data_loader, loader_metadata=loader_metadata, max_samples=max_samples
    )

    unique_models: list[nn.Module] = []
    for candidate in (reference, quantized_model):
        if all(candidate is not existing for existing in unique_models):
            unique_models.append(candidate)
    snapshots = {id(model): _snapshot_model_runtime_state(model) for model in unique_models}

    samples = 0
    batches = 0
    class_dimension: int | None = None
    effective_top5_k: int | None = None
    agreement_count = 0
    reference_top1 = 0
    reference_top5 = 0
    quantized_top1 = 0
    quantized_top5 = 0
    reference_elapsed = 0.0
    quantized_elapsed = 0.0

    try:
        for model in unique_models:
            model.to(target_device)
            model.eval()

        with torch.inference_mode():
            for raw_batch in data_loader:
                if max_samples is not None and samples >= max_samples:
                    break
                batch_inputs, batch_targets = _split_classification_batch(raw_batch)
                targets = _classification_targets(batch_targets)
                batch_size_value = int(targets.shape[0])
                if batch_size_value == 0:
                    continue
                take = batch_size_value
                if max_samples is not None:
                    take = min(take, max_samples - samples)
                if take < batch_size_value:
                    batch_inputs = _slice_classification_batch(
                        batch_inputs, take=take, batch_size=batch_size_value
                    )
                    targets = targets[:take]

                targets = targets.to(target_device, non_blocking=True)
                reference_inputs = _clone_batch_to_device(batch_inputs, target_device)
                quantized_inputs = _clone_batch_to_device(batch_inputs, target_device)

                reference_logits, elapsed = _timed_classification_logits(
                    reference, reference_inputs, target_device, role="reference"
                )
                reference_elapsed += elapsed
                quantized_logits, elapsed = _timed_classification_logits(
                    quantized_model, quantized_inputs, target_device, role="quantized"
                )
                quantized_elapsed += elapsed

                if reference_logits.shape[0] != take:
                    raise ModelWorkbenchError(
                        "Reference logits batch dimension does not match the labels: "
                        f"{reference_logits.shape[0]} versus {take}."
                    )
                if quantized_logits.shape[0] != take:
                    raise ModelWorkbenchError(
                        "Quantized logits batch dimension does not match the labels: "
                        f"{quantized_logits.shape[0]} versus {take}."
                    )
                if reference_logits.shape[1] != quantized_logits.shape[1]:
                    raise ModelWorkbenchError(
                        "Reference and quantized classifiers returned different class "
                        f"dimensions: {reference_logits.shape[1]} versus "
                        f"{quantized_logits.shape[1]}."
                    )
                current_class_dimension = int(reference_logits.shape[1])
                if current_class_dimension <= 0:
                    raise ModelWorkbenchError("Classifier logits must contain at least one class.")
                if class_dimension is None:
                    class_dimension = current_class_dimension
                    effective_top5_k = min(5, class_dimension)
                elif class_dimension != current_class_dimension:
                    raise ModelWorkbenchError(
                        "Classifier class dimension changed between batches: "
                        f"{class_dimension} then {current_class_dimension}."
                    )
                if bool((targets < 0).any()) or bool((targets >= class_dimension).any()):
                    target_min = int(targets.min().item())
                    target_max = int(targets.max().item())
                    raise ModelWorkbenchError(
                        f"Classification targets [{target_min}, {target_max}] are outside "
                        f"the logits class range [0, {class_dimension - 1}]."
                    )

                assert effective_top5_k is not None
                ref_prediction = reference_logits.argmax(dim=1)
                quant_prediction = quantized_logits.argmax(dim=1)
                reference_top1 += int(ref_prediction.eq(targets).sum().item())
                quantized_top1 += int(quant_prediction.eq(targets).sum().item())
                reference_top5 += int(
                    reference_logits.topk(effective_top5_k, dim=1).indices
                    .eq(targets.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )
                quantized_top5 += int(
                    quantized_logits.topk(effective_top5_k, dim=1).indices
                    .eq(targets.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )
                agreement_count += int(ref_prediction.eq(quant_prediction).sum().item())
                samples += take
                batches += 1
                if progress_callback is not None:
                    try:
                        progress_callback(samples, expected_total)
                    except Exception as exc:
                        raise ModelWorkbenchError(
                            f"progress_callback failed: {type(exc).__name__}: {exc}"
                        ) from exc
    except ModelWorkbenchError:
        raise
    except Exception as exc:
        raise ModelWorkbenchError(
            f"Classification benchmark failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        restore_errors: list[str] = []
        for model in unique_models:
            try:
                _restore_model_runtime_state(model, snapshots[id(model)])
            except Exception as exc:  # pragma: no cover - highly unusual device failure
                restore_errors.append(f"{type(model).__name__}: {exc}")
        if restore_errors:
            raise ModelWorkbenchError(
                "Could not restore model runtime state after benchmarking: "
                + "; ".join(restore_errors)
            )

    if samples == 0 or class_dimension is None or effective_top5_k is None:
        raise ModelWorkbenchError("The validation loader produced no labeled samples.")

    def _model_metrics(correct_top1: int, correct_top5: int, elapsed: float) -> dict[str, Any]:
        return {
            "top1_accuracy_percent": 100.0 * correct_top1 / samples,
            "top5_accuracy_percent": 100.0 * correct_top5 / samples,
            "correct_top1": int(correct_top1),
            "correct_top5": int(correct_top5),
            "elapsed_seconds": float(elapsed),
            "throughput_samples_per_second": float(samples / elapsed) if elapsed > 0 else 0.0,
        }

    reference_metrics = _model_metrics(reference_top1, reference_top5, reference_elapsed)
    quantized_metrics = _model_metrics(quantized_top1, quantized_top5, quantized_elapsed)
    return {
        "api_version": WORKBENCH_DATASET_BENCHMARK_API_VERSION,
        "device": str(target_device),
        "samples": samples,
        "batches": batches,
        "class_dimension": class_dimension,
        "class_count": class_dimension,
        "effective_top5_k": effective_top5_k,
        "requested_max_samples": max_samples,
        "reference": reference_metrics,
        "quantized": quantized_metrics,
        "delta": {
            "top1_accuracy_percentage_points": (
                quantized_metrics["top1_accuracy_percent"]
                - reference_metrics["top1_accuracy_percent"]
            ),
            "top5_accuracy_percentage_points": (
                quantized_metrics["top5_accuracy_percent"]
                - reference_metrics["top5_accuracy_percent"]
            ),
        },
        "prediction_agreement_percent": 100.0 * agreement_count / samples,
        "prediction_agreement_count": agreement_count,
        "loader_metadata": copy.deepcopy(loader_metadata),
    }


def _normalize_classification_dataset_kind(dataset_kind: str) -> str:
    normalized = str(dataset_kind).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "image_folder": "image_folder",
        "imagefolder": "image_folder",
        "imagenet": "image_folder",
        "folder": "image_folder",
        "custom_factory": "custom_factory",
        "custom": "custom_factory",
        "factory": "custom_factory",
    }
    if normalized not in aliases:
        raise ModelWorkbenchError(
            "dataset_kind must be 'image_folder' or 'custom_factory', got "
            f"{dataset_kind!r}."
        )
    return aliases[normalized]


def _integer_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ModelWorkbenchError(f"{name} must be an integer, not a boolean.")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelWorkbenchError(f"{name} must be an integer, got {value!r}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ModelWorkbenchError(f"{name} must be an integer, got {value!r}.")
    return converted


def _positive_int(value: Any, name: str) -> int:
    converted = _integer_value(value, name)
    if converted <= 0:
        raise ModelWorkbenchError(f"{name} must be greater than zero, got {converted}.")
    return converted


def _nonnegative_int(value: Any, name: str) -> int:
    converted = _integer_value(value, name)
    if converted < 0:
        raise ModelWorkbenchError(f"{name} must be non-negative, got {converted}.")
    return converted


def _classification_validation_transform(
    *, model: nn.Module | ConversionResult, source: str, model_name: str
) -> tuple[Any, str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    if source == "torchvision":
        try:
            from torchvision import models as torchvision_models

            weights_enum = torchvision_models.get_model_weights(model_name)
            weights = weights_enum.DEFAULT
            if weights is None:
                raise LookupError(f"{model_name!r} has no DEFAULT weights metadata")
            return (
                weights.transforms(),
                f"torchvision:{model_name}:DEFAULT",
                {"weights": str(weights)},
                warnings,
            )
        except Exception as exc:
            warnings.append(
                f"Could not resolve torchvision DEFAULT transforms for {model_name!r} "
                f"({_brief_exception(exc)}); using standard ImageNet evaluation transforms."
            )
    elif source == "timm":
        try:
            import timm

            candidate = model.model if isinstance(model, ConversionResult) else model
            try:
                data_config = timm.data.resolve_model_data_config(candidate)
            except Exception:
                candidate = timm.create_model(model_name, pretrained=False)
                data_config = timm.data.resolve_model_data_config(candidate)
            return (
                timm.data.create_transform(**data_config, is_training=False),
                f"timm:{model_name}:data_config",
                dict(data_config),
                warnings,
            )
        except Exception as exc:
            warnings.append(
                f"Could not resolve timm transforms for {model_name!r} "
                f"({_brief_exception(exc)}); using standard ImageNet evaluation transforms."
            )

    try:
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    except Exception as exc:
        raise ModelWorkbenchError(
            "torchvision is required to build image classification transforms: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return (
        transform,
        "imagenet_standard_224",
        {
            "resize": 256,
            "crop": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        warnings,
    )


def _resolve_dataset_factory(value: str | Callable[..., Any]) -> Callable[..., Any]:
    # Reuse the model factory import grammar, but keep the dataset-specific
    # error below so a dashboard user knows which field needs correction.
    try:
        return _resolve_custom_factory(value)
    except ModelWorkbenchError as exc:
        raise ModelWorkbenchError(f"Invalid dataset custom_factory: {exc}") from exc


def _call_dataset_factory(
    factory: Callable[..., Any],
    *,
    model: nn.Module,
    source: str,
    model_name: str,
    path: str | None,
    split: str,
    transform: Any,
    batch_size: int,
    max_samples: int | None,
    seed: int,
    num_workers: int,
    factory_kwargs: dict[str, Any],
) -> Any:
    available: dict[str, Any] = {
        "model": model,
        "source": source,
        "model_source": source,
        "model_name": model_name,
        "path": path,
        "root": path,
        "dataset_path": path,
        "split": split,
        "transform": transform,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "subset_size": max_samples,
        "seed": seed,
        "num_workers": num_workers,
    }
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None

    args: list[Any] = []
    if signature is None:
        kwargs = dict(factory_kwargs)
    else:
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        unknown = set(factory_kwargs) - set(parameters)
        if unknown and not accepts_kwargs:
            raise ModelWorkbenchError(
                "Unsupported custom dataset factory arguments: " + ", ".join(sorted(unknown))
            )
        values = dict(available)
        values.update(factory_kwargs)
        kwargs: dict[str, Any] = {}
        for parameter in parameters.values():
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                if parameter.name in values:
                    args.append(values[parameter.name])
                elif parameter.default is inspect.Parameter.empty:
                    raise ModelWorkbenchError(
                        f"Custom dataset factory requires unsupported positional argument "
                        f"{parameter.name!r}."
                    )
            elif parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ) and parameter.name in values:
                kwargs[parameter.name] = values[parameter.name]
        if accepts_kwargs:
            kwargs.update({key: value for key, value in values.items() if key not in kwargs})
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        name = getattr(factory, "__qualname__", repr(factory))
        raise ModelWorkbenchError(
            f"Custom dataset factory {name} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _safe_len(value: Any) -> int | None:
    try:
        return int(len(value))
    except (TypeError, AttributeError, NotImplementedError):
        return None


def _deterministic_dataset_subset(
    dataset: Dataset[Any], *, max_samples: int | None, seed: int
) -> Dataset[Any]:
    size = _safe_len(dataset)
    if max_samples is None or size is None or max_samples >= size:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def _apply_canonical_imagenet_targets(
    dataset: Dataset[Any],
) -> tuple[dict[int, int], list[str]]:
    warnings: list[str] = []
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if not isinstance(class_to_idx, Mapping):
        return {}, warnings
    index_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "imagenet_class_index.json")
    )
    if not os.path.isfile(index_path):
        warnings.append(
            f"Canonical ImageNet class mapping was not found at {index_path}; local "
            "ImageFolder targets will be used."
        )
        return {}, warnings
    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            class_index = json.load(handle)
        wnid_to_index = {
            str(value[0]): int(key)
            for key, value in class_index.items()
            if isinstance(value, Sequence) and value
        }
    except Exception as exc:
        warnings.append(
            f"Could not read canonical ImageNet class mapping ({_brief_exception(exc)}); "
            "local ImageFolder targets will be used."
        )
        return {}, warnings
    index_map = {
        int(local_index): wnid_to_index[str(class_name)]
        for class_name, local_index in class_to_idx.items()
        if str(class_name) in wnid_to_index
    }
    if index_map:
        dataset.target_transform = _ClassificationTargetMap(index_map)
    total_classes = len(class_to_idx)
    if not index_map:
        warnings.append(
            "No ImageFolder class names matched canonical ImageNet WordNet IDs; local "
            "class indices will be used."
        )
    elif len(index_map) < total_classes:
        warnings.append(
            f"Only {len(index_map)}/{total_classes} ImageFolder classes matched canonical "
            "ImageNet WordNet IDs; unmatched classes retain local indices."
        )
    return index_map, warnings


def _classification_dataset_classes(dataset: Any) -> tuple[int | None, list[str] | None]:
    current = dataset
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        classes = getattr(current, "classes", None)
        if isinstance(classes, Sequence) and not isinstance(classes, (str, bytes)):
            return len(classes), [str(value) for value in classes]
        targets = getattr(current, "targets", None)
        if targets is None and hasattr(current, "tensors"):
            tensors = getattr(current, "tensors")
            if isinstance(tensors, Sequence) and len(tensors) >= 2:
                targets = tensors[1]
        if targets is not None:
            try:
                target_tensor = torch.as_tensor(targets).reshape(-1)
                if target_tensor.numel():
                    unique = torch.unique(target_tensor).tolist()
                    return len(unique), [str(value) for value in unique]
            except (TypeError, ValueError, RuntimeError):
                pass
        current = getattr(current, "dataset", None)
    return None, None


def _resolve_benchmark_device(value: str | torch.device) -> torch.device:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ModelWorkbenchError(
            f"device must be 'auto', 'cpu', or a CUDA device, got {value!r}."
        ) from exc
    if resolved.type not in {"cpu", "cuda"}:
        raise ModelWorkbenchError(
            f"device must be 'auto', 'cpu', or a CUDA device, got {value!r}."
        )
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ModelWorkbenchError("CUDA was requested but is not available.")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ModelWorkbenchError(
                f"CUDA device index {resolved.index} is unavailable; found "
                f"{torch.cuda.device_count()} device(s)."
            )
    return resolved


def _snapshot_model_runtime_state(model: nn.Module) -> dict[str, Any]:
    devices = {
        tensor.device
        for tensor in list(model.parameters()) + list(model.buffers())
        if tensor.device.type != "meta"
    }
    if len(devices) > 1:
        rendered = ", ".join(sorted(str(device) for device in devices))
        raise ModelWorkbenchError(
            f"Multi-device model {type(model).__name__} is not supported by the "
            f"dashboard benchmark ({rendered})."
        )
    return {
        "device": next(iter(devices), torch.device("cpu")),
        "training": [(module, bool(module.training)) for module in model.modules()],
    }


def _restore_model_runtime_state(model: nn.Module, snapshot: Mapping[str, Any]) -> None:
    model.to(snapshot["device"])
    for module, was_training in snapshot["training"]:
        module.training = bool(was_training)


def _classification_benchmark_total(
    data_loader: Any, *, loader_metadata: Mapping[str, Any], max_samples: int | None
) -> int:
    available: int | None = None
    metadata_size = loader_metadata.get("subset_size")
    if isinstance(metadata_size, int) and metadata_size >= 0:
        available = metadata_size
    if available is None:
        available = _safe_len(getattr(data_loader, "dataset", None))
    if available is None:
        available = 0
    return min(available, max_samples) if max_samples is not None else available


def _split_classification_batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, Mapping):
        target_key = next(
            (key for key in ("targets", "target", "labels", "label", "y") if key in batch),
            None,
        )
        if target_key is None:
            raise ModelWorkbenchError(
                "Mapping batches must contain one of: targets, target, labels, label, y."
            )
        inputs = {key: value for key, value in batch.items() if key != target_key}
        if not inputs:
            raise ModelWorkbenchError("A classification batch contains labels but no inputs.")
        if len(inputs) == 1:
            only_key = next(iter(inputs))
            if str(only_key) in {"input", "inputs", "image", "images", "x", "data"}:
                return inputs[only_key], batch[target_key]
        return inputs, batch[target_key]
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ModelWorkbenchError(
                "Sequence batches must contain at least (inputs, targets)."
            )
        return batch[0], batch[1]
    raise ModelWorkbenchError(
        "Classification batches must be a mapping or an (inputs, targets) sequence, "
        f"got {type(batch).__name__}."
    )


def _classification_targets(value: Any) -> torch.Tensor:
    try:
        targets = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except Exception as exc:
        raise ModelWorkbenchError(
            f"Could not convert classification targets to a tensor: {exc}"
        ) from exc
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if targets.ndim != 1:
        raise ModelWorkbenchError(
            "Classification targets must have shape [batch] or [batch, 1], got "
            f"{list(targets.shape)}."
        )
    return targets.to(dtype=torch.long)


def _slice_classification_batch(value: Any, *, take: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[:take]
        return value
    if isinstance(value, Mapping):
        return {
            key: _slice_classification_batch(child, take=take, batch_size=batch_size)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _slice_classification_batch(child, take=take, batch_size=batch_size)
            for child in value
        )
    if isinstance(value, list):
        return [
            _slice_classification_batch(child, take=take, batch_size=batch_size)
            for child in value
        ]
    return value


def _clone_batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True).clone()
    if isinstance(value, Mapping):
        return {key: _clone_batch_to_device(child, device) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_batch_to_device(child, device) for child in value)
    if isinstance(value, list):
        return [_clone_batch_to_device(child, device) for child in value]
    return copy.deepcopy(value)


def _timed_classification_logits(
    model: nn.Module, inputs: Any, device: torch.device, *, role: str
) -> tuple[torch.Tensor, float]:
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = _invoke_model(model, inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        logits = _extract_classification_logits(output)
    except ModelWorkbenchError:
        raise
    except Exception as exc:
        raise ModelWorkbenchError(
            f"The {role} model failed on a validation batch: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if logits.ndim != 2:
        raise ModelWorkbenchError(
            f"The {role} model must return 2-D logits [batch, classes], got "
            f"shape {list(logits.shape)}."
        )
    return logits, elapsed


def _extract_classification_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        for key in ("logits", "out", "output", "scores"):
            if key in output:
                return _extract_classification_logits(output[key])
        tensors = [value for value in output.values() if isinstance(value, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        raise ModelWorkbenchError(
            "Model output mapping does not identify logits; use a logits/out/output/scores key."
        )
    logits = getattr(output, "logits", None)
    if logits is not None and logits is not output:
        return _extract_classification_logits(logits)
    if isinstance(output, (tuple, list)) and output:
        for value in output:
            try:
                return _extract_classification_logits(value)
            except ModelWorkbenchError:
                continue
    raise ModelWorkbenchError(
        f"Could not extract classification logits from {type(output).__name__} output."
    )


def _normalize_source(source: str) -> str:
    normalized = str(source).strip().lower().replace("_", "")
    aliases = {
        "torchvision": "torchvision",
        "torchvision.models": "torchvision",
        "tv": "torchvision",
        "timm": "timm",
        "custom": "custom",
        "factory": "custom",
        "all": "all",
    }
    return aliases.get(normalized, str(source).strip().lower())


def _resolve_custom_factory(value: str | Callable[..., nn.Module]) -> Callable[..., nn.Module]:
    if callable(value):
        return value
    if not isinstance(value, str) or ":" not in value:
        raise ModelWorkbenchError(
            "custom_factory must use 'package.module:callable' syntax."
        )
    module_name, attribute_path = value.split(":", 1)
    if not module_name or not attribute_path:
        raise ModelWorkbenchError(
            "custom_factory must use 'package.module:callable' syntax."
        )
    try:
        resolved: Any = importlib.import_module(module_name)
        for part in attribute_path.split("."):
            resolved = getattr(resolved, part)
    except Exception as exc:
        raise ModelWorkbenchError(f"Could not import custom factory {value!r}: {exc}") from exc
    if not callable(resolved):
        raise ModelWorkbenchError(f"Imported custom factory {value!r} is not callable.")
    return resolved


def _call_custom_factory(
    factory: Callable[..., nn.Module], model_name: str, pretrained: bool
) -> nn.Module:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None

    kwargs: dict[str, Any] = {}
    args: list[Any] = []
    if signature is not None:
        parameters = signature.parameters
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in parameters.values())
        if "model_name" in parameters or accepts_kwargs:
            kwargs["model_name"] = model_name
        elif "name" in parameters:
            kwargs["name"] = model_name
        else:
            required_positional = [
                p
                for p in parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]
            if required_positional:
                args.append(model_name)
        if "pretrained" in parameters or accepts_kwargs:
            kwargs["pretrained"] = pretrained
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        name = getattr(factory, "__qualname__", repr(factory))
        raise ModelWorkbenchError(f"Custom factory {name} failed: {type(exc).__name__}: {exc}") from exc


def _initial_module_info(
    path: str,
    module: nn.Module,
    supported_ops: Mapping[type[nn.Module], type[nn.Module]],
) -> dict[str, Any]:
    module_type = type(module)
    exact_quant = supported_ops.get(module_type)
    if exact_quant is not None:
        return {
            "status": STATUS_EXACT,
            "reason": f"Exact registry match: {module_type.__name__} -> {exact_quant.__name__}.",
            "candidates": [exact_quant.__name__, "fp32"],
            "recommended": exact_quant.__name__,
            "quant_target": exact_quant.__name__,
            "native_base": _qualified_type_name(module_type),
            "custom_type": False,
        }

    # GenericAdapter has a few behavior-preserving composite converters that
    # intentionally live outside the native OpRegistry lookup.  Surface those
    # capabilities here so the workbench does not label timm attention or
    # fused BatchNorm+activation blocks as unsupported.
    if (
        isinstance(module, nn.BatchNorm2d)
        and type(module) is not nn.BatchNorm2d
        and (hasattr(module, "act") or hasattr(module, "drop"))
    ):
        return {
            "status": STATUS_EXACT,
            "reason": (
                "GenericAdapter has a behavior-preserving fused converter: "
                f"{module_type.__name__} -> QuantBatchNormAct2d."
            ),
            "candidates": ["QuantBatchNormAct2d", "fp32"],
            "recommended": "QuantBatchNormAct2d",
            "quant_target": "QuantBatchNormAct2d",
            "native_base": _qualified_type_name(nn.BatchNorm2d),
            "custom_type": True,
        }

    if GenericAdapter._is_timm_attention_like(module):
        return {
            "status": STATUS_CUSTOM_EXPANDED,
            "reason": (
                "GenericAdapter decomposes this qkv/proj attention block into "
                "QBench attention arithmetic."
            ),
            "candidates": ["DecomposedQkvAttention", "fp32"],
            "recommended": "DecomposedQkvAttention",
            "quant_target": "DecomposedQkvAttention",
            "native_base": None,
            "custom_type": True,
        }

    if GenericAdapter._is_timm_mlp_like(module):
        return {
            "status": STATUS_CUSTOM_EXPANDED,
            "reason": (
                "GenericAdapter decomposes this fc1/fc2 MLP block into QBench operations."
            ),
            "candidates": ["DecomposedMlpBlock", "fp32"],
            "recommended": "DecomposedMlpBlock",
            "quant_target": "DecomposedMlpBlock",
            "native_base": None,
            "custom_type": True,
        }

    matching_base: type[nn.Module] | None = None
    matching_quant: type[nn.Module] | None = None
    for base in module_type.__mro__[1:]:
        if base in supported_ops:
            matching_base = base
            matching_quant = supported_ops[base]
            break

    custom_type = not module_type.__module__.startswith("torch.nn")
    if matching_base is not None and matching_quant is not None:
        overrides_forward = module_type.forward is not matching_base.forward
        native_name = _qualified_type_name(matching_base)
        if not overrides_forward:
            return {
                "status": STATUS_TRANSPARENT_SUBCLASS,
                "reason": (
                    f"Subclass inherits {matching_base.__name__}.forward unchanged; "
                    f"it can use {matching_quant.__name__}."
                ),
                "candidates": [matching_quant.__name__, f"alias:{native_name}", "fp32"],
                "recommended": matching_quant.__name__,
                "quant_target": matching_quant.__name__,
                "native_base": native_name,
                "custom_type": custom_type,
            }
        return {
            "status": STATUS_CUSTOM_EXPANDED,
            "reason": (
                f"Subclass overrides {matching_base.__name__}.forward; inspect and convert its operations."
            ),
            "candidates": ["expand", f"alias:{native_name}", "fp32"],
            "recommended": "expand",
            "quant_target": None,
            "native_base": native_name,
            "custom_type": True,
        }

    children = list(module.named_children())
    if path == "" or children or _is_passthrough_module(module):
        return {
            "status": STATUS_PASSTHROUGH,
            "reason": "Structural/container operation is preserved while child operations are converted.",
            "candidates": ["passthrough", "fp32"],
            "recommended": "passthrough",
            "quant_target": type(module).__name__,
            "native_base": None,
            "custom_type": custom_type,
        }

    return {
        "status": STATUS_FP32,
        "reason": "No registered QBench replacement; preserve this module in FP32.",
        "candidates": ["fp32"],
        "recommended": "fp32",
        "quant_target": type(module).__name__,
        "native_base": None,
        "custom_type": custom_type,
    }


def _operation_record(
    node: Node,
    graph_module: GraphModule,
    module_info: Mapping[str, Mapping[str, Any]],
    non_tensor_nodes: set[str] | None = None,
    capture_kind: str = "fx",
) -> dict[str, Any]:
    owner_path = _node_owner_path(node)
    shape = _node_tensor_shape(node)

    if node.op == "placeholder":
        return _make_op_record(node, owner_path, "input", "Model input", ["passthrough"], "passthrough", shape)
    if node.op == "output":
        return _make_op_record(node, owner_path, "output", "Model output", ["passthrough"], "passthrough", shape)
    if node.op == "get_attr":
        return _make_op_record(
            node,
            owner_path,
            STATUS_PASSTHROUGH,
            "Parameter/buffer lookup; represented as a structural edge.",
            ["passthrough"],
            "passthrough",
            shape,
        )

    if node.op == "call_module":
        path = str(node.target)
        owner_path = path
        module = graph_module.get_submodule(path)
        info = module_info.get(path) or _initial_module_info(
            path, module, OpRegistry.get_supported_ops()
        )
        record = _make_op_record(
            node,
            owner_path,
            str(info["status"]),
            str(info["reason"]),
            list(info["candidates"]),
            str(info["recommended"]),
            shape,
        )
        record["label"] = type(module).__name__
        record["module_type"] = type(module).__name__
        record["quantized_target"] = info.get("quant_target")
        return record

    if capture_kind == "torch_export":
        owner_info = module_info.get(owner_path)
        if (
            owner_info
            and owner_info.get("quant_target")
            and owner_info.get("status")
            in {STATUS_EXACT, STATUS_TRANSPARENT_SUBCLASS, STATUS_CUSTOM_EXPANDED}
        ):
            quant_target = str(owner_info["quant_target"])
            record = _make_op_record(
                node,
                owner_path,
                str(owner_info["status"]),
                (
                    f"Input-specialized operation belongs to {owner_path or '<root>'}, "
                    f"which converts with {quant_target}."
                ),
                [quant_target, "fp32"],
                quant_target,
                shape,
            )
            record["quantized_target"] = quant_target
            return record

        if _is_export_structural_target(node.target):
            return _make_op_record(
                node,
                owner_path,
                STATUS_PASSTHROUGH,
                "Shape/index/no-op operation is preserved without quantization.",
                ["passthrough"],
                "passthrough",
                shape,
            )

    if _is_structural_node(node, non_tensor_nodes or set()):
        return _make_op_record(
            node,
            owner_path,
            STATUS_PASSTHROUGH,
            "Shape guard/control operation is preserved without quantization.",
            ["passthrough"],
            "passthrough",
            shape,
        )

    quant_target = _functional_quant_target(node.target, node.op)
    if quant_target and _registry_contains(quant_target):
        label = _target_label(node.target)
        record = _make_op_record(
            node,
            owner_path,
            STATUS_FUNCTIONAL,
            f"Functional operation {label} has QBench replacement {quant_target}.",
            [quant_target, "fp32"],
            quant_target,
            shape,
        )
        record["quantized_target"] = quant_target
        return record

    if _is_structural_target(node.target, node.op):
        return _make_op_record(
            node,
            owner_path,
            STATUS_PASSTHROUGH,
            "Shape/index/control operation is preserved without quantization.",
            ["passthrough"],
            "passthrough",
            shape,
        )

    owner_info = module_info.get(owner_path)
    specialized_target = owner_info.get("quant_target") if owner_info else None
    if specialized_target in {
        "QuantBatchNormAct2d",
        "DecomposedQkvAttention",
        "DecomposedMlpBlock",
    }:
        record = _make_op_record(
            node,
            owner_path,
            STATUS_CUSTOM_EXPANDED,
            (
                f"Operation {_target_label(node.target)} is handled by the owning "
                f"module's specialized converter {specialized_target}."
            ),
            [str(specialized_target), "fp32"],
            str(specialized_target),
            shape,
        )
        record["quantized_target"] = str(specialized_target)
        return record

    label = _target_label(node.target)
    return _make_op_record(
        node,
        owner_path,
        STATUS_UNSUPPORTED,
        f"No registered QBench rewrite for functional operation {label}; keep it in FP32.",
        ["fp32"],
        "fp32",
        shape,
    )


def _make_op_record(
    node: Node,
    owner_path: str,
    status: str,
    reason: str,
    candidates: list[str],
    recommended: str,
    shape: list[int | str] | None,
) -> dict[str, Any]:
    return {
        "id": "",
        "label": _target_label(node.target),
        "op": node.op,
        "target": _qualified_target_name(node.target),
        "module_path": owner_path,
        "module_type": _node_owner_type(node),
        "status": status,
        "reason": reason,
        "parent": None,
        "candidates": candidates,
        "recommended": recommended,
        "kind": "operation",
        "shape": shape,
        "quantized_target": None,
    }


def _build_target_preview(
    source_nodes: Sequence[Mapping[str, Any]],
    source_edges: Sequence[Mapping[str, Any]],
    module_info: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    target_nodes: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    target_id_by_source: dict[str, str] = {}

    for source_node in source_nodes:
        source_id = str(source_node["id"])
        target_id = f"target:{source_id}"
        target_id_by_source[source_id] = target_id
        target_node = copy.deepcopy(dict(source_node))
        target_node["id"] = target_id
        parent = source_node.get("parent")
        target_node["parent"] = f"target:{parent}" if parent else None

        recommended = str(source_node.get("recommended", "fp32"))
        quantized_target = source_node.get("quantized_target")
        status = str(source_node.get("status", STATUS_FP32))
        if quantized_target or (
            recommended not in {"fp32", "passthrough", "expand"}
            and not recommended.startswith("alias:")
        ):
            target_name = str(quantized_target or recommended)
            target_node["label"] = target_name
            target_node["target"] = target_name
            target_node["module_type"] = target_name
            target_node["status"] = "proposed_quantized"
            mapping_kind = "functional_rewrite" if status == STATUS_FUNCTIONAL else "one_to_one"
        elif recommended.startswith("alias:"):
            target_name = recommended.split(":", 1)[1].rsplit(".", 1)[-1]
            target_node["label"] = target_name
            target_node["target"] = recommended.split(":", 1)[1]
            target_node["module_type"] = target_name
            target_node["status"] = "native_alias"
            mapping_kind = "native_alias"
        elif status in (STATUS_FP32, STATUS_UNSUPPORTED) or recommended == "fp32":
            target_node["label"] = f"FP32 · {source_node.get('label', '')}"
            target_node["status"] = STATUS_FP32
            mapping_kind = "fp32_fallback"
        elif status == STATUS_CUSTOM_EXPANDED or recommended == "expand":
            target_node["label"] = f"Expanded · {source_node.get('label', '')}"
            target_node["status"] = STATUS_CUSTOM_EXPANDED
            mapping_kind = "decomposed"
        else:
            target_node["status"] = STATUS_PASSTHROUGH
            mapping_kind = "passthrough"

        target_nodes.append(target_node)
        mappings.append(
            {
                "source_node_ids": [source_id],
                "target_node_ids": [target_id],
                "kind": mapping_kind,
                "reason": str(source_node.get("reason", "")),
            }
        )

    # Add explicit group-level one-to-many mappings for expanded custom modules.
    for path, info in module_info.items():
        if not path or info.get("status") != STATUS_CUSTOM_EXPANDED:
            continue
        source_group = _module_node_id(path)
        child_targets = [
            target_id_by_source[str(node["id"])]
            for node in source_nodes
            if node.get("kind") == "operation" and node.get("module_path") == path
        ]
        if child_targets:
            mappings.append(
                {
                    "source_node_ids": [source_group],
                    "target_node_ids": child_targets,
                    "kind": "decomposed",
                    "reason": "The custom module's forward maps to multiple independently converted operations.",
                }
            )

    target_edges = [
        {
            **dict(edge),
            "source": target_id_by_source.get(str(edge["source"]), f"target:{edge['source']}"),
            "target": target_id_by_source.get(str(edge["target"]), f"target:{edge['target']}"),
        }
        for edge in source_edges
    ]
    return {"nodes": target_nodes, "edges": target_edges}, mappings


def _path_is_at_or_below(candidate: str, root: str) -> bool:
    return candidate == root or candidate.startswith(root + ".")


def _preview_source_subtree_ids(
    source_nodes: Sequence[Mapping[str, Any]], path: str
) -> set[str]:
    module_id = _module_node_id(path)
    return {
        str(node.get("id"))
        for node in source_nodes
        if str(node.get("id")) == module_id
        or _path_is_at_or_below(str(node.get("module_path", "")), path)
    }


def _preview_target_subtree_ids(
    target_nodes: Sequence[Mapping[str, Any]], path: str
) -> set[str]:
    target_module_id = f"target:{_module_node_id(path)}"
    return {
        str(node.get("id"))
        for node in target_nodes
        if str(node.get("id")) == target_module_id
        or _path_is_at_or_below(str(node.get("module_path", "")), path)
    }


def _rewire_collapsed_preview_edges(
    edges: Sequence[Mapping[str, Any]],
    removed_ids: set[str],
    alias_target_id: str,
) -> list[dict[str, Any]]:
    rewired: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_edge in edges:
        edge = copy.deepcopy(dict(raw_edge))
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in removed_ids:
            source = alias_target_id
        if target in removed_ids:
            target = alias_target_id
        if not source or not target or source == target:
            continue
        edge["source"] = source
        edge["target"] = target
        key = (source, target, str(edge.get("kind", "dataflow")))
        if key in seen:
            continue
        seen.add(key)
        rewired.append(edge)
    return rewired


def _resolve_preview_quant_choice(
    choice: str, row: Mapping[str, Any]
) -> str | None:
    if choice.lower().startswith(("quant", "decomposed")):
        return choice
    if choice != "quantize":
        return None
    candidates = [str(value) for value in row.get("candidates", []) or []]
    recommended = str(row.get("recommended", ""))
    for candidate in [recommended, *candidates]:
        if candidate.lower().startswith(("quant", "decomposed")):
            return candidate
    return None


def _update_module_mapping_kind(
    mappings: list[dict[str, Any]],
    path: str,
    kind: str,
    target_id: str,
) -> None:
    source_id = _module_node_id(path)
    for mapping in mappings:
        source_ids = [str(value) for value in mapping.get("source_node_ids", [])]
        target_ids = [str(value) for value in mapping.get("target_node_ids", [])]
        if source_ids == [source_id] and target_ids == [target_id]:
            mapping["kind"] = kind
            return
    mappings.append(
        {
            "source_node_ids": [source_id],
            "target_node_ids": [target_id],
            "kind": kind,
            "reason": f"Resolved plan decision for {path!r}.",
        }
    )


def _sanitize_preview_graph_and_mappings(
    source_nodes: Sequence[Mapping[str, Any]],
    target_graph: MutableMapping[str, list[dict[str, Any]]],
    mappings: list[dict[str, Any]],
) -> None:
    """Remove dangling IDs after collapses while preserving stable node IDs."""

    unique_nodes: list[dict[str, Any]] = []
    target_ids: set[str] = set()
    for raw_node in target_graph.get("nodes", []):
        node = raw_node if isinstance(raw_node, dict) else dict(raw_node)
        node_id = str(node.get("id", ""))
        if not node_id or node_id in target_ids:
            continue
        target_ids.add(node_id)
        unique_nodes.append(node)
    for node in unique_nodes:
        parent = node.get("parent")
        if parent is not None and str(parent) not in target_ids:
            node.pop("parent", None)
    target_graph["nodes"] = unique_nodes

    valid_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for raw_edge in target_graph.get("edges", []):
        edge = raw_edge if isinstance(raw_edge, dict) else dict(raw_edge)
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        key = (source, target, str(edge.get("kind", "dataflow")))
        if (
            source not in target_ids
            or target not in target_ids
            or source == target
            or key in seen_edges
        ):
            continue
        seen_edges.add(key)
        valid_edges.append(edge)
    target_graph["edges"] = valid_edges

    source_ids = {str(node.get("id")) for node in source_nodes}
    valid_mappings: list[dict[str, Any]] = []
    for raw_mapping in mappings:
        mapping = raw_mapping if isinstance(raw_mapping, dict) else dict(raw_mapping)
        mapped_sources = list(
            dict.fromkeys(
                str(value)
                for value in mapping.get("source_node_ids", [])
                if str(value) in source_ids
            )
        )
        mapped_targets = list(
            dict.fromkeys(
                str(value)
                for value in mapping.get("target_node_ids", [])
                if str(value) in target_ids
            )
        )
        if not mapped_sources or not mapped_targets:
            continue
        mapping["source_node_ids"] = mapped_sources
        mapping["target_node_ids"] = mapped_targets
        valid_mappings.append(mapping)
    mappings[:] = valid_mappings


def _functional_quant_target(target: Any, node_op: str) -> str | None:
    # GenericAdapter currently rewrites call_function nodes only.  Be strict
    # here: advertising method/conv/torch.softmax mappings would make the
    # preview claim conversions the implementation cannot produce.
    if node_op != "call_function":
        return None

    identities: list[tuple[Any, str]] = [
        (F.linear, "QuantLinear"),
        (F.relu, "QuantReLU"),
        (torch.relu, "QuantReLU"),
        (F.relu6, "QuantReLU6"),
        (F.gelu, "QuantGELU"),
        (F.silu, "QuantSiLU"),
        (F.hardswish, "QuantHardswish"),
        (F.hardsigmoid, "QuantHardsigmoid"),
        (F.softmax, "QuantSoftmax"),
        (torch.matmul, "QuantMatMul"),
        (operator.matmul, "QuantMatMul"),
        (torch.bmm, "QuantBMM"),
        (torch.add, "QuantAdd"),
        (operator.add, "QuantAdd"),
        (torch.sub, "QuantSub"),
        (operator.sub, "QuantSub"),
        (torch.mul, "QuantMul"),
        (operator.mul, "QuantMul"),
        (torch.div, "QuantDiv"),
        (operator.truediv, "QuantDiv"),
        (torch.cat, "QuantCat"),
    ]
    for candidate, quant_name in identities:
        if target is candidate:
            return quant_name
    return None


def _is_export_structural_target(target: Any) -> bool:
    name = str(target)
    if name == "<built-in function getitem>":
        return True
    structural_prefixes = (
        "aten.reshape.",
        "aten.view.",
        "aten.flatten.",
        "aten.transpose.",
        "aten.permute.",
        "aten.unbind.",
        "aten.squeeze.",
        "aten.unsqueeze.",
        "aten.clone.",
        "aten.detach.",
        "aten.contiguous.",
        "aten.dropout.",
    )
    return name.startswith(structural_prefixes)


def _registry_contains(op_name: str) -> bool:
    try:
        OpRegistry.get(op_name)
        return True
    except ValueError:
        return False


def _is_passthrough_module(module: nn.Module) -> bool:
    types: tuple[type[nn.Module], ...] = (
        nn.Identity,
        nn.Flatten,
        nn.Unflatten,
        nn.Sequential,
        nn.ModuleList,
        nn.ModuleDict,
    )
    return isinstance(module, types)


def _is_structural_node(node: Node, non_tensor_nodes: set[str]) -> bool:
    """Return whether an FX node is shape/control bookkeeping, not tensor math.

    A target name alone is not enough here: ``operator.eq`` and
    ``operator.floordiv`` can be real tensor operations.  They are structural
    only when non-tensor provenance reaches them, or when a comparison exists
    solely to feed a runtime assertion.
    """

    if node.name in non_tensor_nodes:
        return True
    if node.op != "call_function":
        return False

    target_name = _target_label(node.target).lower().strip("_")
    if target_name in {"assert", "check"}:
        return True

    comparison_targets = {
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.is_,
        operator.is_not,
    }
    if node.target not in comparison_targets:
        return False

    input_nodes = list(_iter_fx_nodes((node.args, node.kwargs)))
    if input_nodes and all(input_node.name in non_tensor_nodes for input_node in input_nodes):
        return True

    users = list(node.users)
    return bool(users) and all(
        user.op == "call_function"
        and _target_label(user.target).lower().strip("_") in {"assert", "check"}
        for user in users
    )


def _is_structural_target(target: Any, node_op: str) -> bool:
    name = str(target) if node_op == "call_method" else getattr(target, "__name__", "")
    name = name.lower().strip("_")
    return name in {
        "getitem",
        "getattr",
        "len",
        "size",
        "shape",
        "dim",
        "numel",
        "view",
        "reshape",
        "flatten",
        "unflatten",
        "permute",
        "transpose",
        "contiguous",
        "unsqueeze",
        "squeeze",
        "expand",
        "expand_as",
        "chunk",
        "split",
        "unbind",
        "clone",
        "detach",
        "to",
        "type_as",
    }


def _node_owner_path(node: Node) -> str:
    explicit = node.meta.get("qbench_owner_path")
    if explicit is not None:
        return str(explicit)
    stack = node.meta.get("nn_module_stack")
    if isinstance(stack, Mapping) and stack:
        last = next(reversed(stack.values()))
        if isinstance(last, tuple) and last:
            return str(last[0])
    if node.op == "call_module":
        return str(node.target)
    return ""


def _node_owner_type(node: Node) -> str:
    explicit = node.meta.get("qbench_owner_type")
    if explicit:
        return str(explicit).rsplit(".", 1)[-1]
    return ""


def _node_tensor_shape(node: Node) -> list[int | str] | None:
    tensor_meta = node.meta.get("tensor_meta")
    shape = getattr(tensor_meta, "shape", None)
    if shape is None:
        exported_value = node.meta.get("val", node.meta.get("example_value"))
        shape = getattr(exported_value, "shape", None)
    if shape is None:
        return None
    return [int(dim) if isinstance(dim, int) else str(dim) for dim in shape]


def _target_label(target: Any) -> str:
    if isinstance(target, str):
        return target
    return getattr(target, "__name__", str(target))


def _qualified_target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", "")
    name = getattr(target, "__qualname__", getattr(target, "__name__", str(target)))
    return f"{module}.{name}" if module else name


def _qualified_type_name(module_type: type[Any]) -> str:
    return f"{module_type.__module__}.{module_type.__qualname__}"


def _module_node_id(path: str) -> str:
    return f"module:{path}" if path else "module:$root"


def _iter_fx_nodes(value: Any) -> Iterator[Node]:
    if isinstance(value, Node):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_fx_nodes(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_fx_nodes(child)


def _export_model_graph(model: nn.Module, sample_input: Any) -> tuple[GraphModule, str]:
    """Capture the branch actually executed by ``sample_input`` with torch.export.

    Export is an analysis fallback, not the converted model.  Keeping those
    roles separate preserves the eager model's state_dict/API while still
    giving the workbench an accurate graph for input-dependent architectures.
    """

    if isinstance(sample_input, Mapping):
        args: tuple[Any, ...] = ()
        kwargs = dict(sample_input)
    elif isinstance(sample_input, tuple):
        args = sample_input
        kwargs = {}
    else:
        args = (sample_input,)
        kwargs = {}

    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    strict_error: Exception | None = None
    try:
        try:
            exported = torch.export.export(model, args=args, kwargs=kwargs, strict=True)
            mode = "strict"
        except Exception as exc:
            strict_error = exc
            exported = torch.export.export(model, args=args, kwargs=kwargs, strict=False)
            mode = f"non-strict (strict failed: {type(strict_error).__name__})"
        graph_module = exported.module()
        return graph_module, mode
    finally:
        for module, was_training in training_states:
            module.training = was_training


def _propagate_shapes(graph_module: GraphModule, sample_input: Any) -> None:
    from torch.fx.passes.shape_prop import ShapeProp

    training_states = [(module, module.training) for module in graph_module.modules()]
    graph_module.eval()
    try:
        # ShapeProp prints an internal interpreter traceback to stderr before
        # re-raising.  The caller turns that exception into a concise capture
        # diagnostic, so keep the server log readable here.
        with torch.no_grad(), contextlib.redirect_stderr(io.StringIO()):
            if isinstance(sample_input, Mapping):
                # ShapeProp support for kwargs varies by torch release; bind
                # them to forward's declared parameter order when possible.
                signature = inspect.signature(graph_module.forward)
                bound = signature.bind_partial(**sample_input)
                ShapeProp(graph_module).propagate(*bound.args, **bound.kwargs)
            elif isinstance(sample_input, tuple):
                ShapeProp(graph_module).propagate(*sample_input)
            else:
                ShapeProp(graph_module).propagate(sample_input)
    finally:
        # Assign directly so unusual per-child train/eval states are retained.
        for module, was_training in training_states:
            module.training = was_training


def _normalize_choice(value: Any) -> str:
    if isinstance(value, Mapping):
        action = value.get("choice", value.get("action", value.get("replacement", value.get("recommended"))))
        if action == "alias" and value.get("target"):
            return f"alias:{value['target']}"
        value = action
    if value is None:
        return "fp32"
    choice = str(value).strip()
    lowered = choice.lower()
    if lowered in {"skip", "keep", "fallback", "keep_fp32", "fp32 fallback"}:
        return "fp32"
    if lowered in {"decompose", "expanded", "custom_expanded"}:
        return "expand"
    if lowered in {"structural", "pass-through", "pass_through"}:
        return "passthrough"
    if lowered.startswith("native:"):
        return "alias:" + choice.split(":", 1)[1]
    return choice


def _choice_matches_candidates(choice: str, candidates: Sequence[Any]) -> bool:
    normalized_candidates = {_normalize_choice(candidate) for candidate in candidates}
    if choice in normalized_candidates:
        return True
    # A registered Quant* candidate and the generic "quantize" action mean the
    # same thing at module level.
    if choice == "quantize" and any(c.lower().startswith(("quant", "decomposed")) for c in normalized_candidates):
        return True
    return False


def _resolve_native_alias(alias_name: str, module: nn.Module) -> type[nn.Module]:
    token = alias_name.strip()
    supported = OpRegistry.get_supported_ops()
    matches = []
    for native_cls in supported:
        names = {
            native_cls.__name__,
            native_cls.__qualname__,
            _qualified_type_name(native_cls),
            f"nn.{native_cls.__name__}",
            f"torch.nn.{native_cls.__name__}",
        }
        if token in names or token.lower() in {name.lower() for name in names}:
            matches.append(native_cls)
    if not matches:
        raise ValueError(f"{alias_name!r} is not a registered native QBench operation")
    native_cls = matches[0]
    if not isinstance(module, native_cls):
        raise TypeError(
            f"{type(module).__name__} is not a subclass/instance of {native_cls.__name__}"
        )
    return native_cls


def _set_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child_name, replacement)


def _minimal_module_paths(paths: Iterable[str]) -> list[str]:
    """Drop descendants when an ancestor already defines the whole island."""

    result: list[str] = []
    for path in sorted({str(path) for path in paths if path}, key=lambda value: (value.count("."), value)):
        if any(path == ancestor or path.startswith(ancestor + ".") for ancestor in result):
            continue
        result.append(path)
    return result


def _materialize_functional_linears(graph_module: GraphModule) -> int:
    """Turn parameter-backed F.linear nodes into native Linear submodules."""

    graph = graph_module.graph
    rewritten = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or _functional_quant_target(node.target, node.op) != "QuantLinear":
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], Node) or node.args[1].op != "get_attr":
            continue
        try:
            weight = _fetch_attr(graph_module, str(node.args[1].target))
            bias = None
            if len(node.args) >= 3 and isinstance(node.args[2], Node) and node.args[2].op == "get_attr":
                bias = _fetch_attr(graph_module, str(node.args[2].target))
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            name = _unique_module_name(graph_module, f"_qbench_{node.name}_linear")
            linear = nn.Linear(
                int(weight.shape[1]),
                int(weight.shape[0]),
                bias=bias is not None,
                device=weight.device,
                dtype=weight.dtype,
            )
            with torch.no_grad():
                linear.weight.copy_(weight)
                if bias is not None and linear.bias is not None:
                    linear.bias.copy_(bias)
            graph_module.add_module(name, linear)
            with graph.inserting_before(node):
                replacement = graph.call_module(name, args=(node.args[0],), kwargs={})
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            rewritten += 1
        except Exception:
            continue
    if rewritten:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return rewritten


def _fetch_attr(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _unique_module_name(module: nn.Module, prefix: str) -> str:
    name = prefix.replace(".", "_")
    index = 1
    while hasattr(module, name):
        name = f"{prefix}_{index}".replace(".", "_")
        index += 1
    return name


def _is_qbench_module(module: nn.Module) -> bool:
    module_type = type(module).__name__
    return module_type.startswith(("Quant", "Decomposed", "Observed"))


def _quantized_module_inventory(model: nn.Module) -> dict[str, Any]:
    entries = [
        (path, type(module).__name__)
        for path, module in model.named_modules()
        if path and _is_qbench_module(module)
    ]
    return {
        "total": len(entries),
        "by_type": dict(Counter(module_type for _path, module_type in entries)),
        "paths": [path for path, _module_type in entries],
    }


def _invoke_model(model: nn.Module, sample_input: Any) -> Any:
    if isinstance(sample_input, Mapping):
        return model(**sample_input)
    if isinstance(sample_input, tuple):
        return model(*sample_input)
    return model(sample_input)


def _validate_eager_sample(model: nn.Module, sample_input: Any) -> None:
    """Verify that a trace failure is not actually an invalid model input."""

    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            _invoke_model(model, sample_input)
    finally:
        # Restore each flag directly: calling ``train`` on a parent recursively
        # overwrites the intentionally mixed train/eval state of its children.
        for module, was_training in training_states:
            module.training = was_training


def _compare_nested_outputs(
    reference: Any,
    quantized: Any,
    *,
    path: str,
    leaves: list[dict[str, Any]],
    atol: float,
    rtol: float,
) -> bool:
    if isinstance(reference, torch.Tensor) or isinstance(quantized, torch.Tensor):
        if not isinstance(reference, torch.Tensor) or not isinstance(quantized, torch.Tensor):
            leaves.append(
                {
                    "path": path,
                    "kind": "type_mismatch",
                    "reference_type": type(reference).__name__,
                    "quantized_type": type(quantized).__name__,
                }
            )
            return False
        shape_match = tuple(reference.shape) == tuple(quantized.shape)
        record: dict[str, Any] = {
            "path": path,
            "kind": "tensor",
            "reference_shape": list(reference.shape),
            "quantized_shape": list(quantized.shape),
            "reference_dtype": str(reference.dtype),
            "quantized_dtype": str(quantized.dtype),
            "shape_match": shape_match,
            "numel": int(reference.numel()),
        }
        if shape_match:
            ref_cpu = reference.detach().cpu()
            quant_cpu = quantized.detach().cpu()
            if ref_cpu.is_floating_point() or ref_cpu.is_complex():
                ref_numeric = ref_cpu.to(torch.float64)
                quant_numeric = quant_cpu.to(torch.float64)
                error = (ref_numeric - quant_numeric).abs()
                record.update(
                    max_abs_error=float(error.max().item()) if error.numel() else 0.0,
                    mean_abs_error=float(error.mean().item()) if error.numel() else 0.0,
                    allclose=bool(torch.allclose(ref_numeric, quant_numeric, atol=atol, rtol=rtol)),
                )
            else:
                equal = bool(torch.equal(ref_cpu, quant_cpu))
                record.update(max_abs_error=0.0 if equal else math.inf, mean_abs_error=0.0 if equal else math.inf, allclose=equal)
        else:
            record.update(max_abs_error=math.inf, mean_abs_error=math.inf, allclose=False)
        leaves.append(record)
        return shape_match

    if isinstance(reference, Mapping) or isinstance(quantized, Mapping):
        if not isinstance(reference, Mapping) or not isinstance(quantized, Mapping):
            leaves.append({"path": path, "kind": "type_mismatch"})
            return False
        ref_keys = list(reference.keys())
        quant_keys = list(quantized.keys())
        structure_match = set(ref_keys) == set(quant_keys)
        for key in ref_keys:
            if key in quantized:
                structure_match &= _compare_nested_outputs(
                    reference[key], quantized[key], path=f"{path}.{key}", leaves=leaves, atol=atol, rtol=rtol
                )
        if not structure_match:
            leaves.append(
                {
                    "path": path,
                    "kind": "mapping_keys",
                    "reference_keys": [str(key) for key in ref_keys],
                    "quantized_keys": [str(key) for key in quant_keys],
                    "equal": False,
                }
            )
        return bool(structure_match)

    if isinstance(reference, (tuple, list)) or isinstance(quantized, (tuple, list)):
        if type(reference) is not type(quantized) or len(reference) != len(quantized):
            leaves.append(
                {
                    "path": path,
                    "kind": "sequence_structure",
                    "reference_type": type(reference).__name__,
                    "quantized_type": type(quantized).__name__,
                    "reference_length": len(reference) if isinstance(reference, (tuple, list)) else None,
                    "quantized_length": len(quantized) if isinstance(quantized, (tuple, list)) else None,
                    "equal": False,
                }
            )
            return False
        return all(
            _compare_nested_outputs(
                ref_item,
                quant_item,
                path=f"{path}[{index}]",
                leaves=leaves,
                atol=atol,
                rtol=rtol,
            )
            for index, (ref_item, quant_item) in enumerate(zip(reference, quantized))
        )

    equal = reference == quantized
    if isinstance(equal, torch.Tensor):
        equal = bool(equal.all().item())
    leaves.append(
        {
            "path": path,
            "kind": "value",
            "reference": _json_safe(reference),
            "quantized": _json_safe(quantized),
            "equal": bool(equal),
        }
    )
    return type(reference) is type(quantized)


def _output_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        summary: dict[str, Any] = {
            "type": "tensor",
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
        }
        if detached.numel() and (detached.is_floating_point() or detached.is_complex()):
            numeric = detached.real.float()
            summary.update(
                min=float(numeric.min().item()),
                max=float(numeric.max().item()),
                mean=float(numeric.mean().item()),
            )
        return summary
    if isinstance(value, Mapping):
        return {str(key): _output_summary(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_output_summary(child) for child in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_output_summary(child) for child in value]}
    return {"type": type(value).__name__, "value": _json_safe(value)}


def _brief_exception(exc: Exception, limit: int = 360) -> str:
    message = " ".join(str(exc).strip().splitlines()[:1]).strip()
    rendered = f"{type(exc).__name__}: {message or 'no additional detail'}"
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(child) for child in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, type):
        return _qualified_type_name(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = [
    "WorkbenchAnalysis",
    "ConversionPlan",
    "ConversionResult",
    "ModelWorkbenchError",
    "WORKBENCH_ANALYSIS_SCHEMA_VERSION",
    "WORKBENCH_DATASET_BENCHMARK_API_VERSION",
    "WORKBENCH_REPLACEMENT_API_VERSION",
    "list_model_names",
    "load_model",
    "analyze_model",
    "build_conversion_plan",
    "list_replacement_targets",
    "inspect_replacement_target",
    "validate_replacement_spec",
    "preview_conversion_plan",
    "convert_model",
    "run_sample_inference",
    "build_classification_validation_loader",
    "benchmark_classification_models",
]
