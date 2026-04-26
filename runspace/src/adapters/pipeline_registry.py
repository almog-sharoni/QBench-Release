from typing import Callable, Iterable, Optional, Type
import torch.nn as nn

# Registry value: (loader_fn, metrics_cls, components_map, required_input_keys)
# components_map: dict[logical_name -> module_path_prefix]
# required_input_keys: tuple of top-level dict keys the pipeline's forward needs
_REGISTRY: dict[str, tuple[Callable[[dict], nn.Module], Optional[Type], dict[str, str], tuple[str, ...]]] = {}


def register_pipeline(name: str, metrics_cls: Optional[Type] = None,
                      components: Optional[dict[str, str]] = None,
                      required_input_keys: Optional[Iterable[str]] = None) -> Callable:
    """
    Decorator to register a feature-matching pipeline loader.

    metrics_cls: zero-argument factory (class or callable) that returns a metrics
                 accumulator. Must be callable with no arguments — wrap parameterized
                 classes in a lambda at registration. If None, FeatureMatchingAdapter
                 falls back to FeatureMatchingMetrics.
    components:  dict mapping logical component names to module path prefixes,
                 e.g. {'superpoint': 'backbone.superpoint', 'superglue': 'backbone.superglue'}.
                 Used to resolve quantize_components config values.
    required_input_keys: top-level dict keys the pipeline's forward expects (e.g.
                 ('image',) for SuperPoint, ('image0', 'image1') for SuperPoint+SuperGlue).
                 Used by the runner to pre-check dataset compatibility before building models.
    """
    def decorator(fn: Callable[[dict], nn.Module]) -> Callable:
        _REGISTRY[name] = (fn, metrics_cls, components or {}, tuple(required_input_keys or ()))
        return fn
    return decorator


def load_pipeline(name: str, model_cfg: dict) -> nn.Module:
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Pipeline '{name}' not registered. Available: {available}")
    loader, _, _, _ = _REGISTRY[name]
    return loader(model_cfg)


def get_pipeline_metrics_cls(name: str) -> Optional[Type]:
    """
    Returns the registered metrics factory for a pipeline, or None if the
    pipeline registered no custom metrics class.

    Raises KeyError if the pipeline itself is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Pipeline '{name}' not registered. Available: {list(_REGISTRY.keys())}"
        )
    _, metrics_cls, _, _ = _REGISTRY[name]
    return metrics_cls


def get_pipeline_components(name: str) -> dict[str, str]:
    """
    Returns the pipeline's declared logical-component → module-prefix map.
    Empty dict means the pipeline declares no components.

    Raises KeyError if the pipeline itself is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Pipeline '{name}' not registered. Available: {list(_REGISTRY.keys())}"
        )
    return dict(_REGISTRY[name][2])


def get_pipeline_required_keys(name: str) -> tuple[str, ...]:
    """
    Returns the tuple of top-level input-dict keys the pipeline's forward
    expects. Empty tuple means the pipeline did not declare any (no pre-check).

    Raises KeyError if the pipeline itself is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Pipeline '{name}' not registered. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name][3]


def resolve_component_prefixes(pipeline_name: str,
                                component_names: list[str]) -> list[str]:
    """
    Translate logical component names to module path prefixes.

    A component value may be either a single prefix string or a list of prefix
    strings (used for fine-grained components that span multiple submodule
    paths, e.g. 'superglue_backbone' = kenc + gnn + final_proj). List values
    are flattened into the returned list.

    Raises KeyError if a name is not declared for this pipeline.
    """
    if pipeline_name not in _REGISTRY:
        raise KeyError(f"Pipeline '{pipeline_name}' not registered.")
    _, _, components, _ = _REGISTRY[pipeline_name]
    prefixes: list[str] = []
    for comp in component_names:
        if comp not in components:
            raise KeyError(
                f"Component '{comp}' not declared for pipeline '{pipeline_name}'. "
                f"Declared components: {list(components)}"
            )
        value = components[comp]
        if isinstance(value, (list, tuple)):
            prefixes.extend(value)
        else:
            prefixes.append(value)
    return prefixes


def list_pipelines() -> list[str]:
    return list(_REGISTRY.keys())
