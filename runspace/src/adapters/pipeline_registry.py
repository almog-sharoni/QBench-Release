from typing import Callable, Optional, Type
import torch.nn as nn

# Registry value: (loader_fn, metrics_cls, components_map)
# components_map: dict[logical_name -> module_path_prefix]
_REGISTRY: dict[str, tuple[Callable[[dict], nn.Module], Optional[Type], dict[str, str]]] = {}


def register_pipeline(name: str, metrics_cls: Optional[Type] = None,
                      components: Optional[dict[str, str]] = None) -> Callable:
    """
    Decorator to register a feature-matching pipeline loader.

    metrics_cls: zero-argument factory (class or callable) that returns a metrics
                 accumulator. Must be callable with no arguments — wrap parameterized
                 classes in a lambda at registration. If None, FeatureMatchingAdapter
                 falls back to FeatureMatchingMetrics.
    components:  dict mapping logical component names to module path prefixes,
                 e.g. {'superpoint': 'backbone.superpoint', 'superglue': 'backbone.superglue'}.
                 Used to resolve quantize_components config values.
    """
    def decorator(fn: Callable[[dict], nn.Module]) -> Callable:
        _REGISTRY[name] = (fn, metrics_cls, components or {})
        return fn
    return decorator


def load_pipeline(name: str, model_cfg: dict) -> nn.Module:
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Pipeline '{name}' not registered. Available: {available}")
    loader, _, _ = _REGISTRY[name]
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
    _, metrics_cls, _ = _REGISTRY[name]
    return metrics_cls


def resolve_component_prefixes(pipeline_name: str,
                                component_names: list[str]) -> list[str]:
    """
    Translate logical component names to module path prefixes.
    Raises KeyError if a name is not declared for this pipeline.
    """
    if pipeline_name not in _REGISTRY:
        raise KeyError(f"Pipeline '{pipeline_name}' not registered.")
    _, _, components = _REGISTRY[pipeline_name]
    prefixes = []
    for comp in component_names:
        if comp not in components:
            raise KeyError(
                f"Component '{comp}' not declared for pipeline '{pipeline_name}'. "
                f"Declared components: {list(components)}"
            )
        prefixes.append(components[comp])
    return prefixes


def list_pipelines() -> list[str]:
    return list(_REGISTRY.keys())
