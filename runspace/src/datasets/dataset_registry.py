from typing import Callable
from torch.utils.data import DataLoader

_REGISTRY: dict[str, Callable[[dict], DataLoader]] = {}


def register_dataset(name: str) -> Callable:
    def decorator(fn: Callable[[dict], DataLoader]) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def build_data_loader(name: str, dataset_cfg: dict) -> DataLoader:
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Dataset '{name}' not registered. Available: {available}")
    return _REGISTRY[name](dataset_cfg)


def list_datasets() -> list[str]:
    return list(_REGISTRY.keys())
