from typing import Callable, Iterable, Optional
from torch.utils.data import DataLoader

# Registry value: (loader_fn, provided_keys)
# provided_keys: tuple of top-level keys the dataset's collated batch dict contains.
# Empty tuple means the dataset did not declare (no pre-check performed).
_REGISTRY: dict[str, tuple[Callable[[dict], DataLoader], tuple[str, ...]]] = {}


def register_dataset(name: str, provided_keys: Optional[Iterable[str]] = None) -> Callable:
    """
    Register a dataset builder. `provided_keys` should list every top-level key
    the dataset's collate function emits (e.g. ('image0', 'image1', 'K0') for
    scannet_pairs). The runner uses this to pre-check pipeline/dataset compatibility.
    """
    def decorator(fn: Callable[[dict], DataLoader]) -> Callable:
        _REGISTRY[name] = (fn, tuple(provided_keys or ()))
        return fn
    return decorator


def build_data_loader(name: str, dataset_cfg: dict) -> DataLoader:
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Dataset '{name}' not registered. Available: {available}")
    return _REGISTRY[name][0](dataset_cfg)


def get_dataset_provided_keys(name: str) -> tuple[str, ...]:
    """
    Returns the tuple of top-level keys the dataset's collate fn emits.
    Empty tuple means the dataset didn't declare (skip pre-check).

    Raises KeyError if the dataset is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Dataset '{name}' not registered. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name][1]


def list_datasets() -> list[str]:
    return list(_REGISTRY.keys())
