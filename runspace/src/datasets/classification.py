import json
import os
from typing import Any, Dict, Optional

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def _resolve_dataset_dir(path: str, project_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def _build_torchvision_transform(model_name: str, dataset_config: Dict[str, Any]):
    from torchvision.models import get_model_weights

    weights = get_model_weights(model_name).DEFAULT
    transform = weights.transforms()

    # Allow dataset_config overrides (e.g. crop_size, resize_size, mean, std).
    for key, value in dataset_config.items():
        if hasattr(transform, key):
            setattr(transform, key, value)

    print(
        f"[datasets] Using torchvision transforms for '{model_name}': "
        f"crop_size={getattr(transform, 'crop_size', '?')}, "
        f"mean={getattr(transform, 'mean', '?')}, std={getattr(transform, 'std', '?')}"
    )
    return transform


def _build_timm_transform(model_name: str, dataset_config: Dict[str, Any]):
    import timm

    tmp_model = timm.create_model(model_name, pretrained=False)
    data_cfg = timm.data.resolve_model_data_config(tmp_model)
    del tmp_model

    # Allow dataset_config overrides on top of timm's model defaults.
    data_cfg.update({k: v for k, v in dataset_config.items() if k in data_cfg})

    transform = timm.data.create_transform(**data_cfg, is_training=False)
    print(
        f"[datasets] Using timm transforms for '{model_name}': "
        f"input_size={data_cfg.get('input_size')}, "
        f"mean={data_cfg.get('mean')}, std={data_cfg.get('std')}"
    )
    return transform


def _detect_source(model_name: str) -> str:
    import timm as _timm
    return "timm" if _timm.is_model(model_name) else "torchvision"


# Each entry: source → {"builder": callable(model_name, dataset_config), "defaults": dict}
# Defaults are merged with dataset_config (dataset_config takes precedence).
# Register new sources here without touching _resolve_transform.
_SOURCE_REGISTRY: Dict[str, Any] = {
    "timm": {
        "builder": _build_timm_transform,
        "defaults": {},  # timm derives its own defaults from the model
    },
    "torchvision": {
        "builder": _build_torchvision_transform,
        "defaults": {},  # torchvision derives its own defaults from the model weights
    },
}


def _resolve_transform(config: Dict[str, Any]):
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})

    model_name = model_config.get("name", "")
    model_source = model_config.get("source", "auto")

    if model_source == "auto":
        model_source = _detect_source(model_name)

    entry = _SOURCE_REGISTRY.get(model_source)
    if entry is None:
        raise ValueError(
            f"No transform builder registered for model source '{model_source}'. "
            f"Known sources: {list(_SOURCE_REGISTRY)}"
        )

    merged_config = {**entry["defaults"], **dataset_config}
    return entry["builder"](model_name, merged_config)


def _apply_class_index_mapping(dataset, index_path: str) -> None:
    if not os.path.exists(index_path):
        return

    with open(index_path, "r") as f:
        class_index = json.load(f)

    wnid_to_idx = {v[0]: int(k) for k, v in class_index.items()}
    local_class_to_idx = dataset.class_to_idx
    idx_map = {}
    for wnid, local_idx in local_class_to_idx.items():
        if wnid in wnid_to_idx:
            idx_map[local_idx] = wnid_to_idx[wnid]

    def target_transform(target):
        return idx_map.get(target, target)

    dataset.target_transform = target_transform


def build_classification_data_loader(
    config: Dict[str, Any], project_root: str
) -> Optional[DataLoader]:
    dataset_config = config.get("dataset", {})

    path = dataset_config.get("path")
    if not path:
        raise ValueError("Missing required dataset.path for dataset.type='classification'.")

    data_dir = _resolve_dataset_dir(path, project_root)
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist. Skipping data loading.")
        return None

    transform = _resolve_transform(config)
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    class_index_path = dataset_config.get("class_index_path")
    if class_index_path:
        resolved_index_path = _resolve_dataset_dir(class_index_path, project_root)
        _apply_class_index_mapping(dataset, resolved_index_path)

    batch_size = dataset_config.get("batch_size", 32)
    num_workers = dataset_config.get("num_workers", 0)

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
