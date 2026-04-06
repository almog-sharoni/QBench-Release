import json
import os
from typing import Any, Dict, Optional

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def _resolve_dataset_dir(path: str, project_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def _build_transform(dataset_config: Dict[str, Any]):
    image_size = dataset_config.get("image_size", 224)
    resize_size = dataset_config.get("resize_size", 256)

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


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
    dataset_config: Dict[str, Any], project_root: str
) -> Optional[DataLoader]:
    path = dataset_config.get("path")
    if not path:
        raise ValueError("Missing required dataset.path for dataset.type='classification'.")

    data_dir = _resolve_dataset_dir(path, project_root)
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist. Skipping data loading.")
        return None

    transform = _build_transform(dataset_config)
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
