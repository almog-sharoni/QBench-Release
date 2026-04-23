import os
import json
import torch
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
from torch.utils.data import DataLoader

from src.datasets.dataset_registry import register_dataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


class _ImageNetTargetTransform:
    def __init__(self, idx_map):
        self.idx_map = idx_map or {}

    def __call__(self, target):
        return self.idx_map.get(target, target)


def _build_imagenet_loader(dataset_cfg: dict) -> DataLoader:
    path = dataset_cfg['path']
    if not os.path.isabs(path):
        data_dir = os.path.join(_PROJECT_ROOT, path)
    else:
        data_dir = path

    image_size = dataset_cfg.get('image_size', 224)
    resize_size = dataset_cfg.get('resize_size', 256)
    model_source = dataset_cfg.get('model_source', 'torchvision')
    model_name = dataset_cfg.get('model_name', '')

    if model_source == 'timm':
        try:
            import timm
            _tmp = timm.create_model(model_name, pretrained=False)
            data_cfg = timm.data.resolve_model_data_config(_tmp)
            del _tmp
            transform = timm.data.create_transform(**data_cfg, is_training=False)
            print(f"Using timm transforms for {model_name}: "
                  f"input_size={data_cfg.get('input_size')}, "
                  f"mean={data_cfg.get('mean')}, std={data_cfg.get('std')}")
        except Exception as e:
            print(f"Warning: Could not resolve timm transforms for {model_name}: {e}. "
                  "Using standard ImageNet transforms.")
            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    dataset = tv_datasets.ImageFolder(data_dir, transform=transform)

    index_path = os.path.join(_PROJECT_ROOT, 'tests/data/imagenet_class_index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            class_index = json.load(f)
        wnid_to_idx = {v[0]: int(k) for k, v in class_index.items()}
        local_class_to_idx = dataset.class_to_idx
        idx_map = {}
        for wnid, local_idx in local_class_to_idx.items():
            if wnid in wnid_to_idx:
                idx_map[local_idx] = wnid_to_idx[wnid]
        mapped = len(idx_map)
        total_classes = len(local_class_to_idx)
        print(f"Label mapping: {mapped}/{total_classes} classes matched to ImageNet canonical indices.")
        if mapped == 0:
            print("Warning: No classes matched. Check that val folder names are WordNet IDs (e.g. n01440764).")
        elif mapped < total_classes:
            print(f"Warning: {total_classes - mapped} classes could not be mapped — using local indices.")
        dataset.target_transform = _ImageNetTargetTransform(idx_map)
    else:
        print(f"Warning: imagenet_class_index.json not found at {index_path}. "
              "Using ImageFolder's default alphabetical label ordering.")

    batch_size = int(dataset_cfg['batch_size'])
    num_workers = int(dataset_cfg.get('num_workers', 0))

    worker_mp_ctx = dataset_cfg.get('multiprocessing_context')
    if worker_mp_ctx is None:
        env_ctx = os.environ.get("QBENCH_DATALOADER_CONTEXT")
        if env_ctx:
            worker_mp_ctx = env_ctx.strip().lower()
    if worker_mp_ctx in ("", "none", "default"):
        worker_mp_ctx = None

    persistent_workers = bool(dataset_cfg.get('persistent_workers', num_workers > 0)) and num_workers > 0
    pin_memory = torch.cuda.is_available()

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        if worker_mp_ctx:
            loader_kwargs['multiprocessing_context'] = worker_mp_ctx
        prefetch_factor = dataset_cfg.get('prefetch_factor')
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = int(prefetch_factor)

    return DataLoader(dataset, **loader_kwargs)


@register_dataset('imagenet')
def build_imagenet_data_loader(dataset_cfg: dict) -> DataLoader:
    return _build_imagenet_loader(dataset_cfg)


@register_dataset('imagenette')
def build_imagenette_data_loader(dataset_cfg: dict) -> DataLoader:
    return _build_imagenet_loader(dataset_cfg)
