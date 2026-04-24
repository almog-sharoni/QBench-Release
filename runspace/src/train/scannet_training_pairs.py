"""ScanNet training pairs dataset — extends scannet_pairs with depth maps.

Design C Phase 2 needs per-pixel depth + camera intrinsics + relative pose to
generate ground-truth correspondences on-the-fly at training time. The vanilla
`scannet_pairs` loader emits only pose+intrinsics; this module adds depth.

Depth files live alongside the color JPGs under /data/scannet/posed_images/, as
16-bit PNGs in millimeters (scale by 1/1000 to meters). For pairs files that
use `scans_test/<scene>/sens/frame-XXXXXX.color.jpg` (e.g. the paper test set),
depth is at `scans_test/<scene>/sens/frame-XXXXXX.depth.png`. For pairs files
that use `<scene>/<frame>.jpg` under /data/scannet/posed_images/, depth is at
`<scene>/<frame>.png` in the same directory.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.dataset_registry import register_dataset
from src.datasets.scannet_pairs import (
    _parse_pairs_file,
    _read_gray,
    _resolve_path,
    _scale_K,
)


def _depth_path_for(img_path: str) -> str:
    """Infer the depth PNG path from a color JPG path."""
    root, ext = os.path.splitext(img_path)
    if root.endswith(".color"):
        return root[:-len(".color")] + ".depth.png"
    return root + ".png"


def _read_depth(path: str, new_w: int, new_h: int) -> torch.Tensor:
    """Load 16-bit mm depth → float32 meters → nearest-neighbor resize."""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"could not read depth: {path}")
    if d.ndim != 2:
        raise ValueError(f"unexpected depth shape {d.shape} at {path}")
    d = cv2.resize(d, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(d.astype(np.float32) / 1000.0)


class ScanNetTrainingPairsDataset(Dataset):
    """Like ScanNetPairsDataset, but also yields depth0 / depth1 tensors."""

    def __init__(self, root: str, pairs_file: str,
                 image_size: tuple[int, int] = (480, 640),
                 max_pairs: int = -1):
        self.root = root
        self.image_size = image_size
        self.pairs = _parse_pairs_file(pairs_file)
        if max_pairs > 0:
            self.pairs = self.pairs[:max_pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        path0 = _resolve_path(p['img0'], self.root)
        path1 = _resolve_path(p['img1'], self.root)
        depth_path0 = _depth_path_for(path0)
        depth_path1 = _depth_path_for(path1)

        new_h, new_w = self.image_size
        img0, orig_w0, orig_h0 = _read_gray(path0, new_w, new_h)
        img1, orig_w1, orig_h1 = _read_gray(path1, new_w, new_h)
        depth0 = _read_depth(depth_path0, new_w, new_h)
        depth1 = _read_depth(depth_path1, new_w, new_h)

        K0 = _scale_K(p['K0'], orig_w0, orig_h0, new_w, new_h)
        K1 = _scale_K(p['K1'], orig_w1, orig_h1, new_w, new_h)

        return {
            'image0': img0,
            'image1': img1,
            'depth0': depth0,
            'depth1': depth1,
            'T_0to1': p['T_0to1'],
            'K0': K0,
            'K1': K1,
            'pair_id': f"{p['img0']}_{p['img1']}",
        }


def _collate_fn(batch: list[dict]) -> dict:
    return {
        'image0': torch.stack([b['image0'] for b in batch]),
        'image1': torch.stack([b['image1'] for b in batch]),
        'depth0': torch.stack([b['depth0'] for b in batch]),
        'depth1': torch.stack([b['depth1'] for b in batch]),
        'T_0to1': torch.stack([b['T_0to1'] for b in batch]),
        'K0': torch.stack([b['K0'] for b in batch]),
        'K1': torch.stack([b['K1'] for b in batch]),
        'pair_id': [b['pair_id'] for b in batch],
    }


@register_dataset('scannet_training_pairs',
                  provided_keys=('image0', 'image1', 'depth0', 'depth1',
                                 'T_0to1', 'K0', 'K1', 'pair_id'))
def build_scannet_training_pairs_data_loader(dataset_cfg: dict) -> DataLoader:
    root = dataset_cfg['path']
    pairs_file = dataset_cfg['pairs_file']
    raw_size = dataset_cfg.get('image_size', [480, 640])
    image_size = tuple(raw_size) if isinstance(raw_size, list) else raw_size
    max_pairs = int(dataset_cfg.get('max_pairs', -1))
    batch_size = int(dataset_cfg.get('batch_size', 1))
    num_workers = int(dataset_cfg.get('num_workers', 0))
    shuffle = bool(dataset_cfg.get('shuffle', True))

    dataset = ScanNetTrainingPairsDataset(root=root, pairs_file=pairs_file,
                                          image_size=image_size, max_pairs=max_pairs)
    persistent_workers = bool(dataset_cfg.get('persistent_workers', num_workers > 0)) and num_workers > 0
    pin_memory = torch.cuda.is_available()
    prefetch_factor = int(dataset_cfg.get('prefetch_factor', 4)) if num_workers > 0 else None
    loader_kwargs = dict(batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=_collate_fn,
                         pin_memory=pin_memory,
                         persistent_workers=persistent_workers)
    if prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
