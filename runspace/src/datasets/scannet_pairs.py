import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from src.datasets.dataset_registry import register_dataset


def _resolve_path(path: str, root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _build_resize(size) -> T.Resize:
    if isinstance(size, (list, tuple)):
        return T.Resize(list(size))
    return T.Resize(size)


def _load_image(path: str, grayscale: bool, resize: T.Resize) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = resize(img)
    if grayscale:
        img = T.Grayscale(num_output_channels=1)(img)
    return T.ToTensor()(img)


def _scale_K(K: torch.Tensor, orig_w: int, orig_h: int,
             new_w: int, new_h: int) -> torch.Tensor:
    sx = new_w / orig_w
    sy = new_h / orig_h
    K = K.clone()
    K[0] *= sx
    K[1] *= sy
    return K


def _parse_pairs_file(pairs_file: str) -> list[dict]:
    """
    ScanNet GT pairs format (scannet_sample_pairs_with_gt.txt):
      img0 img1 rot0 rot1 K0(9 floats) K1(9 floats) T_0to1(16 floats)
    Total: 2 + 2 + 9 + 9 + 16 = 38 fields per line.
    rot0/rot1 are EXIF rotation integers (0-3) and are skipped.
    """
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 38:
                continue
            img0, img1 = parts[0], parts[1]
            nums = list(map(float, parts[2:]))
            # nums[0:2] = rot0, rot1 (skip)
            K0 = torch.tensor(nums[2:11]).reshape(3, 3)
            K1 = torch.tensor(nums[11:20]).reshape(3, 3)
            T_0to1 = torch.tensor(nums[20:36]).reshape(4, 4)
            pairs.append({'img0': img0, 'img1': img1,
                          'K0': K0, 'K1': K1, 'T_0to1': T_0to1})
    return pairs


class ScanNetPairsDataset(Dataset):
    """
    Loads overlapping image pairs from a ScanNet-style pairs file.
    Each item: dict {
        'image0': Tensor[1, H, W],
        'image1': Tensor[1, H, W],
        'T_0to1': Tensor[4, 4],
        'K0': Tensor[3, 3],
        'K1': Tensor[3, 3],
        'pair_id': str,
    }
    """

    def __init__(self, root: str, pairs_file: str,
                 image_size: tuple[int, int] = (480, 640),
                 max_pairs: int = -1):
        self.root = root
        self.image_size = image_size
        self.resize = _build_resize(image_size)
        self.pairs = _parse_pairs_file(pairs_file)
        if max_pairs > 0:
            self.pairs = self.pairs[:max_pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        path0 = _resolve_path(p['img0'], self.root)
        path1 = _resolve_path(p['img1'], self.root)

        orig0 = Image.open(path0)
        orig_w0, orig_h0 = orig0.size
        orig1 = Image.open(path1)
        orig_w1, orig_h1 = orig1.size

        new_h, new_w = self.image_size
        img0 = self.resize(orig0.convert('RGB'))
        img0 = T.Grayscale(num_output_channels=1)(img0)
        img0 = T.ToTensor()(img0)
        img1 = self.resize(orig1.convert('RGB'))
        img1 = T.Grayscale(num_output_channels=1)(img1)
        img1 = T.ToTensor()(img1)

        K0 = _scale_K(p['K0'], orig_w0, orig_h0, new_w, new_h)
        K1 = _scale_K(p['K1'], orig_w1, orig_h1, new_w, new_h)

        return {
            'image0': img0,
            'image1': img1,
            'T_0to1': p['T_0to1'],
            'K0': K0,
            'K1': K1,
            'pair_id': f"{p['img0']}_{p['img1']}",
        }


def _collate_fn(batch: list[dict]) -> dict:
    return {
        'image0': torch.stack([b['image0'] for b in batch]),
        'image1': torch.stack([b['image1'] for b in batch]),
        'T_0to1': torch.stack([b['T_0to1'] for b in batch]),
        'K0': torch.stack([b['K0'] for b in batch]),
        'K1': torch.stack([b['K1'] for b in batch]),
        'pair_id': [b['pair_id'] for b in batch],
    }


@register_dataset('scannet_pairs')
def build_scannet_pairs_data_loader(dataset_cfg: dict) -> DataLoader:
    root = dataset_cfg['path']
    pairs_file = dataset_cfg['pairs_file']
    raw_size = dataset_cfg.get('image_size', [480, 640])
    image_size = tuple(raw_size) if isinstance(raw_size, list) else raw_size
    max_pairs = int(dataset_cfg.get('max_pairs', -1))
    batch_size = int(dataset_cfg.get('batch_size', 1))
    num_workers = int(dataset_cfg.get('num_workers', 0))

    dataset = ScanNetPairsDataset(root=root, pairs_file=pairs_file,
                                  image_size=image_size, max_pairs=max_pairs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=_collate_fn)
