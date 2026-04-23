import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from src.datasets.dataset_registry import register_dataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


class ImageDirectoryDataset(Dataset):
    """
    Loads all images from a directory as grayscale tensors.
    Each item is a dict: {'image': Tensor[1, H, W], 'image_path': str}
    """

    def __init__(self, root: str, image_size: tuple[int, int], grayscale: bool):
        self.root = root
        self.grayscale = grayscale
        self.files = sorted([
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if os.path.splitext(fname)[1].lower() in _EXTENSIONS
        ])
        if not self.files:
            raise FileNotFoundError(f"No images found in {root}")

        ops = []
        ops.append(T.Resize(image_size))
        if grayscale:
            ops.append(T.Grayscale(num_output_channels=1))
        ops.append(T.ToTensor())
        self.transform = T.Compose(ops)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        tensor = self.transform(img)
        return {'image': tensor, 'image_path': path}


def _collate_fn(batch: list[dict]) -> dict:
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'image_path': [item['image_path'] for item in batch],
    }


@register_dataset('image_directory', provided_keys=('image', 'image_path'))
def build_feature_matching_data_loader(dataset_cfg: dict) -> DataLoader:
    raw_path = dataset_cfg['path']
    root = raw_path if os.path.isabs(raw_path) else os.path.join(_PROJECT_ROOT, raw_path)
    raw_size = dataset_cfg.get('image_size', [480, 640])
    image_size = tuple(raw_size) if isinstance(raw_size, list) else raw_size
    grayscale = bool(dataset_cfg.get('grayscale', True))
    batch_size = int(dataset_cfg.get('batch_size', 1))
    num_workers = int(dataset_cfg.get('num_workers', 0))

    dataset = ImageDirectoryDataset(root=root, image_size=image_size, grayscale=grayscale)
    persistent_workers = bool(dataset_cfg.get('persistent_workers', num_workers > 0)) and num_workers > 0
    pin_memory = torch.cuda.is_available()
    prefetch_factor = int(dataset_cfg.get('prefetch_factor', 4)) if num_workers > 0 else None
    loader_kwargs = dict(batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=_collate_fn,
                         pin_memory=pin_memory,
                         persistent_workers=persistent_workers)
    if prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


class ImageDirectoryPairsDataset(Dataset):
    """
    Loads consecutive image pairs (image[i], image[i+1]) from a flat directory.
    Each item is a dict: {'image0': Tensor[1,H,W], 'image1': Tensor[1,H,W]}.
    Suitable as a demo/test dataset for image-pair pipelines (e.g. SuperPoint+SuperGlue).
    """

    def __init__(self, root: str, image_size: tuple[int, int], grayscale: bool):
        self.root = root
        files = sorted([
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if os.path.splitext(fname)[1].lower() in _EXTENSIONS
        ])
        if len(files) < 2:
            raise FileNotFoundError(f"Need at least 2 images in {root}, found {len(files)}")
        self.pairs = list(zip(files, files[1:]))

        ops = [T.Resize(image_size)]
        if grayscale:
            ops.append(T.Grayscale(num_output_channels=1))
        ops.append(T.ToTensor())
        self.transform = T.Compose(ops)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p0, p1 = self.pairs[idx]
        img0 = self.transform(Image.open(p0).convert('RGB'))
        img1 = self.transform(Image.open(p1).convert('RGB'))
        return {'image0': img0, 'image1': img1,
                'image_path0': p0, 'image_path1': p1}


def _collate_pairs_fn(batch: list[dict]) -> dict:
    return {
        'image0': torch.stack([b['image0'] for b in batch]),
        'image1': torch.stack([b['image1'] for b in batch]),
        'image_path0': [b['image_path0'] for b in batch],
        'image_path1': [b['image_path1'] for b in batch],
    }


@register_dataset('image_directory_pairs',
                  provided_keys=('image0', 'image1', 'image_path0', 'image_path1'))
def build_image_directory_pairs_data_loader(dataset_cfg: dict) -> DataLoader:
    """
    Pairs consecutive images in a directory. Suitable for testing pair-based pipelines.
    dataset_cfg keys: path (required), image_size, grayscale, batch_size, num_workers
    """
    raw_path = dataset_cfg['path']
    root = raw_path if os.path.isabs(raw_path) else os.path.join(_PROJECT_ROOT, raw_path)
    raw_size = dataset_cfg.get('image_size', [480, 640])
    image_size = tuple(raw_size) if isinstance(raw_size, list) else raw_size
    grayscale = bool(dataset_cfg.get('grayscale', True))
    batch_size = int(dataset_cfg.get('batch_size', 1))
    num_workers = int(dataset_cfg.get('num_workers', 0))

    dataset = ImageDirectoryPairsDataset(root=root, image_size=image_size, grayscale=grayscale)
    persistent_workers = bool(dataset_cfg.get('persistent_workers', num_workers > 0)) and num_workers > 0
    pin_memory = torch.cuda.is_available()
    prefetch_factor = int(dataset_cfg.get('prefetch_factor', 4)) if num_workers > 0 else None
    loader_kwargs = dict(batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=_collate_pairs_fn,
                         pin_memory=pin_memory,
                         persistent_workers=persistent_workers)
    if prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
