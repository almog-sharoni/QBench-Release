from torch.utils.data import DataLoader
from src.datasets.dataset_registry import register_dataset


@register_dataset('wikitext2')
def build_wikitext2_data_loader(dataset_cfg: dict) -> DataLoader:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
    print(f"Loaded {len(dataset)} samples from wikitext-2.")
    batch_size = int(dataset_cfg['batch_size'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
