import torch
from torch.utils.data import DataLoader, Dataset
from src.datasets.dataset_registry import register_dataset


@register_dataset('wikitext2')
def build_wikitext2_data_loader(dataset_cfg: dict) -> DataLoader:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
    print(f"Loaded {len(dataset)} samples from wikitext-2.")
    batch_size = int(dataset_cfg['batch_size'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class _TokenBlockDataset(Dataset):
    """Wraps a [num_blocks, seq_len] LongTensor of token ids as an LM dataset."""

    def __init__(self, blocks: torch.Tensor):
        self.blocks = blocks

    def __len__(self):
        return self.blocks.size(0)

    def __getitem__(self, idx):
        ids = self.blocks[idx]
        # labels == input_ids; the MetricsEngine shifts internally for
        # next-token cross-entropy / perplexity.
        return {"input_ids": ids, "labels": ids}


@register_dataset('wikitext2_lm', provided_keys=('input_ids',))
def build_wikitext2_lm_data_loader(dataset_cfg: dict) -> DataLoader:
    """
    Causal-LM perplexity loader for wikitext-2 (raw, test split).

    Concatenates the test corpus, tokenizes it with the model's tokenizer
    (resolved from ``dataset_cfg['model_name']``), and chunks it into contiguous
    fixed-length blocks of ``seq_len`` token ids. Each block is one evaluation
    example; perplexity is computed over the whole corpus regardless of
    ``batch_size``.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model_name = dataset_cfg.get('model_name')
    if not model_name:
        raise ValueError(
            "wikitext2_lm requires the model name to load a matching tokenizer; "
            "it is normally injected by the Runner from config['model']['name']."
        )

    seq_len = int(dataset_cfg.get('seq_len', 512))
    batch_size = int(dataset_cfg.get('batch_size', 4))
    max_blocks = dataset_cfg.get('max_blocks')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"].squeeze(0)

    num_blocks = ids.size(0) // seq_len
    if num_blocks == 0:
        raise ValueError(
            f"wikitext2_lm: corpus has only {ids.size(0)} tokens, fewer than "
            f"seq_len={seq_len}. Lower dataset.seq_len."
        )
    blocks = ids[: num_blocks * seq_len].view(num_blocks, seq_len).contiguous()

    if max_blocks is not None:
        blocks = blocks[: int(max_blocks)]

    print(
        f"wikitext2_lm: {ids.size(0)} tokens -> {blocks.size(0)} blocks "
        f"of seq_len={seq_len} (tokenizer={model_name})."
    )

    return DataLoader(
        _TokenBlockDataset(blocks),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(dataset_cfg.get('num_workers', 0)),
    )
