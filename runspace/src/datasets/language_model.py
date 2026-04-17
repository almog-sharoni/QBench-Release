from typing import Any, Dict

from torch.utils.data import DataLoader


def build_language_model_data_loader(
    config: Dict[str, Any], project_root: str
) -> DataLoader:
    import importlib

    load_dataset = importlib.import_module("datasets").load_dataset

    dataset_config = config.get("dataset", config)  # accept full config or bare dataset_config
    dataset_name = dataset_config.get("name")
    dataset_path = dataset_config.get("path")
    if not dataset_name or not dataset_path:
        raise ValueError(
            "Missing required dataset.name and dataset.path for dataset.type='language_model'. "
            "Example: name: wikitext, path: wikitext-2-raw-v1"
        )

    split = dataset_config.get("split", "test")
    batch_size = dataset_config.get("batch_size", 4)
    num_workers = dataset_config.get("num_workers", 0)

    dataset = load_dataset(dataset_name, dataset_path, split=split)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
