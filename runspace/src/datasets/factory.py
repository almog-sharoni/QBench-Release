from typing import Any, Callable, Dict

from .classification import build_classification_data_loader


DatasetBuilder = Callable[[Dict[str, Any], str], Any]

DATASET_BUILDERS: Dict[str, DatasetBuilder] = {
    "classification": build_classification_data_loader,
}


def build_data_loader(config: Dict[str, Any], project_root: str):
    dataset_config = config.get("dataset", {})
    dataset_type = str(dataset_config.get("type", "")).strip().lower()

    if not dataset_type:
        raise ValueError(
            "Missing required dataset.type in config['dataset']. "
            "Example: dataset: {type: classification, ...}"
        )

    if dataset_type not in DATASET_BUILDERS:
        available = ", ".join(sorted(DATASET_BUILDERS.keys()))
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. Available dataset types: {available}"
        )

    return DATASET_BUILDERS[dataset_type](dataset_config, project_root)
