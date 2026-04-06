from .base import TaskMetricsBase
from .classification import ClassificationMetrics
from .language_model import LanguageModelMetrics

_TASK_METRICS_REGISTRY: dict = {
    "classification": ClassificationMetrics,
    "language_model": LanguageModelMetrics,
}

_ADAPTER_TYPE_TO_TASK: dict = {
    "generic": "classification",
    "slm": "language_model",
}


def create_task_metrics(adapter_type: str) -> TaskMetricsBase:
    task = _ADAPTER_TYPE_TO_TASK.get(adapter_type)
    if task is None:
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. Cannot determine task metrics. "
            f"Known adapter types: {list(_ADAPTER_TYPE_TO_TASK)}"
        )
    return _TASK_METRICS_REGISTRY[task]()
