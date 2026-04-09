import torch
from abc import ABC, abstractmethod


class TaskMetricsBase(ABC):
    @abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate statistics from one batch."""
        ...

    @abstractmethod
    def compute(self) -> dict:
        """Return final metrics dict. Keys are task-specific."""
        ...

    @abstractmethod
    def metric_labels(self) -> dict[str, str]:
        """Map each metric key produced by compute() to its human-readable label."""
        ...

    @abstractmethod
    def percentage_keys(self) -> set[str]:
        """Return the subset of metric keys whose values should be displayed as a percentage."""
        ...

    @staticmethod
    @abstractmethod
    def compute_certainty(predictions: torch.Tensor) -> float:
        """Return mean certainty score for a single batch of predictions."""
        ...
