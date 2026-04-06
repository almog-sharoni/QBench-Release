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
