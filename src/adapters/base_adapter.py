from abc import ABC, abstractmethod
import torch.nn as nn

class BaseAdapter(ABC):
    """
    Abstract base class for model adapters.
    Adapters are responsible for:
    - Building FP and quantized versions of a model
    - Preparing batches
    - Providing FP/Q forward functions
    - Exposing layer names for layer insertion
    """

    @abstractmethod
    def build_model(self, quantized: bool = False):
        """Builds and returns the model (FP or Quantized)."""
        pass

    @abstractmethod
    def prepare_batch(self, batch):
        """Prepares a batch of data for the model."""
        pass

    @abstractmethod
    def forward(self, model, batch):
        """Runs the forward pass."""
        pass

    @abstractmethod
    def get_layer_names(self, model) -> list[str]:
        """Returns a list of layer names where custom layers can be inserted."""
        pass
