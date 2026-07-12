from abc import ABC, abstractmethod
import torch.nn as nn


_ACTIVATION_BOUNDARY_FORMAT_ATTRS = {
    "input_quantization": ("input_q_type", "q_type"),
    "output_quantization": ("output_q_type", "q_type"),
}

class BaseAdapter(ABC):
    """
    Abstract base class for model adapters.
    Adapters are responsible for:
    - Building FP and quantized versions of a model
    - Preparing batches
    - Providing FP/Q forward functions
    - Exposing layer names for layer insertion
    """

    @staticmethod
    def _enabled_module_activation_boundaries(model: nn.Module):
        """Return enabled, non-FP32 module fake-quant boundaries.

        Hardware activation transport temporarily disables these flags before
        executing the model. Keeping the check based on the live flags lets the
        same model fail closed when called directly and run normally once a
        stage transport runtime has been installed.
        """
        enabled = []
        for module_name, module in model.named_modules():
            for flag, format_attrs in _ACTIVATION_BOUNDARY_FORMAT_ATTRS.items():
                if not bool(getattr(module, flag, False)):
                    continue
                q_type = None
                for attr in format_attrs:
                    value = getattr(module, attr, None)
                    if value is not None:
                        q_type = str(value).strip().lower()
                        break
                if q_type == "fp32":
                    continue
                enabled.append((module_name or "<root>", flag, q_type or "unknown"))
        return enabled

    def _install_activation_transport_guard(self, model: nn.Module):
        """Reject direct execution through legacy module activation fake quant."""
        existing = getattr(
            model,
            "_qbench_activation_transport_guard_handle",
            None,
        )
        if existing is not None:
            return existing
        configured = self._enabled_module_activation_boundaries(model)

        def require_stage_transport(guarded_model, _inputs):
            active = self._enabled_module_activation_boundaries(guarded_model)
            if not active:
                return
            preview = ", ".join(
                f"{name}.{flag} ({q_type})"
                for name, flag, q_type in active[:5]
            )
            if len(active) > 5:
                preview += f", ... (+{len(active) - 5} more)"
            raise RuntimeError(
                "Legacy module-level activation fake quantization cannot execute. "
                "Activation quantization must use producer-stage hardware transport "
                "via DynamicInputQuantizer, UniformInputQuantizer, or "
                "InputOnlyActivationQuantizer. Enabled legacy boundaries: "
                f"{preview}. For a weight-only or FP32 structural run, set "
                "adapter.input_quantization=false and "
                "adapter.output_quantization=false."
            )

        model._qbench_activation_transport_guarded = True
        model._qbench_legacy_activation_boundaries = tuple(configured)
        handle = model.register_forward_pre_hook(require_stage_transport)
        model._qbench_activation_transport_guard_handle = handle
        return handle

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

    @abstractmethod
    def create_metrics(self):
        """Returns a metrics accumulator appropriate for this adapter's task."""
        pass
