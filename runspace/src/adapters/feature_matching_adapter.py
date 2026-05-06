import copy
import torch
import torch.nn as nn
from .generic_adapter import GenericAdapter
from .pipeline_registry import load_pipeline, resolve_component_prefixes
from src.quantization.constants import DEFAULT_QUANTIZATION_TYPE
import src.adapters.pipelines  # triggers all @register_pipeline decorators  # noqa: F401


class FeatureMatchingPipeline(nn.Module):
    """
    Thin nn.Module wrapper around any feature-matching backbone.
    Allows OpRegistry layer replacement to recurse into backbone.
    forward(batch: dict) -> dict
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch: dict) -> dict:
        return self.backbone(batch)


class FeatureMatchingAdapter(GenericAdapter):
    """
    Adapter for feature-matching models (SuperPoint, SuperPoint+SuperGlue, etc.).
    Loads the backbone via PipelineRegistry (keyed by model name).
    All quantization infrastructure is inherited from GenericAdapter.

    quantize_components: optional list of logical component names declared by the
    pipeline (e.g. ['superpoint', 'superglue']). Resolved to module path prefixes
    via the pipeline registry. If omitted, all eligible ops are quantized.
    """

    def __init__(
        self,
        pipeline_name: str,
        model_cfg: dict,
        quantization_type: str = DEFAULT_QUANTIZATION_TYPE,
        quantized_ops: list = None,
        excluded_ops: list = None,
        quantize_first_layer: bool = False,
        weight_quantization: bool = True,
        input_quantization: bool = False,
        output_quantization: bool = False,
        layer_config: dict = None,
        quant_mode: str = 'tensor',
        chunk_size: int = None,
        weight_mode: str = 'channel',
        weight_chunk_size: int = None,
        output_quantization_type: str = None,
        output_mode: str = 'tensor',
        output_chunk_size: int = None,
        rounding: str = 'nearest',
        run_id: str = 'default',
        skip_calibration: bool = False,
        build_quantized: bool = True,
        quantize_components: list = None,
        strict_format_check: bool = False,
    ):
        self._pipeline_name = pipeline_name
        self._model_cfg = model_cfg
        self._reference_model = None

        target_prefixes = []
        if quantize_components:
            target_prefixes = resolve_component_prefixes(pipeline_name, quantize_components)

        super().__init__(
            model_name=pipeline_name,
            model_source='custom',
            weights=None,
            quantization_type=quantization_type,
            quantized_ops=quantized_ops if quantized_ops is not None else ['Conv2d'],
            excluded_ops=excluded_ops if excluded_ops is not None else [],
            quantize_first_layer=quantize_first_layer,
            weight_quantization=weight_quantization,
            input_quantization=input_quantization,
            output_quantization=output_quantization,
            layer_config=layer_config,
            quant_mode=quant_mode,
            chunk_size=chunk_size,
            weight_mode=weight_mode,
            weight_chunk_size=weight_chunk_size,
            output_quantization_type=output_quantization_type,
            output_mode=output_mode,
            output_chunk_size=output_chunk_size,
            rounding=rounding,
            run_id=run_id,
            skip_calibration=skip_calibration,
            build_quantized=build_quantized,
            target_module_prefixes=target_prefixes,
            strict_format_check=strict_format_check,
        )

    def build_model(self, quantized: bool = False) -> nn.Module:
        backbone = load_pipeline(self._pipeline_name, self._model_cfg)
        model = FeatureMatchingPipeline(backbone)

        if quantized:
            self._replace_layers(model)
            if torch.cuda.is_available():
                model.to(torch.device('cuda'))
            if not self.skip_calibration:
                self._calibrate_model(model)

        return model

    def create_metrics(self):
        from src.adapters.pipeline_registry import get_pipeline_metrics_cls
        metrics_cls = get_pipeline_metrics_cls(self._pipeline_name)
        if metrics_cls is not None:
            return metrics_cls()
        from src.eval.metrics import FeatureMatchingMetrics
        return FeatureMatchingMetrics()

    def prepare_batch(self, batch: dict) -> tuple:
        inputs = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)
                  and k not in ('T_0to1', 'K0', 'K1')}
        return inputs, batch

    def forward(self, model: nn.Module, batch: tuple) -> dict:
        inputs, _ = batch
        return model(inputs)

    def build_reference_model(self) -> nn.Module:
        if self._reference_model is None:
            self._reference_model = self.build_model(quantized=False)
        return copy.deepcopy(self._reference_model)
