import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from runspace.src.registry.op_registry import OpRegistry
from runspace.src.quantization.quantizer import quantize
from runspace.src.ops.quant_base import QuantizedLayerMixin, quantize_tensor

@OpRegistry.register("QuantBatchNorm2d", original_cls=nn.BatchNorm2d, compliance_status="match leo ng")
class QuantBatchNorm2d(nn.BatchNorm2d, QuantizedLayerMixin):
    q_type: str
    weight_scale: torch.Tensor | None
    weight_fp8: torch.Tensor | None

    """
    Quantized BatchNorm2d layer that simulates FP8 quantization.
    Quantizes input and weight (gamma). Bias, running_mean, running_var remain FP32.
    """
    def __init__(self, *args, q_type="fp8_e4m3", **kwargs):
        super().__init__(*args, **kwargs)
        self.q_type = q_type
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_fp8', None)

    # calibrate_weights is provided by QuantizedLayerMixin

    def forward(self, input):
        # Check for FP8 support
        if not hasattr(torch, 'float8_e4m3fn'):
             raise RuntimeError("FP8 support (torch.float8_e4m3fn) is required but not available.")

        # Use shared quantization logic
        # input_fp8 = self.quantize_input(input)
        input_fp8 = input
    
        # Dequantize weights (gamma)
        if self.weight_fp8 is not None and self.weight_scale is not None:
            w_decomp = self.weight_fp8.float() * self.weight_scale
            if getattr(self, 'capture_activations', False):
                self.last_quant_weight = w_decomp.detach()
        else:
            w_decomp = self.weight

        # Run BatchNorm
        # Quantize running stats if they exist
        rm_quant = self.running_mean
        rv_quant = self.running_var
        
        if self.running_mean is not None:
             # Use weight_mode for stats? Or just tensor?
             # running_mean is 1D [C]. quantize_tensor 'channel' mode on 1D falls back to 'tensor'.
             # So we use weight_mode to support 'chunk' if set, otherwise it will be 'tensor' (scalar).
             mode = getattr(self, 'weight_mode', 'tensor')
             chunk_size = getattr(self, 'weight_chunk_size', None)
             
             # rm_quant, rm_unscaled, _ = quantize_tensor(self.running_mean, q_type=self.q_type, bias=self.quantization_bias, mode=mode, chunk_size=chunk_size, return_unscaled=True)
             
             if getattr(self, 'capture_activations', False):
                 # if self.q_type == 'fp8_e4m3' and hasattr(torch, 'float8_e4m3fn'):
                 #     self.last_quant_rm = rm_unscaled.to(torch.float8_e4m3fn)
                 # elif self.q_type == 'fp8_e5m2' and hasattr(torch, 'float8_e5m2'):
                 #     self.last_quant_rm = rm_unscaled.to(torch.float8_e5m2)
                 # else:
                 #     self.last_quant_rm = rm_unscaled
                 self.last_quant_rm = self.running_mean
             
        if self.running_var is not None:
             mode = getattr(self, 'weight_mode', 'tensor')
             chunk_size = getattr(self, 'weight_chunk_size', None)
             # rv_quant, rv_unscaled, _ = quantize_tensor(self.running_var, q_type=self.q_type, bias=self.quantization_bias, mode=mode, chunk_size=chunk_size, return_unscaled=True)

             if getattr(self, 'capture_activations', False):
                 # if self.q_type == 'fp8_e4m3' and hasattr(torch, 'float8_e4m3fn'):
                 #     self.last_quant_rv = rv_unscaled.to(torch.float8_e4m3fn)
                 # elif self.q_type == 'fp8_e5m2' and hasattr(torch, 'float8_e5m2'):
                 #     self.last_quant_rv = rv_unscaled.to(torch.float8_e5m2)
                 # else:
                 #     self.last_quant_rv = rv_unscaled
                 self.last_quant_rv = self.running_var

        return F.batch_norm(
            input_fp8, 
            rm_quant, 
            rv_quant, 
            w_decomp, 
            self.bias, 
            self.training, 
            self.momentum, 
            self.eps
        )


@OpRegistry.register("QuantBatchNorm1d", original_cls=nn.BatchNorm1d)
class QuantBatchNorm1d(nn.BatchNorm1d, QuantizedLayerMixin):
    q_type: str
    weight_scale: torch.Tensor | None
    weight_fp8: torch.Tensor | None

    """
    Quantized BatchNorm1d: quantizes input and gamma (weight); bias / running stats stay FP32.
    """
    def __init__(self, *args, q_type="fp8_e4m3", **kwargs):
        super().__init__(*args, **kwargs)
        self.q_type = q_type
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_fp8', None)

    def forward(self, input):
        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError("FP8 support (torch.float8_e4m3fn) is required but not available.")

        input_fp8 = self.quantize_input(input)

        if self.weight_fp8 is not None and self.weight_scale is not None:
            w_decomp = self.weight_fp8.float() * self.weight_scale
            if getattr(self, 'capture_activations', False):
                self.last_quant_weight = w_decomp.detach()
        else:
            w_decomp = self.weight

        if getattr(self, 'capture_activations', False):
            if self.running_mean is not None:
                self.last_quant_rm = self.running_mean
            if self.running_var is not None:
                self.last_quant_rv = self.running_var

        return F.batch_norm(
            input_fp8,
            self.running_mean,
            self.running_var,
            w_decomp,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


@OpRegistry.register("QuantBatchNormAct2d")
class QuantBatchNormAct2d(nn.Module, QuantizedLayerMixin):
    q_type: str
    quantization_bias: int | None
    bn: QuantBatchNorm2d
    drop: nn.Module
    act: nn.Module

    """
    Quantized wrapper for timm-style BatchNormAct2d blocks.
    Preserves drop/activation behavior while quantizing the internal BN op.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        drop: nn.Module = None,
        act: nn.Module = None,
        q_type: str = "fp8_e4m3",
        quantization_bias: int = None,
    ):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias

        self.bn = QuantBatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            q_type=q_type,
        )
        self.drop = copy.deepcopy(drop) if drop is not None else nn.Identity()
        self.act = copy.deepcopy(act) if act is not None else nn.Identity()

    def _sync_runtime_config(self):
        # Mirror adapter/layer runtime knobs to the internal quant BN.
        for attr in (
            "q_type",
            "quantization_bias",
            "input_q_type",
            "input_quantization",
            "weight_quantization",
            "input_mode",
            "input_chunk_size",
            "weight_mode",
            "weight_chunk_size",
            "rounding",
            "chunk_formats",
            "run_id",
            "layer_name",
            "capture_activations",
        ):
            if hasattr(self, attr):
                setattr(self.bn, attr, getattr(self, attr))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._sync_runtime_config()
        x = self.bn(input)
        x = self.drop(x)
        x = self.act(x)
        return x

    @classmethod
    def from_native(cls, native_bn_act, q_type="fp8_e4m3", quantization_bias: int = None):
        wrapped = cls(
            num_features=native_bn_act.num_features,
            eps=native_bn_act.eps,
            momentum=native_bn_act.momentum,
            affine=native_bn_act.affine,
            track_running_stats=native_bn_act.track_running_stats,
            drop=getattr(native_bn_act, "drop", None),
            act=getattr(native_bn_act, "act", None),
            q_type=q_type,
            quantization_bias=quantization_bias,
        )

        if native_bn_act.affine:
            wrapped.bn.weight.data.copy_(native_bn_act.weight.data)
            wrapped.bn.bias.data.copy_(native_bn_act.bias.data)

        if native_bn_act.track_running_stats:
            wrapped.bn.running_mean.data.copy_(native_bn_act.running_mean.data)
            wrapped.bn.running_var.data.copy_(native_bn_act.running_var.data)
            wrapped.bn.num_batches_tracked.data.copy_(native_bn_act.num_batches_tracked.data)

        return wrapped
