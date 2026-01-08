import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry.op_registry import OpRegistry
from .quant_base import QuantizedLayerMixin
from ..quantization.quantizer import quantize

@OpRegistry.register("QuantLayerNorm2d", original_cls=nn.LayerNorm2d, under_construction=True)
def quant_ln2d():
    return nn.LayerNorm2d
