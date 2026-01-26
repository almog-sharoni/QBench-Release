import torch
import threading
from ..quantization.quantizer import quantize_tf32

# Thread-local storage to track context state
_local_state = threading.local()

def _get_patch_state():
    if not hasattr(_local_state, 'enabled'):
        _local_state.enabled = False
    return _local_state.enabled

def _set_patch_state(enabled: bool):
    _local_state.enabled = enabled

# Original functions
_orig_torch_add = torch.add
_orig_tensor_add = torch.Tensor.add
_orig_tensor_add_ = torch.Tensor.add_
_orig_tensor_plus = torch.Tensor.__add__
_orig_tensor_iplus = torch.Tensor.__iadd__

def _should_quantize(result: torch.Tensor) -> bool:
    """
    Check if result should be quantized to TF32.
    Criteria:
    1. Patch enabled (checked by caller)
    2. 4D Tensor (heuristic for Conv activation: N, C, H, W)
    3. Spatial Width dimension (dim 3) > 128
    """
    if result.dim() != 4:
        return False
    
    # Check spatial width (dim 3)
    if result.shape[3] <= 128:
        return False
        
    return True

def _patched_torch_add(input, other, *args, **kwargs):
    result = _orig_torch_add(input, other, *args, **kwargs)
    if _get_patch_state() and _should_quantize(result):
        return quantize_tf32(result)
    return result

def _patched_tensor_add(self, other, *args, **kwargs):
    result = _orig_tensor_add(self, other, *args, **kwargs)
    if _get_patch_state() and _should_quantize(result):
        return quantize_tf32(result)
    return result



def _patched_tensor_add_(self, other, *args, **kwargs):
    result = _orig_tensor_add_(self, other, *args, **kwargs)
    if _get_patch_state() and _should_quantize(self):
        q_res = quantize_tf32(self)
        self.data.copy_(q_res)
    return result

def _patched_tensor_plus(self, other):
    result = _orig_tensor_plus(self, other)
    if _get_patch_state() and _should_quantize(result):
        return quantize_tf32(result)
    return result

def _patched_tensor_iplus(self, other):
    result = _orig_tensor_iplus(self, other)
    if _get_patch_state() and _should_quantize(self):
        q_res = quantize_tf32(self)
        self.data.copy_(q_res)
    return result


class TF32Patcher:
    """
    Context manager to patch torch.add and related methods to simulate TF32 accumulation.
    """
    def __init__(self):
        self.active = False

    def __enter__(self):
        _set_patch_state(True)
        self.active = True
        
        # Monkey patch
        torch.add = _patched_torch_add
        torch.Tensor.add = _patched_tensor_add
        torch.Tensor.add_ = _patched_tensor_add_
        torch.Tensor.__add__ = _patched_tensor_plus
        torch.Tensor.__iadd__ = _patched_tensor_iplus
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_patch_state(False)
        self.active = False
        
        # Restore
        torch.add = _orig_torch_add
        torch.Tensor.add = _orig_tensor_add
        torch.Tensor.add_ = _orig_tensor_add_
        torch.Tensor.__add__ = _orig_tensor_plus
        torch.Tensor.__iadd__ = _orig_tensor_iplus

