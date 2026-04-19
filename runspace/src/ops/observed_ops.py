import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry.op_registry import OpRegistry
from .quant_base import QuantizedLayerMixin


def _simple_nms(scores, nms_radius: int):
    """Fast NMS via max_pool2d (mirrors superpoint.simple_nms)."""
    assert nms_radius >= 0

    def max_pool(x):
        return F.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def _remove_borders(keypoints, scores, border: int, height: int, width: int):
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def _top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


# ---------------------------------------------------------------------------
# Passthrough / linear ops — like ReLU, value representation unchanged
# ---------------------------------------------------------------------------

@OpRegistry.register("ObservedDiscardTrash", passthrough=True)
class ObservedDiscardTrash(nn.Module):
    """Drop the dustbin (last) score channel (stageB3)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :-1]


@OpRegistry.register("ObservedReorderReshape", passthrough=True)
class ObservedReorderReshape(nn.Module):
    """Pixel-shuffle reorder/reshape (stageB4-B7): (B,64,H/8,W/8) -> (B,H,W)."""
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        return scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)


@OpRegistry.register("ObservedThreshold", compliance_status="FP32 activation")
class ObservedThreshold(nn.Module):
    """Threshold score map and extract keypoint coordinates (stageB10-B12)."""
    def __init__(self, threshold: float = 0.005):
        super().__init__()
        self.threshold = threshold

    def forward(self, scores: torch.Tensor):
        keypoints = [torch.nonzero(s > self.threshold) for s in scores]
        score_list = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
        return keypoints, score_list


@OpRegistry.register("ObservedRemoveBorders", passthrough=True)
class ObservedRemoveBorders(nn.Module):
    """Remove keypoints too close to image borders (stageB9)."""
    def __init__(self, border: int = 4):
        super().__init__()
        self.border = border

    def forward(self, keypoints, scores, h_full: int, w_full: int):
        result = [_remove_borders(k, s, self.border, h_full, w_full)
                  for k, s in zip(keypoints, scores)]
        return list(zip(*result))


@OpRegistry.register("ObservedTopKKeypoints", passthrough=True)
class ObservedTopKKeypoints(nn.Module):
    """Keep only the top-k highest-scoring keypoints (stageB13)."""
    def __init__(self, max_keypoints: int = -1):
        super().__init__()
        self.max_keypoints = max_keypoints

    def forward(self, keypoints, scores):
        if self.max_keypoints < 0:
            return list(keypoints), list(scores)
        result = [_top_k_keypoints(k, s, self.max_keypoints)
                  for k, s in zip(keypoints, scores)]
        return list(zip(*result))


@OpRegistry.register("ObservedKeypointFlip", passthrough=True)
class ObservedKeypointFlip(nn.Module):
    """Convert keypoint coords from (row, col) to (x, y) order (stageB14)."""
    def forward(self, keypoints):
        return [torch.flip(k, [1]).float() for k in keypoints]


@OpRegistry.register("ObservedL2Norm", compliance_status="FP32 activation")
class ObservedL2Norm(nn.Module):
    """L2-normalize the dense descriptor map along the channel dimension (stageC3)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# NMS — max-selection like MaxPool, no arithmetic (UNDER CONSTRUCTION)
# ---------------------------------------------------------------------------

@OpRegistry.register("ObservedSimpleNMS", passthrough=True)
class ObservedSimpleNMS(nn.Module):
    """NMS via repeated max_pool2d (stageB8). Pure max-selection — no arithmetic."""
    def __init__(self, radius: int = 4):
        super().__init__()
        self.radius = radius

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return _simple_nms(scores, self.radius)


# ---------------------------------------------------------------------------
# FP32-required ops — floating-point precision critical for correct placement
# ---------------------------------------------------------------------------

@OpRegistry.register("ObservedCoordOps", compliance_status="FP32 required")
class ObservedCoordOps(nn.Module):
    """Shift, normalize, and map keypoint coords to [-1,1] (stageD1-D3)."""
    def __init__(self, s: int = 8):
        super().__init__()
        self.s = s

    def forward(self, keypoints: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        _, _, h, w = descriptors.shape
        s = self.s
        kp = keypoints - s / 2 + 0.5
        kp = kp / torch.tensor(
            [(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
            dtype=kp.dtype, device=kp.device)[None]
        return kp * 2 - 1


@OpRegistry.register("ObservedGridSample", compliance_status="FP32 required")
class ObservedGridSample(nn.Module):
    """Bilinear descriptor sampling + per-keypoint L2 norm (stageD4)."""
    def forward(self, descriptors: torch.Tensor, keypoints_norm: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = descriptors.shape
        args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
        desc = F.grid_sample(
            descriptors, keypoints_norm.view(b, 1, -1, 2),
            mode='bilinear', **args)
        return F.normalize(desc.reshape(b, c, -1), p=2, dim=1)


# ===========================================================================
# SuperGlue ops
# ===========================================================================

def _log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def _arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


# --- Linear / passthrough ops -------------------------------------------------

@OpRegistry.register("ObservedKeypointNormalize", compliance_status="FP32 activation")
class ObservedKeypointNormalize(nn.Module):
    """Center + scale keypoint coordinates relative to image dimensions."""
    def forward(self, kpts: torch.Tensor, image_shape) -> torch.Tensor:
        _, _, height, width = image_shape
        one = kpts.new_tensor(1)
        size = torch.stack([one*width, one*height])[None]
        center = size / 2
        scaling = size.max(1, keepdim=True).values * 0.7
        return (kpts - center[:, None, :]) / scaling[:, None, :]


@OpRegistry.register("ObservedConcat", is_activation=False)
class ObservedConcat(nn.Module, QuantizedLayerMixin):
    """torch.cat wrapped as a module; quantizes each operand (mirrors QuantCat)."""
    def __init__(self, dim: int = 1, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.dim = dim
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, *tensors):
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            tensors = tensors[0]
        quantized = [self.quantize_input(t) for t in tensors]
        return torch.cat(quantized, dim=self.dim)


@OpRegistry.register("ObservedAdd", is_activation=False)
class ObservedAdd(nn.Module, QuantizedLayerMixin):
    """Element-wise add (residual connections); quantizes each operand (mirrors QuantAdd)."""
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q1 = self.quantize_input(x)
        q2 = self.quantize_input(y)
        return torch.add(q1, q2)


# --- Attention matmul / similarity ops --------------------------------------

@OpRegistry.register("ObservedAttentionScores", is_activation=False)
class ObservedAttentionScores(nn.Module, QuantizedLayerMixin):
    """Q·K^T scaled-dot-product (multi-head einsum + √d post-scale).

    Quantizes both operands to FP8 before the einsum, mirroring QuantMatMul.
    """
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        q = self.quantize_input(query)
        k = self.quantize_input(key)
        dim = q.shape[1]
        return torch.einsum('bdhn,bdhm->bhnm', q, k) / dim**0.5


@OpRegistry.register("ObservedAttentionApply", is_activation=False)
class ObservedAttentionApply(nn.Module, QuantizedLayerMixin):
    """Score·V einsum (multi-head).

    Quantizes both operands to FP8 before the einsum, mirroring QuantMatMul.
    """
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, prob: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        p = self.quantize_input(prob)
        v = self.quantize_input(value)
        return torch.einsum('bhnm,bdhm->bdhn', p, v)


@OpRegistry.register("ObservedDescMatmul", is_activation=False)
class ObservedDescMatmul(nn.Module, QuantizedLayerMixin):
    """Descriptor similarity einsum (desc0 · desc1ᵀ / √d).

    Quantizes both operands to FP8 before the einsum, mirroring QuantMatMul.
    """
    def __init__(self, descriptor_dim: int, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.scale = descriptor_dim ** 0.5
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, mdesc0: torch.Tensor, mdesc1: torch.Tensor) -> torch.Tensor:
        q0 = self.quantize_input(mdesc0)
        q1 = self.quantize_input(mdesc1)
        return torch.einsum('bdn,bdm->bnm', q0, q1) / self.scale


# --- FP32-required op --------------------------------------------------------

@OpRegistry.register("ObservedSinkhorn", compliance_status="FP32 required")
class ObservedSinkhorn(nn.Module):
    """Differentiable optimal transport via Sinkhorn in log-space.

    Iterative logsumexp — needs FP32 precision.
    """
    def __init__(self, iters: int = 100):
        super().__init__()
        self.iters = iters

    def forward(self, scores: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m*one).to(scores), (n*one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha_e = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha_e], -1)], 1)

        norm = -(ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = _log_sinkhorn_iterations(couplings, log_mu, log_nu, self.iters)
        return Z - norm


# --- Match selection --------------------------------------------------------

@OpRegistry.register("ObservedMatchSelect", compliance_status="FP32 activation")
class ObservedMatchSelect(nn.Module):
    """Mutual-nearest-neighbor match selection with score threshold."""
    def __init__(self, match_threshold: float = 0.2):
        super().__init__()
        self.match_threshold = match_threshold

    def forward(self, scores: torch.Tensor):
        max0 = scores[:, :-1, :-1].max(2)
        max1 = scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = _arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = _arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.match_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        return indices0, indices1, mscores0, mscores1
