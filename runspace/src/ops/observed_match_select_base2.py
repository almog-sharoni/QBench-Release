"""Design C' Base2 — match selector for log2-domain scores.

Mirrors `ObservedMatchSelect` in `observed_ops.py` with two substitutions:
  * The `match_threshold=0.2` probability comparison becomes
    `log2_threshold = math.log2(0.2)` comparison on the log2 score itself —
    no `exp` on the runtime path.
  * Returned match certainty scores are in log2 space (downstream consumers
    read them as log values).

No `torch.exp`, `torch.log`, or `torch.logsumexp` appears in this file
(validated by the §5.2 static grep gate).
"""
import math

import torch
import torch.nn as nn

from ..registry.op_registry import OpRegistry


# Threshold precomputed offline at FP64 per plan §R4 — do not quantize.
_LOG2_MATCH_THRESHOLD_DEFAULT = math.log2(0.2)  # ~ -2.3219280948873626


def _arange_like(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


@OpRegistry.register("ObservedMatchSelectBase2", compliance_status="FP32 activation")
class ObservedMatchSelectBase2(nn.Module):
    """Mutual-nearest-neighbor match selection, log2-space threshold.

    Input is log2_T of shape (B, M+1, N+1) — the dustbin row/col are excluded
    from the argmax search, as in the natural-base version.
    Certainty scores (mscores0/1) are returned in log2 space.
    """
    def __init__(self, match_threshold: float = 0.2):
        super().__init__()
        # Stored as a plain float (immutable, FP64-precomputed) per plan §R4.
        self.match_threshold = float(match_threshold)
        self.log2_threshold = math.log2(self.match_threshold)

    def forward(self, scores: torch.Tensor):
        """scores: (B, M+1, N+1) in log2 space."""
        max0 = scores[:, :-1, :-1].max(2)
        max1 = scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = _arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = _arange_like(indices1, 1)[None] == indices0.gather(1, indices1)

        # Threshold comparison is in log2 space; no exp needed.
        neg_inf = scores.new_tensor(float('-inf'))
        mscores0 = torch.where(mutual0, max0.values, neg_inf)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), neg_inf)
        valid0 = mutual0 & (mscores0 > self.log2_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        # Replace -inf with 0 in the returned scores so downstream mean()
        # aggregators don't NaN on all-invalid batches.
        zero = scores.new_tensor(0)
        mscores0 = torch.where(valid0, mscores0, zero)
        mscores1 = torch.where(valid1, mscores1, zero)
        return indices0, indices1, mscores0, mscores1
