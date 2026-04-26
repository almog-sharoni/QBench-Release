"""Design C' Base2 — base-2-arithmetic variant of ObservedLearnedSinkhorn.

Retrofit of the Design C unrolled Sinkhorn head to base-2 transcendentals
(exp2, log2, log2sumexp2) for LEO-NG FP8 E4M3 deployment. The class mirrors
`ObservedLearnedSinkhorn` verbatim except for:
  * `torch.logsumexp` -> `log2sumexp2`
  * `_assemble`: targets computed via `torch.log2`, not `torch.log`
  * `_build_features`: the four log-space feature groups are rescaled by
    `math.log(2)` before the MLP concat, so MLP weights pretrained under
    natural-base Design C transfer without reinterpretation.
  * Running state named `log2_T`; return value is in log2 (not divided by
    `log2(e)`). Downstream consumers must read it as log2.

No `torch.exp`, `torch.log`, or `torch.logsumexp` appears in this file
(validated by the §5.2 static grep gate).
"""
import math

import torch
import torch.nn as nn

from ..registry.op_registry import OpRegistry
from .quant_softmax import log2sumexp2


_LN2 = math.log(2.0)          # feature boundary rescaling constant (~0.6931)
_LOG2E = 1.0 / math.log(2.0)  # score scale — converts nats to bits (~1.4427)


@OpRegistry.register(
    "ObservedLearnedSinkhornBase2",
    compliance_status="FP8 E4M3 preferred, FP16 on log2_T state",
)
class ObservedLearnedSinkhornBase2(nn.Module):
    """Base-2 learned unrolled Sinkhorn head (Design C').

    State dict layout is identical to `ObservedLearnedSinkhorn`, so Phase 2
    checkpoints load without renaming.
    """
    FEATURE_DIM = 16  # 1 (LSE) + 1 (offset) + 5 (top-k gap) + 1 (spread) + 8 (proj desc)

    def __init__(
        self,
        iters: int = 3,
        descriptor_dim: int = 256,
        proj_dim: int = 8,
        hidden: int = 64,
        topk: int = 5,
    ):
        super().__init__()
        self.iters = iters
        self.descriptor_dim = descriptor_dim
        self.proj_dim = proj_dim
        self.topk = topk

        self.proj_A = nn.Linear(descriptor_dim, proj_dim, bias=True)
        self.proj_B = nn.Linear(descriptor_dim, proj_dim, bias=True)

        self.row_mlp = self._build_mlp(hidden)
        self.col_mlp = self._build_mlp(hidden)

    def _build_mlp(self, hidden: int) -> nn.Module:
        mlp = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
        return mlp

    def _project_descriptors(self, mdesc0: torch.Tensor, mdesc1: torch.Tensor):
        B = mdesc0.shape[0]
        dA = self.proj_A(mdesc0.transpose(1, 2))
        dB = self.proj_B(mdesc1.transpose(1, 2))
        dA = torch.cat([dA, dA.new_zeros(B, 1, self.proj_dim)], dim=1)
        dB = torch.cat([dB, dB.new_zeros(B, 1, self.proj_dim)], dim=1)
        return dA, dB

    def _build_features(self, log2_T: torch.Tensor, offset: torch.Tensor,
                        desc_proj: torch.Tensor, axis: str) -> torch.Tensor:
        """Build the 16-dim feature vector per row (or col).

        The four log-space groups (lse, offset, top-k gap, spread) are rescaled
        by ln(2) so the MLP sees natural-log-scale inputs — the weights learned
        under Design C natural-base training therefore apply unchanged.
        """
        if axis == 'row':
            lse = log2sumexp2(log2_T, dim=2)
            mx = log2_T.max(dim=2).values
            top_vals = log2_T.topk(self.topk, dim=2).values
            top_gap = mx.unsqueeze(-1) - top_vals
        else:  # 'col'
            lse = log2sumexp2(log2_T, dim=1)
            mx = log2_T.max(dim=1).values
            top_vals = log2_T.topk(self.topk, dim=1).values.transpose(1, 2)
            top_gap = mx.unsqueeze(-1) - top_vals

        spread = lse - mx

        # Feature boundary rescaling (plan §3.4): convert log2 -> natural log.
        lse_nat = lse * _LN2
        offset_nat = offset * _LN2
        top_gap_nat = top_gap * _LN2
        spread_nat = spread * _LN2

        feats = torch.cat([
            lse_nat.unsqueeze(-1),
            offset_nat.unsqueeze(-1),
            top_gap_nat,
            spread_nat.unsqueeze(-1),
            desc_proj,  # not rescaled — not a log quantity
        ], dim=-1)
        return feats

    def _assemble(self, scores: torch.Tensor, alpha: torch.Tensor):
        """Build S, log2_mu, log2_nu, norm. Mirrors ObservedLearnedSinkhorn.

        `scores` and `alpha` come in natural-log scale from the descriptor
        matmul; we rescale the assembled S by log2(e) so the entire base-2
        state variable satisfies `log2_T = log_T * log2(e)` (plan §2).
        Equivalently: the assembled log2 targets equal the natural-log targets
        times log2(e).
        """
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha_e = alpha.expand(b, 1, 1)
        S = torch.cat([torch.cat([scores, bins0], -1),
                       torch.cat([bins1, alpha_e], -1)], 1)
        # Rescale score matrix: log2-domain logits are log-domain logits * log2(e).
        S = S * _LOG2E

        norm = -torch.log2(ms + ns)
        log2_mu = torch.cat([norm.expand(m), torch.log2(ns)[None] + norm])
        log2_nu = torch.cat([norm.expand(n), torch.log2(ms)[None] + norm])
        log2_mu = log2_mu[None].expand(b, -1)
        log2_nu = log2_nu[None].expand(b, -1)
        return S, log2_mu, log2_nu, norm

    def forward(self, scores: torch.Tensor, alpha: torch.Tensor,
                mdesc0: torch.Tensor, mdesc1: torch.Tensor,
                return_trace: bool = False):
        """Run T unrolled iterations in base-2 log domain.

        Returns `log2_T` of shape (B, M+1, N+1). When `return_trace=True`,
        returns `(log2_T, trace, log2_mu, log2_nu)`.
        """
        S, log2_mu, log2_nu, norm = self._assemble(scores, alpha)
        B, Mp1, Np1 = S.shape
        u = S.new_zeros(B, Mp1)
        v = S.new_zeros(B, Np1)
        dA, dB = self._project_descriptors(mdesc0, mdesc1)

        trace = [] if return_trace else None

        for _ in range(self.iters):
            u = log2_mu - log2sumexp2(S + v.unsqueeze(1), dim=2)
            log2_T_cur = S + u.unsqueeze(2) + v.unsqueeze(1)
            feats_r = self._build_features(log2_T_cur, u, dA, axis='row')
            # MLP features are fed in natural-log scale (rescaled inside
            # _build_features), so the MLP output is also in natural log units.
            # Scale it to log2 units before adding to u_b2 (which lives in log2).
            u = u + self.row_mlp(feats_r).squeeze(-1) * _LOG2E

            v = log2_nu - log2sumexp2(S + u.unsqueeze(2), dim=1)
            log2_T_cur = S + u.unsqueeze(2) + v.unsqueeze(1)
            feats_c = self._build_features(log2_T_cur, v, dB, axis='col')
            v = v + self.col_mlp(feats_c).squeeze(-1) * _LOG2E

            if return_trace:
                trace.append(S + u.unsqueeze(2) + v.unsqueeze(1) - norm)

        final = S + u.unsqueeze(2) + v.unsqueeze(1) - norm
        if return_trace:
            return final, trace, log2_mu, log2_nu
        return final
