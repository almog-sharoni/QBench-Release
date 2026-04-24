"""Design C training losses: match NLL, marginal L2, auxiliary intermediate-iter NLL.

All operate on log transport matrices of shape (B, M+1, N+1) where the last
row/column are the dustbin channel.
"""
from __future__ import annotations

import torch


def match_nll_loss(
    log_T: torch.Tensor,
    gt_matches0: torch.Tensor,
    gt_matches1: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood at GT match + dustbin positions (SuperGlue paper eq. 3).

    Args:
      log_T: (B, M+1, N+1) log transport matrix.
      gt_matches0: (B, M) long tensor. Values in [0, N-1] = matched col index.
                   Value -1 = kpt0 unmatched (assigned to dustbin col = N).
      gt_matches1: (B, N) long tensor. Values in [0, M-1] = matched row index.
                   Value -1 = kpt1 unmatched (assigned to dustbin row = M).
    Returns:
      Scalar loss averaged across the batch, with each image's loss averaged
      over its own matched + unmatched counts.
    """
    B, Mp1, Np1 = log_T.shape
    M = Mp1 - 1
    N = Np1 - 1
    device = log_T.device

    total = log_T.new_zeros(())
    denom = 0
    for b in range(B):
        g0 = gt_matches0[b]  # (M,)
        g1 = gt_matches1[b]  # (N,)
        matched_mask0 = g0 >= 0
        unmatched_mask0 = ~matched_mask0
        unmatched_mask1 = g1 < 0

        losses = []

        # Matched entries: one per (i, g0[i]) with g0[i] >= 0.
        if matched_mask0.any():
            i_idx = torch.nonzero(matched_mask0, as_tuple=False).squeeze(-1)
            j_idx = g0[matched_mask0]
            losses.append(-log_T[b, i_idx, j_idx])

        # kpt0 unmatched → dustbin column N.
        if unmatched_mask0.any():
            i_idx = torch.nonzero(unmatched_mask0, as_tuple=False).squeeze(-1)
            losses.append(-log_T[b, i_idx, N])

        # kpt1 unmatched → dustbin row M.
        if unmatched_mask1.any():
            j_idx = torch.nonzero(unmatched_mask1, as_tuple=False).squeeze(-1)
            losses.append(-log_T[b, M, j_idx])

        if losses:
            per_image = torch.cat(losses)
            total = total + per_image.mean()
            denom += 1

    if denom == 0:
        return log_T.new_zeros(())
    return total / denom


def marginal_l2_loss(
    log_T: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
) -> torch.Tensor:
    """L2 penalty on deviation of row/col log-marginals from targets."""
    row_err = torch.logsumexp(log_T, dim=2) - log_mu  # (B, M+1)
    col_err = torch.logsumexp(log_T, dim=1) - log_nu  # (B, N+1)
    return (row_err.pow(2).mean() + col_err.pow(2).mean()) * 0.5


def design_c_total_loss(
    final_log_T: torch.Tensor,
    trace: list[torch.Tensor],
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    gt_matches0: torch.Tensor,
    gt_matches1: torch.Tensor,
    lambda_marg: float = 0.1,
    lambda_aux: float = 0.3,
) -> dict:
    """Combine match NLL + marginal + auxiliary intermediate-iter NLL.

    `trace` is the per-iteration log_T list emitted by
    `ObservedLearnedSinkhorn.forward(..., return_trace=True)` — length T.
    The final entry equals `final_log_T`. Auxiliary loss applies at all
    intermediate iterations (t=1 .. T-1) with weight `lambda_aux` each.
    """
    loss_match = match_nll_loss(final_log_T, gt_matches0, gt_matches1)
    loss_marg = marginal_l2_loss(final_log_T, log_mu, log_nu)
    loss_aux = final_log_T.new_zeros(())
    n_aux = max(len(trace) - 1, 0)
    for t in range(n_aux):
        loss_aux = loss_aux + match_nll_loss(trace[t], gt_matches0, gt_matches1)
    total = loss_match + lambda_marg * loss_marg + lambda_aux * loss_aux
    return {
        'total': total,
        'match': loss_match.detach(),
        'marg': loss_marg.detach(),
        'aux': loss_aux.detach(),
    }
