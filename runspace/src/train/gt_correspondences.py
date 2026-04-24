"""Depth + pose → ground-truth match indices for SuperGlue training.

Given keypoints in image-0 and image-1 along with depth maps, camera intrinsics,
and the relative pose T_0to1, project each kpt0 through depth0 into world space
and into image-1, then find the nearest kpt1 pixel. If the projection lands
within `reproj_threshold_px`, the two keypoints match. Keypoints without valid
depth or with out-of-frame projections are assigned to the dustbin.

Returns per-batch-image tensors `gt_matches0 (M,)` and `gt_matches1 (N,)`:
  value in [0, other_side_count-1]  ⇒ matched index
  value = -1                        ⇒ unmatched (dustbin)
"""
from __future__ import annotations

import torch


def _sample_depth(depth: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor depth lookup. Out-of-frame returns 0.

    Args:
      depth: (H, W) float depth map.
      pixels: (K, 2) float pixel coords in (x, y).
    Returns:
      (K,) float depth values.
    """
    H, W = depth.shape
    x = pixels[:, 0].round().long()
    y = pixels[:, 1].round().long()
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    out = depth.new_zeros(pixels.shape[0])
    if in_bounds.any():
        out[in_bounds] = depth[y[in_bounds], x[in_bounds]]
    return out


def _reproject(
    kpts: torch.Tensor,
    depth: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T_0to1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproject (x, y) pixels in camera 0 into camera 1 via depth+pose.

    Args:
      kpts: (K, 2) float pixel coords (x, y) in image-0.
      depth: (H, W) depth map for image-0 (in meters, 0 = invalid).
      K0, K1: (3, 3) intrinsics.
      T_0to1: (4, 4) rigid transform cam0→cam1.
    Returns:
      projected: (K, 2) float pixel coords in image-1.
      valid: (K,) bool — True iff depth was finite and the projection landed
             in front of camera-1.
    """
    K = kpts.shape[0]
    device = kpts.device
    if K == 0:
        return kpts.new_zeros(0, 2), kpts.new_zeros(0, dtype=torch.bool)

    d = _sample_depth(depth, kpts)                          # (K,)
    ones = torch.ones(K, device=device, dtype=kpts.dtype)
    pix_h = torch.stack([kpts[:, 0], kpts[:, 1], ones], dim=-1)  # (K, 3)

    K0_inv = torch.linalg.inv(K0)
    rays = pix_h @ K0_inv.T                                  # (K, 3) unit-z rays
    pts_cam0 = rays * d.unsqueeze(-1)                        # (K, 3)

    pts_cam0_h = torch.cat([pts_cam0, ones.unsqueeze(-1)], dim=-1)  # (K, 4)
    pts_cam1_h = pts_cam0_h @ T_0to1.T                       # (K, 4)
    pts_cam1 = pts_cam1_h[:, :3]                             # (K, 3)

    z1 = pts_cam1[:, 2]
    valid = (d > 0) & (z1 > 1e-4)

    z1_safe = torch.where(valid, z1, torch.ones_like(z1))
    pix1_h = pts_cam1 @ K1.T                                 # (K, 3)
    pix1 = pix1_h[:, :2] / z1_safe.unsqueeze(-1)             # (K, 2)
    return pix1, valid


def compute_gt_matches(
    kpts0: torch.Tensor,
    kpts1: torch.Tensor,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T_0to1: torch.Tensor,
    reproj_threshold_px: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image GT match indices (dustbin = -1).

    Args:
      kpts0: (M, 2) float pixel coords (x, y).
      kpts1: (N, 2) float pixel coords.
      depth0, depth1: (H, W) depth maps.
      K0, K1: (3, 3) intrinsics (scaled to match the image size used by
              kpts{0,1}).
      T_0to1: (4, 4).
      reproj_threshold_px: match acceptance radius in pixels.
    Returns:
      gt0: (M,) long. Each entry ∈ [0, N-1] or -1.
      gt1: (N,) long. Each entry ∈ [0, M-1] or -1.
    """
    M = kpts0.shape[0]
    N = kpts1.shape[0]
    device = kpts0.device

    proj01, valid01 = _reproject(kpts0, depth0, K0, K1, T_0to1)
    T_1to0 = torch.linalg.inv(T_0to1)
    proj10, valid10 = _reproject(kpts1, depth1, K1, K0, T_1to0)

    gt0 = torch.full((M,), -1, dtype=torch.long, device=device)
    gt1 = torch.full((N,), -1, dtype=torch.long, device=device)

    if M == 0 or N == 0:
        return gt0, gt1

    # For each valid kpt0 projection, find the nearest kpt1.
    # dist0[i, j] = ||proj01[i] - kpts1[j]||
    dist0 = torch.cdist(proj01, kpts1)                       # (M, N)
    dist0 = torch.where(
        valid01.unsqueeze(-1),
        dist0,
        torch.full_like(dist0, float('inf')),
    )
    min0_vals, min0_idx = dist0.min(dim=1)                   # (M,)

    dist1 = torch.cdist(proj10, kpts0)                       # (N, M)
    dist1 = torch.where(
        valid10.unsqueeze(-1),
        dist1,
        torch.full_like(dist1, float('inf')),
    )
    min1_vals, min1_idx = dist1.min(dim=1)                   # (N,)

    # Mutual-NN with threshold.
    for i in range(M):
        if min0_vals[i] > reproj_threshold_px:
            continue
        j = int(min0_idx[i].item())
        if min1_vals[j] > reproj_threshold_px:
            continue
        if int(min1_idx[j].item()) != i:
            continue
        gt0[i] = j
        gt1[j] = i
    return gt0, gt1


def batch_compute_gt_matches(
    kpts0: torch.Tensor,
    kpts1: torch.Tensor,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T_0to1: torch.Tensor,
    reproj_threshold_px: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched wrapper.

    Shapes:
      kpts0, kpts1: (B, K, 2)  (K_max; if SuperPoint returns variable counts,
                                pad with NaN or handle per-image upstream)
      depth0, depth1: (B, H, W)
      K0, K1, T_0to1: (B, 3, 3), (B, 3, 3), (B, 4, 4)
    """
    B = kpts0.shape[0]
    gt0_list = []
    gt1_list = []
    for b in range(B):
        g0, g1 = compute_gt_matches(
            kpts0[b], kpts1[b],
            depth0[b], depth1[b],
            K0[b], K1[b], T_0to1[b],
            reproj_threshold_px=reproj_threshold_px,
        )
        gt0_list.append(g0)
        gt1_list.append(g1)
    return torch.stack(gt0_list, dim=0), torch.stack(gt1_list, dim=0)
