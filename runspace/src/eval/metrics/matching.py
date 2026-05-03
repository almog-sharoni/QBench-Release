import numpy as np
import torch


def _compute_epipolar_error(kpts0: np.ndarray, kpts1: np.ndarray,
                             T_0to1: np.ndarray, K0: np.ndarray,
                             K1: np.ndarray) -> np.ndarray:
    """Sampson distance between matched keypoints given ground-truth pose."""
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    kpts0 = np.concatenate([kpts0, np.ones((len(kpts0), 1))], axis=1)
    kpts1 = np.concatenate([kpts1, np.ones((len(kpts1), 1))], axis=1)

    R = T_0to1[:3, :3]
    t = T_0to1[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R

    Ep0 = kpts0 @ E.T
    p1Ep0 = np.sum(kpts1 * Ep0, axis=1)
    Etp1 = kpts1 @ E
    denom0 = np.maximum(Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2, 1e-12)
    denom1 = np.maximum(Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2, 1e-12)
    d = p1Ep0 ** 2 * (1.0 / denom0 + 1.0 / denom1)
    return d


def _angle_error_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return np.inf
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def _angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    cos = (np.trace(R1.T @ R2) - 1) / 2
    return np.rad2deg(np.abs(np.arccos(np.clip(cos, -1.0, 1.0))))


def _compute_pose_error(T_0to1: np.ndarray, R: np.ndarray,
                        t: np.ndarray) -> tuple[float, float]:
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    err_t = _angle_error_vec(t.squeeze(), t_gt)
    err_t = min(err_t, 180 - err_t)
    err_R = _angle_error_mat(R, R_gt)
    return err_t, err_R


def _pose_auc(errors: list[float], thresholds: list[int]) -> list[float]:
    """
    Pose AUC per SuperGluePretrainedNetwork: for each threshold t, build the
    cumulative-recall curve r(e) over sorted errors, clip to [0, t], and
    integrate against the normalized error axis [0, 1].
    """
    errors_sorted = np.sort(np.array(errors, dtype=float))
    n = len(errors_sorted)
    if n == 0:
        return [0.0 for _ in thresholds]
    recall = np.arange(1, n + 1) / n
    errors_ext = np.concatenate([[0.0], errors_sorted])
    recall_ext = np.concatenate([[0.0], recall])
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    aucs = []
    for thr in thresholds:
        last = np.searchsorted(errors_ext, thr)
        e = np.concatenate([errors_ext[:last], [float(thr)]])
        r = np.concatenate([recall_ext[:last], [recall_ext[last - 1]]])
        aucs.append(float(_trapz(r, e) / thr))
    return aucs


def _estimate_pose(kpts0: np.ndarray, kpts1: np.ndarray,
                   K0: np.ndarray, K1: np.ndarray,
                   thresh: float) -> tuple | None:
    try:
        import cv2
    except ImportError:
        return None
    if len(kpts0) < 5:
        return None
    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean
    kpts0n = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1n = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    E, mask = cv2.findEssentialMat(kpts0n, kpts1n, np.eye(3), threshold=norm_thresh,
                                   prob=0.99999, method=cv2.RANSAC)
    if E is None:
        return None
    best_num = 0
    best = None
    for _E in np.split(E, len(E) // 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0n, kpts1n, np.eye(3), 1e9, mask=mask)
        if n > best_num:
            best_num = n
            best = (R, t[:, 0], mask[:, 0].astype(bool))
    return best


class MatchingMetrics:
    """
    Measures image-pair matching quality produced by a SuperPoint+SuperGlue pipeline.

    When targets contain 'T_0to1', 'K0', 'K1' (from ScanNetPairsDataset):
      - precision: fraction of matches satisfying epipolar constraint (< 5e-4 Sampson dist)
      - matching_score: correct_inliers / total_keypoints_in_image0
      - pose_auc_5 / _10 / _20: AUC of max(err_t, err_R) at 5/10/20 deg thresholds

    Without GT (image_directory dataset):
      - precision is undefined (set to 0)
      - mean_num_matches: average number of matches per image
      - mean_matching_score: matches / keypoints
    """

    EPIPOLAR_THRESH = 5e-4
    POSE_THRESH_PX = 1.0

    def __init__(self):
        self._reset()

    def _reset(self):
        self._total_pairs = 0
        self._sum_precision = 0.0
        self._sum_matching_score = 0.0
        self._sum_num_matches = 0.0
        self._sum_match_certainty = 0.0
        self._pairs_with_matches = 0
        self._pose_errors: list[float] = []
        self._has_gt = False
        # SuperPoint-side health metrics (per-image, averaged over both images in each pair)
        self._total_images = 0
        self._sum_num_keypoints = 0.0
        self._sum_kp_score = 0.0
        self._sum_desc_norm = 0.0
        # Repeatability: per-pair, epipolar-based (fraction of kp0 within Sampson tolerance of any kp1)
        self._sum_repeatability = 0.0
        self._repeat_pairs = 0

    def update(self, outputs: dict, targets) -> None:
        """
        outputs: dict from Matching.forward():
            'keypoints0':       List[Tensor[N0, 2]]
            'keypoints1':       List[Tensor[N1, 2]]
            'matches0':         List[Tensor[N0]]    (index into kpts1, -1 = unmatched)
            'matching_scores0': List[Tensor[N0]]
        targets: batch dict; may contain 'T_0to1', 'K0', 'K1' for GT evaluation.
        """
        kpts0_list = outputs['keypoints0']
        kpts1_list = outputs['keypoints1']
        matches0 = outputs['matches0']
        scores0 = outputs['matching_scores0']

        # SuperPoint-side tensors (keypoint confidences, descriptors) — optional.
        kp_scores0_list = outputs.get('scores0')
        kp_scores1_list = outputs.get('scores1')
        desc0_list = outputs.get('descriptors0')
        desc1_list = outputs.get('descriptors1')

        has_gt = (isinstance(targets, dict) and
                  'T_0to1' in targets and 'K0' in targets and 'K1' in targets)
        if has_gt:
            self._has_gt = True

        batch_size = len(matches0)
        for i in range(batch_size):
            kp0 = kpts0_list[i].cpu().numpy()   # [N0, 2]
            kp1 = kpts1_list[i].cpu().numpy()   # [N1, 2]
            m0 = matches0[i].cpu().numpy()        # [N0]
            sc0 = scores0[i].cpu().numpy()        # [N0]

            # Accumulate SuperPoint-side health metrics for both images in the pair.
            for kp, sc_opt, desc_opt in (
                (kp0, kp_scores0_list[i] if kp_scores0_list is not None else None,
                      desc0_list[i] if desc0_list is not None else None),
                (kp1, kp_scores1_list[i] if kp_scores1_list is not None else None,
                      desc1_list[i] if desc1_list is not None else None),
            ):
                self._total_images += 1
                self._sum_num_keypoints += float(len(kp))
                if sc_opt is not None:
                    sc_arr = sc_opt.detach().cpu().numpy() if hasattr(sc_opt, 'detach') else np.asarray(sc_opt)
                    if sc_arr.size > 0:
                        self._sum_kp_score += float(sc_arr.mean())
                if desc_opt is not None and hasattr(desc_opt, 'shape') and desc_opt.numel() > 0:
                    # SuperPoint convention: (D, N) with D=256. L2-norm along feature axis.
                    self._sum_desc_norm += float(desc_opt.detach().float().norm(p=2, dim=0).mean().item())

            valid = m0 > -1
            mkpts0 = kp0[valid]
            mkpts1 = kp1[m0[valid]]
            num_matches = int(valid.sum())
            self._sum_num_matches += num_matches
            self._total_pairs += 1

            if num_matches > 0:
                self._sum_match_certainty += float(sc0[valid].mean())
                self._pairs_with_matches += 1

            if not has_gt and len(kp0) > 0:
                self._sum_matching_score += num_matches / len(kp0)

            if has_gt and num_matches > 0:
                T = targets['T_0to1'][i].cpu().numpy()
                K0 = targets['K0'][i].cpu().numpy()
                K1 = targets['K1'][i].cpu().numpy()

                epi_errs = _compute_epipolar_error(mkpts0, mkpts1, T, K0, K1)
                correct = epi_errs < self.EPIPOLAR_THRESH
                precision = float(correct.mean()) if len(correct) > 0 else 0.0
                num_correct = int(correct.sum())
                self._sum_precision += precision
                if len(kp0) > 0:
                    self._sum_matching_score += num_correct / len(kp0)

                # Repeatability: fraction of kp0 whose nearest kp1 under the GT
                # epipolar geometry is within EPIPOLAR_THRESH Sampson distance.
                # Measures keypoint detector stability independent of the matcher.
                if len(kp0) > 0 and len(kp1) > 0:
                    n0 = len(kp0)
                    n1 = len(kp1)
                    kp0_rep = np.repeat(kp0, n1, axis=0)
                    kp1_tile = np.tile(kp1, (n0, 1))
                    d = _compute_epipolar_error(kp0_rep, kp1_tile, T, K0, K1).reshape(n0, n1)
                    repeated = (d.min(axis=1) < self.EPIPOLAR_THRESH)
                    self._sum_repeatability += float(repeated.mean())
                    self._repeat_pairs += 1

                ret = _estimate_pose(mkpts0, mkpts1, K0, K1, self.POSE_THRESH_PX)
                if ret is not None:
                    R, t, _ = ret
                    err_t, err_R = _compute_pose_error(T, R, t)
                    self._pose_errors.append(max(err_t, err_R))
                else:
                    self._pose_errors.append(np.inf)
            elif has_gt:
                self._sum_precision += 0.0
                self._pose_errors.append(np.inf)

    def compute(self) -> dict:
        if self._total_pairs == 0:
            return {
                'matching_precision': 0.0,
                'matching_score': 0.0,
                'mean_num_matches': 0.0,
                'match_certainty': 0.0,
                'pose_auc_5': 0.0,
                'pose_auc_10': 0.0,
                'pose_auc_20': 0.0,
                'fm_num_keypoints': 0.0,
                'fm_mean_score': 0.0,
                'fm_desc_norm': 0.0,
                'fm_repeatability': 0.0,
            }

        result = {
            'matching_precision': self._sum_precision / self._total_pairs if self._has_gt else 0.0,
            'matching_score': self._sum_matching_score / self._total_pairs,
            'mean_num_matches': self._sum_num_matches / self._total_pairs,
            'match_certainty': (self._sum_match_certainty / self._pairs_with_matches
                                if self._pairs_with_matches > 0 else 0.0),
            'pose_auc_5': 0.0,
            'pose_auc_10': 0.0,
            'pose_auc_20': 0.0,
            'fm_num_keypoints': (self._sum_num_keypoints / self._total_images
                                 if self._total_images > 0 else 0.0),
            'fm_mean_score':    (self._sum_kp_score / self._total_images
                                 if self._total_images > 0 else 0.0),
            'fm_desc_norm':     (self._sum_desc_norm / self._total_images
                                 if self._total_images > 0 else 0.0),
            'fm_repeatability': (self._sum_repeatability / self._repeat_pairs
                                 if self._repeat_pairs > 0 else 0.0),
        }

        if self._pose_errors:
            aucs = _pose_auc(self._pose_errors, [5, 10, 20])
            result['pose_auc_5'] = aucs[0]
            result['pose_auc_10'] = aucs[1]
            result['pose_auc_20'] = aucs[2]

        return result

    def reset(self) -> None:
        self._reset()
