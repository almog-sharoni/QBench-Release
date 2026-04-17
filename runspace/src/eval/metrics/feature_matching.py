import torch


class FeatureMatchingMetrics:
    """
    Accumulates keypoint detection and descriptor quality metrics
    across batches. Compatible with the same update()/compute() interface
    as MetricsEngine so the runner can call them interchangeably.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._total_images = 0
        self._sum_keypoints = 0.0
        self._sum_scores = 0.0
        self._sum_desc_norm = 0.0
        self._sum_repeatability = 0.0
        self._repeatability_count = 0

    def update(self, outputs: dict, targets) -> None:
        """
        outputs — dict from FeatureMatchingAdapter.forward():
            'keypoints':    List[Tensor[N, 2]]
            'scores':       List[Tensor[N]]
            'descriptors':  List[Tensor[256, N]]
        targets — same batch dict as inputs; may contain 'T_0to1'/'K0'/'K1'
                  for repeatability computation (ignored if absent).
        """
        keypoints_list = outputs['keypoints']
        scores_list = outputs['scores']
        descriptors_list = outputs['descriptors']

        for kp, sc, desc in zip(keypoints_list, scores_list, descriptors_list):
            self._total_images += 1
            self._sum_keypoints += kp.shape[0]
            if sc.numel() > 0:
                self._sum_scores += sc.mean().item()
            if desc.numel() > 0:
                self._sum_desc_norm += desc.norm(dim=0).mean().item()

        if isinstance(targets, dict) and 'T_0to1' in targets:
            rep = self._compute_repeatability_batch(keypoints_list, targets)
            if rep is not None:
                self._sum_repeatability += rep
                self._repeatability_count += 1

    def _compute_repeatability_batch(self, keypoints_list, batch: dict) -> float | None:
        if len(keypoints_list) < 2:
            return None
        kp0 = keypoints_list[0]
        kp1 = keypoints_list[1]
        T_0to1 = batch['T_0to1'][0]  # [4, 4]
        K0 = batch['K0'][0]           # [3, 3]
        K1 = batch['K1'][0]           # [3, 3]

        if kp0.shape[0] == 0 or kp1.shape[0] == 0:
            return 0.0

        ones = torch.ones(kp0.shape[0], 1, device=kp0.device)
        kp0_h = torch.cat([kp0, ones], dim=1)  # [N, 3]
        pts0_cam = torch.linalg.solve(K0, kp0_h.T)  # [3, N]
        pts0_cam_h = torch.cat([pts0_cam, torch.ones(1, pts0_cam.shape[1], device=kp0.device)], dim=0)
        pts1_cam_h = T_0to1 @ pts0_cam_h
        pts1_proj = K1 @ pts1_cam_h[:3]
        pts1_proj = pts1_proj[:2] / pts1_proj[2:3]  # [2, N]
        projected = pts1_proj.T  # [N, 2]

        dists = torch.cdist(projected, kp1.float())  # [N, M]
        min_dists = dists.min(dim=1).values
        threshold = 3.0  # pixels
        repeatability = (min_dists < threshold).float().mean().item()
        return repeatability

    def compute(self) -> dict:
        if self._total_images == 0:
            return {
                'fm_num_keypoints': 0.0,
                'fm_mean_score': 0.0,
                'fm_desc_norm': 0.0,
                'fm_repeatability': 0.0,
            }
        return {
            'fm_num_keypoints': self._sum_keypoints / self._total_images,
            'fm_mean_score': self._sum_scores / self._total_images,
            'fm_desc_norm': self._sum_desc_norm / self._total_images,
            'fm_repeatability': (
                self._sum_repeatability / self._repeatability_count
                if self._repeatability_count > 0 else 0.0
            ),
        }

    def reset(self) -> None:
        self._reset()
