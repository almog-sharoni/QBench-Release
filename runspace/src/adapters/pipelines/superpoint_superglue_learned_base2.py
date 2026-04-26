"""Pipeline loader for Design C' Base2 — SuperPoint + SuperGlueLearned (Base2 head).

Mirrors `superpoint_superglue_learned.py` but swaps in the base-2 Sinkhorn head
(`ObservedLearnedSinkhornBase2`) and match selector (`ObservedMatchSelectBase2`)
after constructing the `SuperGlueLearned` module. The component prefixes match
`superpoint_superglue_learned` so the adapter and metric infra transfer
unchanged.

State dict structure of `ObservedLearnedSinkhornBase2` is identical to
`ObservedLearnedSinkhorn` (same MLP and proj layer names/shapes), so
Phase 2 checkpoints load via `Trainer.load_checkpoint` without renaming.
"""
import os
import sys

import torch
import torch.nn as nn

from src.adapters.pipeline_registry import register_pipeline
from src.eval.metrics.matching import MatchingMetrics
from src.ops.observed_ops_base2 import ObservedLearnedSinkhornBase2
from src.ops.observed_match_select_base2 import ObservedMatchSelectBase2


def _require(cfg: dict, key: str, pipeline_name: str):
    if key not in cfg:
        raise ValueError(
            f"Pipeline '{pipeline_name}': missing required config key '{key}'. "
            f"Got keys: {sorted(cfg.keys())}"
        )
    return cfg[key]


def _inject_repo(repo_path: str, pipeline_name: str) -> None:
    abs_path = os.path.abspath(repo_path)
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(
            f"Pipeline '{pipeline_name}': repo_path does not exist: {abs_path}"
        )
    if abs_path not in sys.path:
        sys.path.append(abs_path)


@register_pipeline(
    'superpoint_superglue_learned_base2',
    metrics_cls=MatchingMetrics,
    components={
        'superpoint': 'backbone.superpoint',
        'superglue':  'backbone.superglue',
    },
    required_input_keys=('image0', 'image1'),
)
def _load_superpoint_superglue_learned_base2(model_cfg: dict) -> torch.nn.Module:
    """Load SuperPoint + SuperGlueLearned with Design C' Base2 head.

    Accepts the same keys as `superpoint_superglue_learned`. The base-2 head
    is swapped in after construction; if a `learned_head_ckpt` is specified
    via model_cfg, it is loaded directly into the base-2 sinkhorn (state dict
    is layout-compatible with ObservedLearnedSinkhorn).
    """
    repo_path = _require(model_cfg, 'repo_path', 'superpoint_superglue_learned_base2')
    sg_weights = _require(model_cfg, 'sg_weights', 'superpoint_superglue_learned_base2')

    _inject_repo(repo_path, 'superpoint_superglue_learned_base2')

    from models.superpoint import SuperPoint
    from models.superglue_training import SuperGlueLearned

    sp_cfg = dict(model_cfg.get('sp_config') or {})
    sg_cfg = dict(model_cfg.get('sg_config') or {})
    sg_cfg['weights'] = sg_weights
    # Force the natural-base head NOT to preload a checkpoint; we swap it out.
    sg_cfg['learned_head_ckpt'] = None

    sg_iters = int(sg_cfg.get('learned_sinkhorn_iterations', 3))
    sg_proj_dim = int(sg_cfg.get('learned_proj_dim', 8))
    sg_desc_dim = int(sg_cfg.get('descriptor_dim', 256))
    sg_match_thr = float(sg_cfg.get('match_threshold', 0.2))

    class MatchingLearnedBase2(nn.Module):
        """SuperPoint + SuperGlueLearned (Base2 head swap)."""
        def __init__(self):
            super().__init__()
            self.superpoint = SuperPoint(sp_cfg)
            self.superglue = SuperGlueLearned(sg_cfg)
            # Swap natural-base head for the Base2 head. Weight shapes match,
            # so a later `load_state_dict` from a Phase 2 checkpoint works.
            self.superglue.sinkhorn = ObservedLearnedSinkhornBase2(
                iters=sg_iters,
                descriptor_dim=sg_desc_dim,
                proj_dim=sg_proj_dim,
            )
            self.superglue.match_select = ObservedMatchSelectBase2(
                match_threshold=sg_match_thr
            )

            ckpt_path = model_cfg.get('sg_config', {}).get('learned_head_ckpt')
            if ckpt_path:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                state = ckpt['head_state_dict'] if isinstance(ckpt, dict) and 'head_state_dict' in ckpt else ckpt
                self.superglue.sinkhorn.load_state_dict(state)
                if isinstance(ckpt, dict) and 'bin_score' in ckpt:
                    with torch.no_grad():
                        self.superglue.bin_score.copy_(ckpt['bin_score'])
                print(f"Loaded Base2 learned Sinkhorn head weights from {ckpt_path}")

        def forward(self, data):
            pred = {}
            if 'keypoints0' not in data:
                pred0 = self.superpoint({'image': data['image0']})
                pred = {**pred, **{k + '0': v for k, v in pred0.items()}}
            if 'keypoints1' not in data:
                pred1 = self.superpoint({'image': data['image1']})
                pred = {**pred, **{k + '1': v for k, v in pred1.items()}}

            data = {**data, **pred}
            for k in data:
                if isinstance(data[k], (list, tuple)):
                    data[k] = torch.stack(data[k])

            pred = {**pred, **self.superglue(data)}
            return pred

    return MatchingLearnedBase2()
