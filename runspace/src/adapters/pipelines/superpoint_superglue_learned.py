"""Pipeline loader for Design C — SuperPoint + SuperGlueLearned.

Mirrors `superpoint_superglue.py` but instantiates `SuperGlueLearned` (which
swaps in `ObservedLearnedSinkhorn`) in place of the vendored `SuperGlue`.

Component prefixes match `superpoint_superglue.py` so `quantize_components:
["superglue"]` continues to target `backbone.superglue` Conv1d layers. The
learned head's `Linear` layers are inside `backbone.superglue.sinkhorn` and
are NOT quantized unless the config explicitly adds `Linear` to `quantized_ops`.
"""
import os
import sys

import torch
import torch.nn as nn

from src.adapters.pipeline_registry import register_pipeline
from src.eval.metrics.matching import MatchingMetrics


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
    'superpoint_superglue_learned',
    metrics_cls=MatchingMetrics,
    components={
        'superpoint': 'backbone.superpoint',
        'superglue':  'backbone.superglue',
    },
    required_input_keys=('image0', 'image1'),
)
def _load_superpoint_superglue_learned(model_cfg: dict) -> torch.nn.Module:
    """Load SuperPoint + SuperGlueLearned (Design C head).

    model_cfg keys (in addition to superpoint_superglue):
      sg_config.learned_sinkhorn_iterations : int = 3
      sg_config.learned_proj_dim            : int = 8
      sg_config.learned_head_ckpt           : optional path
    """
    repo_path = _require(model_cfg, 'repo_path', 'superpoint_superglue_learned')
    sg_weights = _require(model_cfg, 'sg_weights', 'superpoint_superglue_learned')

    _inject_repo(repo_path, 'superpoint_superglue_learned')

    from models.superpoint import SuperPoint
    from models.superglue_training import SuperGlueLearned

    sp_cfg = dict(model_cfg.get('sp_config') or {})
    sg_cfg = dict(model_cfg.get('sg_config') or {})
    sg_cfg['weights'] = sg_weights

    class MatchingLearned(nn.Module):
        """SuperPoint + SuperGlueLearned, same forward contract as `models.matching.Matching`."""
        def __init__(self):
            super().__init__()
            self.superpoint = SuperPoint(sp_cfg)
            self.superglue = SuperGlueLearned(sg_cfg)

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

    return MatchingLearned()
