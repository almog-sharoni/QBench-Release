import os
import sys
import torch
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
    'superpoint_superglue',
    metrics_cls=MatchingMetrics,
    components={
        'superpoint': 'backbone.superpoint',
        'superglue':  'backbone.superglue',
    },
)
def _load_superpoint_superglue(model_cfg: dict) -> torch.nn.Module:
    """
    Loads the combined SuperPoint + SuperGlue Matching pipeline.

    model_cfg keys:
      repo_path  — path to SuperGluePretrainedNetwork root (required)
      sg_weights — 'indoor' | 'outdoor' (required; selects pre-trained SuperGlue)
      sp_config  — dict of SuperPoint config overrides (optional)
      sg_config  — dict of SuperGlue config overrides (optional)

    Both models auto-load their pre-trained weights from models/weights/ in the repo.
    Quantization targets are selected via quantize_components in the adapter config:
      - ["superpoint"] → only SuperPoint Conv2d layers
      - ["superglue"]  → only SuperGlue Conv1d layers
      - omit           → all eligible ops in both models
    """
    repo_path = _require(model_cfg, 'repo_path', 'superpoint_superglue')
    sg_weights = _require(model_cfg, 'sg_weights', 'superpoint_superglue')

    _inject_repo(repo_path, 'superpoint_superglue')

    from models.matching import Matching

    sp_cfg = dict(model_cfg.get('sp_config') or {})
    sg_cfg = dict(model_cfg.get('sg_config') or {})
    sg_cfg['weights'] = sg_weights

    return Matching({'superpoint': sp_cfg, 'superglue': sg_cfg})
