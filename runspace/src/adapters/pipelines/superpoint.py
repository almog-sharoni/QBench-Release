import os
import sys
import torch
from src.adapters.pipeline_registry import register_pipeline


def _require(cfg: dict, key: str, pipeline_name: str):
    if key not in cfg:
        raise ValueError(
            f"Pipeline '{pipeline_name}': missing required config key '{key}'. "
            f"Got keys: {sorted(cfg.keys())}"
        )
    return cfg[key]


def _inject_repo(repo_path: str, pipeline_name: str) -> None:
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Pipeline '{pipeline_name}': repo_path does not exist: {repo_path}"
        )
    if repo_path not in sys.path:
        sys.path.append(repo_path)


@register_pipeline('superpoint',
                   components={'superpoint': 'backbone'},
                   required_input_keys=('image',))
def _load_superpoint(model_cfg: dict) -> torch.nn.Module:
    repo_path = os.path.abspath(_require(model_cfg, 'repo_path', 'superpoint'))
    weights_path = os.path.abspath(_require(model_cfg, 'weights', 'superpoint'))

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Pipeline 'superpoint': weights file not found at {weights_path}"
        )

    _inject_repo(repo_path, 'superpoint')

    from models.superpoint import SuperPoint
    sp_cfg = dict(model_cfg.get('sp_config') or {})
    model = SuperPoint(sp_cfg)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    return model
