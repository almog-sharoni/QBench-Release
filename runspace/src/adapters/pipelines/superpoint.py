import os
import sys
import torch
from src.adapters.pipeline_registry import register_pipeline


@register_pipeline('superpoint', components={'superpoint': 'backbone'})
def _load_superpoint(model_cfg: dict) -> torch.nn.Module:
    repo_path = os.path.abspath(model_cfg['repo_path'])
    weights_path = os.path.abspath(model_cfg['weights'])

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from models.superpoint import SuperPoint
    model = SuperPoint({})
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    return model
