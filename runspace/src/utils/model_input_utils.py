from typing import Optional, Tuple


def _extract_input_size_from_cfg(cfg) -> Optional[Tuple[int, int, int]]:
    if cfg is None:
        return None

    input_size = None
    if isinstance(cfg, dict):
        input_size = cfg.get("input_size")
    else:
        input_size = getattr(cfg, "input_size", None)

    if not input_size or len(input_size) != 3:
        return None

    try:
        c, h, w = (int(v) for v in input_size)
    except Exception:
        return None
    return c, h, w


def resolve_model_input_size(model, batch_size: int = 1) -> Tuple[int, int, int, int]:
    """Resolve a model's preferred input size, falling back to ImageNet defaults."""
    for attr in ("default_cfg", "pretrained_cfg"):
        size = _extract_input_size_from_cfg(getattr(model, attr, None))
        if size is not None:
            c, h, w = size
            return int(batch_size), c, h, w

    return int(batch_size), 3, 224, 224
