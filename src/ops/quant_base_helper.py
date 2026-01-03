def _get_reduce_dims(input: torch.Tensor):
    if input.dim() == 4: # [N, C, H, W] -> reduce N, H, W
        reduce_dims = (0, 2, 3)
    elif input.dim() == 2: # [N, C] -> reduce N
        reduce_dims = (0,)
    else:
        # Fallback: global scaling or assume N is dim 0
        # Let's assume per-channel means dim 1 is channel
        # If input is [N, C, L], reduce (0, 2)
        reduce_dims = tuple(d for d in range(input.dim()) if d != 1)
    return reduce_dims
