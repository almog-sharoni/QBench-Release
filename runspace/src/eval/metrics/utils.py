import torch


def compute_mse(tensor_a, tensor_b):
    """Computes Mean Squared Error between two tensors."""
    return compute_mean_pow16_error(tensor_a, tensor_b)


def compute_mean_pow16_error(tensor_a, tensor_b):
    delta = tensor_a - tensor_b
    delta_pow16 = delta.pow(16)
    return delta_pow16.mean().item()


def compute_cosine_similarity(tensor_a, tensor_b):
    """Computes Mean Cosine Similarity between two tensors (flattened)."""
    import torch.nn.functional as F
    if tensor_a.dim() > 1:
        return F.cosine_similarity(tensor_a.flatten(1), tensor_b.flatten(1)).mean().item()
    else:
        return F.cosine_similarity(tensor_a, tensor_b, dim=0).item()


def compute_min_max(tensor):
    """Computes min and max values of a tensor."""
    return tensor.min().item(), tensor.max().item()


def get_fp8_e4m3_values():
    """Returns a tensor containing all 256 valid float8_e4m3fn values."""
    if hasattr(torch, 'float8_e4m3fn'):
        all_bytes = torch.arange(256, dtype=torch.uint8)
        return all_bytes.view(torch.float8_e4m3fn).float()
    else:
        return torch.tensor([])


def check_fp8_compliance(tensor, valid_values=None):
    """
    Checks if all unique values in the tensor are present in the valid_values set.
    Returns (passed, invalid_count, invalid_examples).
    """
    if valid_values is None:
        valid_values = get_fp8_e4m3_values().to(tensor.device)

    unique_vals = tensor.unique()
    mask = torch.isin(unique_vals, valid_values)
    invalid_vals = unique_vals[~mask]

    passed = invalid_vals.numel() == 0
    invalid_count = invalid_vals.numel()
    invalid_examples = invalid_vals[:5].tolist() if not passed else []

    return passed, invalid_count, invalid_examples
