class MetricsEngine:
    """
    Computes accuracy, certainty drift, and optional histograms.
    """
    def __init__(self):
        self.correct_1 = 0
        self.correct_5 = 0
        self.total = 0
        self.total_certainty = 0.0
        self.total_loss = 0.0
        self.total_tokens = 0

    def update(self, predictions, targets):
        # predictions: [batch, num_classes] OR [batch, seq_len, num_classes]
        # targets: [batch] OR [batch, seq_len]
        
        import torch.nn.functional as F
        
        # Check for sequence output (SLM)
        if predictions.dim() == 3:
            # Shift logits and labels for next token prediction
            # Logits: [..., :-1, :]
            # Labels: [..., 1:]
            shift_logits = predictions[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            # Flatten
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            # Calculate Loss (CrossEntropy) for Perplexity
            # We need to accumulate total loss and total tokens
            loss = F.cross_entropy(flat_logits, flat_labels, reduction='sum')
            self.total_loss += loss.item()
            self.total_tokens += flat_labels.numel()
            
            # Use flattened for accuracy
            predictions = flat_logits
            targets = flat_labels
        
        # Certainty
        self.total_certainty += compute_certainty(predictions) * predictions.size(0)
        self.total += predictions.size(0)
        
        if targets is None:
            return

        # Top-1
        _, pred_1 = predictions.topk(1, 1, True, True)
        self.correct_1 += pred_1.eq(targets.view(-1, 1)).sum().item()
        
        # Top-5
        _, pred_5 = predictions.topk(5, 1, True, True)
        self.correct_5 += pred_5.eq(targets.view(-1, 1).expand_as(pred_5)).sum().item()

    def compute(self):
        if self.total == 0:
            return {"acc1": 0.0, "acc5": 0.0, "certainty": 0.0, "ppl": 0.0}
            
        acc1 = 100.0 * self.correct_1 / self.total
        acc5 = 100.0 * self.correct_5 / self.total
        certainty = self.total_certainty / self.total
        
        import math
        ppl = math.exp(self.total_loss / self.total_tokens) if self.total_tokens > 0 else 0.0
        
        return {"acc1": acc1, "acc5": acc5, "certainty": certainty, "ppl": ppl}

def compute_mse(tensor_a, tensor_b):
    """Computes Mean Squared Error between two tensors."""
    # import torch.nn.functional as F
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

def compute_certainty(predictions):
    """Computes mean certainty (max softmax probability) of predictions."""
    import torch.nn.functional as F
    probs = F.softmax(predictions, dim=1)
    max_probs, _ = probs.max(dim=1)
    return max_probs.mean().item()

def compute_min_max(tensor):
    """Computes min and max values of a tensor."""
    return tensor.min().item(), tensor.max().item()

def get_fp8_e4m3_values():
    """Returns a tensor containing all 256 valid float8_e4m3fn values."""
    import torch
    if hasattr(torch, 'float8_e4m3fn'):
        # Generate all 256 byte patterns
        all_bytes = torch.arange(256, dtype=torch.uint8)
        # View as float8_e4m3fn and cast to float
        return all_bytes.view(torch.float8_e4m3fn).float()
    else:
        # Fallback if float8 not supported (should not happen in this env)
        return torch.tensor([])

def check_fp8_compliance(tensor, valid_values=None):
    """
    Checks if all unique values in the tensor are present in the valid_values set.
    Returns (passed, invalid_count, invalid_examples).
    """
    if valid_values is None:
        valid_values = get_fp8_e4m3_values().to(tensor.device)
    
    unique_vals = tensor.unique()
    
    # Check membership
    # Using isin if available (torch >= 1.10)
    import torch
    mask = torch.isin(unique_vals, valid_values)
    invalid_vals = unique_vals[~mask]
    
    passed = invalid_vals.numel() == 0
    invalid_count = invalid_vals.numel()
    invalid_examples = invalid_vals[:5].tolist() if not passed else []
    
    return passed, invalid_count, invalid_examples
