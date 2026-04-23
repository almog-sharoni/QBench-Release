import math
import torch
import torch.nn.functional as F


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

        # Check for sequence output (SLM)
        if predictions.dim() == 3:
            shift_logits = predictions[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            loss = F.cross_entropy(flat_logits, flat_labels, reduction='sum')
            self.total_loss += loss.item()
            self.total_tokens += flat_labels.numel()
            predictions = flat_logits
            targets = flat_labels

        self.total_certainty += compute_certainty(predictions) * predictions.size(0)
        self.total += predictions.size(0)

        if targets is None:
            return

        _, pred_1 = predictions.topk(1, 1, True, True)
        self.correct_1 += pred_1.eq(targets.view(-1, 1)).sum().item()

        _, pred_5 = predictions.topk(5, 1, True, True)
        self.correct_5 += pred_5.eq(targets.view(-1, 1).expand_as(pred_5)).sum().item()

    def compute(self):
        if self.total == 0:
            return {"acc1": 0.0, "acc5": 0.0, "certainty": 0.0, "ppl": 0.0}

        acc1 = 100.0 * self.correct_1 / self.total
        acc5 = 100.0 * self.correct_5 / self.total
        certainty = self.total_certainty / self.total
        ppl = math.exp(self.total_loss / self.total_tokens) if self.total_tokens > 0 else 0.0

        return {"acc1": acc1, "acc5": acc5, "certainty": certainty, "ppl": ppl}


def compute_mse(tensor_a, tensor_b):
    return compute_mean_pow16_error(tensor_a, tensor_b)


def compute_mean_pow16_error(tensor_a, tensor_b):
    delta = tensor_a - tensor_b
    delta_pow16 = delta.pow(16)
    return delta_pow16.mean().item()


def compute_cosine_similarity(tensor_a, tensor_b):
    if tensor_a.dim() > 1:
        return F.cosine_similarity(tensor_a.flatten(1), tensor_b.flatten(1)).mean().item()
    else:
        return F.cosine_similarity(tensor_a, tensor_b, dim=0).item()


def compute_certainty(predictions):
    probs = F.softmax(predictions, dim=1)
    max_probs, _ = probs.max(dim=1)
    return max_probs.mean().item()


def compute_min_max(tensor):
    return tensor.min().item(), tensor.max().item()


def check_fp8_compliance(tensor, valid_values):
    if valid_values is None:
        raise ValueError(
            "check_fp8_compliance requires an explicit valid_values table. "
            "For simulated fp formats, use _check_mantissa_precision instead."
        )
    unique_vals = tensor.unique()
    mask = torch.isin(unique_vals, valid_values)
    invalid_vals = unique_vals[~mask]
    passed = invalid_vals.numel() == 0
    invalid_count = invalid_vals.numel()
    invalid_examples = invalid_vals[:5].tolist() if not passed else []
    return passed, invalid_count, invalid_examples


from .feature_matching import FeatureMatchingMetrics  # noqa: F401
from .matching import MatchingMetrics                  # noqa: F401
