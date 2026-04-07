import torch
from .base import TaskMetricsBase


class LanguageModelMetrics(TaskMetricsBase):
    def __init__(self):
        self.total_loss = 0.0
        self.total_tokens = 0

    def update(self, predictions, targets):
        import torch.nn.functional as F
        # predictions: [batch, seq_len, vocab] — 3D
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        loss = F.cross_entropy(flat_logits, flat_labels, reduction='sum')
        self.total_loss += loss.item()
        self.total_tokens += flat_labels.numel()

    def compute(self) -> dict:
        if self.total_tokens == 0:
            raise RuntimeError("No tokens processed; cannot compute metrics.")
        import math
        return {"ppl": math.exp(self.total_loss / self.total_tokens)}

    def metric_labels(self) -> dict[str, str]:
        return {"ppl": "PPL"}

    def percentage_keys(self) -> set[str]:
        return set()

    def compute_certainty(self, predictions: torch.Tensor) -> float:
        import torch.nn.functional as F
        # predictions: [batch, seq_len, vocab_size]
        probs = F.softmax(predictions, dim=2)
        max_probs, _ = probs.max(dim=2)  # [batch, seq_len]
        return max_probs.mean().item()
