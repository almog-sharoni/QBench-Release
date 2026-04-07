from .base import TaskMetricsBase
from .utils import compute_certainty


class ClassificationMetrics(TaskMetricsBase):
    def __init__(self):
        self.correct_1 = 0
        self.correct_5 = 0
        self._total = 0
        self.total_certainty = 0.0

    def update(self, predictions, targets):
        # predictions: [batch, num_classes] — 2D
        batch = predictions.size(0)
        self.total_certainty += compute_certainty(predictions) * batch
        self._total += batch
        _, pred_1 = predictions.topk(1, 1, True, True)
        self.correct_1 += pred_1.eq(targets.view(-1, 1)).sum().item()
        _, pred_5 = predictions.topk(5, 1, True, True)
        self.correct_5 += pred_5.eq(targets.view(-1, 1).expand_as(pred_5)).sum().item()

    def compute(self) -> dict:
        if self._total == 0:
            raise RuntimeError("No samples processed; cannot compute metrics.")
        return {
            "acc1": 100.0 * self.correct_1 / self._total,
            "acc5": 100.0 * self.correct_5 / self._total,
            "certainty": self.total_certainty / self._total,
        }

    def metric_labels(self) -> dict[str, str]:
        return {
            "acc1": "Top-1 Acc",
            "acc5": "Top-5 Acc",
            "certainty": "Certainty",
        }

    def percentage_keys(self) -> set[str]:
        return {"acc1", "acc5"}
