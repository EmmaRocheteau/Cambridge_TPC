from typing import Dict, Optional, List
import torch
import torchmetrics
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"


@dataclass
class MetricConfig:
    task_type: TaskType
    num_classes: Optional[int] = None
    class_weights: Optional[torch.Tensor] = None
    threshold: float = 0.5
    regression_bins: Optional[List[float]] = None


class HealthcareMetrics:
    """
    A comprehensive healthcare metrics calculator supporting multiple task types.
    Handles regression, binary classification, and multiclass classification tasks.
    """

    def __init__(self, config: MetricConfig):
        self.config = config
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize task-specific metrics"""
        if self.config.task_type == TaskType.REGRESSION:
            self.metrics = self._init_regression_metrics()
        elif self.config.task_type == TaskType.BINARY:
            self.metrics = self._init_binary_metrics()
        else:  # MULTICLASS
            self.metrics = self._init_multiclass_metrics()

    def _init_regression_metrics(self) -> torchmetrics.MetricCollection:
        """Initialize regression metrics"""
        metrics = {
            'mse': torchmetrics.MeanSquaredError(),
            'rmse': torchmetrics.MeanSquaredError(squared=False),
            'mae': torchmetrics.MeanAbsoluteError(),
            'r2': torchmetrics.R2Score(),
            'msle': CustomMSLE(),
            'mape': CustomMAPE(epsilon=4 / 24)  # Minimum denominator of 4 hours
        }

        if self.config.regression_bins is not None:
            metrics['binned_accuracy'] = BinnedAccuracy(self.config.regression_bins)
            metrics['cohen_kappa'] = torchmetrics.CohenKappa(
                num_classes=len(self.config.regression_bins) + 1,
                weights='linear'
            )

        return torchmetrics.MetricCollection(metrics)

    def _init_binary_metrics(self) -> torchmetrics.MetricCollection:
        """Initialize binary classification metrics"""
        return torchmetrics.MetricCollection({
            'auroc': torchmetrics.AUROC(task='binary'),
            'auprc': torchmetrics.AveragePrecision(task='binary'),
            'accuracy': torchmetrics.Accuracy(task='binary', threshold=self.config.threshold),
            'balanced_accuracy': CustomBalancedAccuracy(threshold=self.config.threshold),
            'f1': torchmetrics.F1Score(task='binary', threshold=self.config.threshold),
            'precision': torchmetrics.Precision(task='binary', threshold=self.config.threshold),
            'recall': torchmetrics.Recall(task='binary', threshold=self.config.threshold),
            'specificity': Specificity(threshold=self.config.threshold)
        })

    def _init_multiclass_metrics(self) -> torchmetrics.MetricCollection:
        """Initialize multiclass classification metrics"""
        return torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(
                task='multiclass',
                num_classes=self.config.num_classes
            ),
            'balanced_accuracy': CustomBalancedAccuracy(
                num_classes=self.config.num_classes
            ),
            'f1': torchmetrics.F1Score(
                task='multiclass',
                num_classes=self.config.num_classes,
                average='macro'
            ),
            'precision': torchmetrics.Precision(
                task='multiclass',
                num_classes=self.config.num_classes,
                average='macro'
            ),
            'recall': torchmetrics.Recall(
                task='multiclass',
                num_classes=self.config.num_classes,
                average='macro'
            )
        })

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update all metrics with new predictions and targets"""
        self.metrics.update(preds, targets)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute and return all metrics"""
        return self.metrics.compute()

    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.reset()


class CustomMSLE(torchmetrics.Metric):
    """Mean Squared Logarithmic Error"""

    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_log_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds, targets = self._input_format(preds, targets)
        log_diff = torch.log1p(targets) - torch.log1p(preds)
        squared_log_error = torch.square(log_diff)
        self.sum_squared_log_error += torch.sum(squared_log_error)
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_squared_log_error / self.total


class CustomMAPE(torchmetrics.Metric):
    """Mean Absolute Percentage Error with minimum denominator threshold"""

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.add_state("sum_abs_percentage_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds, targets = self._input_format(preds, targets)
        denominator = torch.maximum(
            torch.tensor(self.epsilon, device=targets.device),
            torch.abs(targets)
        )
        percentage_error = torch.abs((targets - preds) / denominator) * 100
        self.sum_abs_percentage_error += torch.sum(percentage_error)
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_abs_percentage_error / self.total


class BinnedAccuracy(torchmetrics.Metric):
    """Accuracy when predictions are binned into ranges"""

    def __init__(self, bins: List[float]):
        super().__init__()
        self.register_buffer("bins", torch.tensor(bins))
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_binned = torch.bucketize(preds, self.bins)
        targets_binned = torch.bucketize(targets, self.bins)
        self.correct += torch.sum(preds_binned == targets_binned)
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total


class Specificity(torchmetrics.Metric):
    """True Negative Rate / Specificity metric"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state("true_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = (preds >= self.threshold).float()
        true_negatives = torch.sum((preds == 0) & (targets == 0))
        false_positives = torch.sum((preds == 1) & (targets == 0))
        self.true_negatives += true_negatives
        self.false_positives += false_positives

    def compute(self) -> torch.Tensor:
        return self.true_negatives / (self.true_negatives + self.false_positives)


class CustomBalancedAccuracy(torchmetrics.Metric):
    """Balanced accuracy that handles both binary and multiclass cases"""

    def __init__(self, threshold: float = 0.5, num_classes: Optional[int] = None):
        super().__init__()
        self.threshold = threshold
        self.num_classes = num_classes

        if num_classes is None:  # Binary case
            self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("true_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total_positives", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        else:  # Multiclass case
            self.add_state("correct_per_class",
                           default=torch.zeros(num_classes),
                           dist_reduce_fx="sum")
            self.add_state("total_per_class",
                           default=torch.zeros(num_classes),
                           dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if self.num_classes is None:  # Binary case
            preds = (preds >= self.threshold).float()
            self.true_positives += torch.sum((preds == 1) & (targets == 1))
            self.true_negatives += torch.sum((preds == 0) & (targets == 0))
            self.total_positives += torch.sum(targets == 1)
            self.total_negatives += torch.sum(targets == 0)
        else:  # Multiclass case
            preds_cls = torch.argmax(preds, dim=1)
            for i in range(self.num_classes):
                class_mask = targets == i
                self.correct_per_class[i] += torch.sum((preds_cls == i) & class_mask)
                self.total_per_class[i] += torch.sum(class_mask)

    def compute(self) -> torch.Tensor:
        if self.num_classes is None:  # Binary case
            sensitivity = self.true_positives / self.total_positives
            specificity = self.true_negatives / self.total_negatives
            return (sensitivity + specificity) / 2
        else:  # Multiclass case
            per_class_acc = self.correct_per_class / self.total_per_class
            return torch.mean(per_class_acc)
