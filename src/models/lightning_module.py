from typing import Dict, Any, List
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import json
import os

from src.models.metrics import HealthcareMetrics, MetricConfig, TaskType


class HealthcarePredictionModule(pl.LightningModule):
    """
    PyTorch Lightning module for healthcare prediction tasks.
    Supports multiple prediction tasks with customizable metrics.
    """

    def __init__(
            self,
            model: nn.Module,
            task_configs: Dict[str, MetricConfig],
            config: Dict[str, Any]
    ):
        super().__init__()
        self.model = model
        self.task_configs = task_configs
        self.config = config
        self.save_hyperparameters(config)

        # Initialize metrics for each task
        self.metrics = {
            phase: {
                task_name: HealthcareMetrics(task_config)
                for task_name, task_config in task_configs.items()
            }
            for phase in ['train', 'val', 'test']
        }

        # Initialize loss functions
        self.loss_fns = self._initialize_loss_functions()

    def _initialize_loss_functions(self) -> Dict[str, nn.Module]:
        """Initialize task-specific loss functions"""
        loss_fns = {}
        for task_name, task_config in self.task_configs.items():
            if task_config.task_type == TaskType.REGRESSION:
                loss_fns[task_name] = nn.MSELoss()
            elif task_config.task_type == TaskType.BINARY:
                if task_config.class_weights is not None:
                    loss_fns[task_name] = nn.BCEWithLogitsLoss(
                        pos_weight=task_config.class_weights
                    )
                else:
                    loss_fns[task_name] = nn.BCEWithLogitsLoss()
            else:  # MULTICLASS
                if task_config.class_weights is not None:
                    loss_fns[task_name] = nn.CrossEntropyLoss(
                        weight=task_config.class_weights
                    )
                else:
                    loss_fns[task_name] = nn.CrossEntropyLoss()
        return loss_fns

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def _shared_step(
            self,
            batch: tuple,
            batch_idx: int,
            phase: str
    ) -> Dict[str, torch.Tensor]:
        """Shared step for training, validation and test"""
        x, y_dict = batch
        y_pred_dict = self(x)

        # Calculate loss for each task
        losses = {}
        for task_name, y_pred in y_pred_dict.items():
            y_true = y_dict[task_name]
            losses[f"{task_name}_loss"] = self.loss_fns[task_name](y_pred, y_true)

            # Update metrics
            self.metrics[phase][task_name].update(y_pred, y_true)

        # Calculate total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        # Log losses
        self.log_dict(
            {f"{phase}_{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0)
        )

        return losses

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        losses = self._shared_step(batch, batch_idx, "train")
        return losses['total_loss']

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "test")

    def _shared_epoch_end(self, phase: str) -> Dict[str, float]:
        """Compute and log metrics at epoch end"""
        metrics_dict = {}

        # Compute metrics for each task
        for task_name, task_metrics in self.metrics[phase].items():
            task_results = task_metrics.compute()

            # Log each metric
            for metric_name, value in task_results.items():
                metric_key = f"{phase}_{task_name}_{metric_name}"
                self.log(metric_key, value, prog_bar=True)
                metrics_dict[metric_key] = value.item()

            # Reset metrics for next epoch
            task_metrics.reset()

        return metrics_dict

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._shared_epoch_end("train")

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._shared_epoch_end("val")

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        metrics = self._shared_epoch_end("test")
        self.save_experiment(metrics)

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0)
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.config['training']['scheduler_patience'],
            factor=0.5,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss"
            }
        }

    def save_experiment(self, metrics: Dict[str, float]) -> None:
        """Save experiment results and configuration"""
        save_dir = os.path.join(
            self.config['logging']['log_dir'],
            self.config['logging']['experiment_name']
        )
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        # Save metrics
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
