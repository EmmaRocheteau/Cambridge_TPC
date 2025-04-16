from datetime import datetime
from pathlib import Path
import argparse
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from src.models.lightning_module import HealthcarePredictionModule
from src.models.metrics import MetricConfig, TaskType
from src.models.tpc import TempPointConv, TPCConfig
from src.data.dataset import HealthcareDataModule


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_experiment_dir(config: dict) -> Path:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_name = config["experiment"].get("name", "unnamed")
    exp_dir = Path(config["experiment"]["results_dir"]) / f"{timestamp}_{exp_name}"

    # Create directories
    (exp_dir / "checkpoints").mkdir(parents=True)
    (exp_dir / "tensorboard").mkdir(parents=True)

    # Save config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    return exp_dir


def create_callbacks(config: dict, exp_dir: Path) -> list:
    callbacks = []

    # Checkpoint callback
    if config["experiment"]["save_best"]:
        callbacks.append(
            ModelCheckpoint(
                dirpath=exp_dir / "checkpoints",
                filename="best-{epoch:02d}-{val_total_loss:.2f}",
                monitor=config["training"]["early_stopping"]["monitor"],
                mode=config["training"]["early_stopping"]["mode"],
                save_top_k=1
            )
        )

    if config["experiment"]["save_last"]:
        callbacks.append(
            ModelCheckpoint(
                dirpath=exp_dir / "checkpoints",
                filename="last",
                save_last=True
            )
        )

    # Early stopping callback
    callbacks.append(
        EarlyStopping(
            **config["training"]["early_stopping"]
        )
    )

    # Learning rate monitoring
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return callbacks


def train_with_hyperopt(config: dict, datamodule: HealthcareDataModule):
    def train_tune(tune_config: dict):
        # Update config with tuned parameters
        config["model"].update(tune_config["model"])
        config["training"].update(tune_config["training"])

        model = setup_model(config)

        trainer = Trainer(
            max_epochs=config["training"]["epochs"],
            callbacks=[TuneReportCallback(
                metrics={
                    "loss": "val_total_loss",
                    "los_msle": "val_los_prediction_msle",
                    "mortality_auroc": "val_mortality_auroc"
                },
                on="validation_end"
            )],
            logger=True
        )

        trainer.fit(model, datamodule=datamodule)

    # Define search space
    search_space = {
        "model": {
            "num_layers": tune.choice([2, 3, 4]),
            "dropout": tune.uniform(0.1, 0.5),
            "temp_dropout_rate": tune.uniform(0.1, 0.5)
        },
        "training": {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([16, 32, 64])
        }
    }

    analysis = tune.run(
        train_tune,
        config=search_space,
        num_samples=10,
        scheduler=tune.schedulers.ASHAScheduler(
            max_t=config["training"]["epochs"],
            grace_period=1,
            reduction_factor=2
        )
    )

    return analysis.best_config


def setup_model(config: dict) -> HealthcarePredictionModule:
    # Create TPC config
    tpc_config = TPCConfig(**config["model"])

    # Create TPC model
    tpc_model = TempPointConv(tpc_config)

    # Create task configs
    task_configs = {
        task_name: MetricConfig(
            task_type=TaskType(task_config["type"]),
            num_classes=task_config.get("num_classes"),
            regression_bins=task_config.get("regression_bins")
        )
        for task_name, task_config in config["tasks"].items()
    }

    return HealthcarePredictionModule(
        model=tpc_model,
        task_configs=task_configs,
        config=config
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/base_config.yaml",
                        help="Path to config file")
    parser.add_argument("--experiment-name", type=str,
                        help="Name for this experiment")
    parser.add_argument("--hyperopt", action="store_true",
                        help="Run hyperparameter optimization")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set experiment name
    config["experiment"]["name"] = args.experiment_name

    # Set random seed
    seed_everything(config["training"]["seed"])

    # Create experiment directory
    exp_dir = setup_experiment_dir(config)

    # Setup data
    datamodule = HealthcareDataModule(config["data"])

    if args.hyperopt:
        from src.utils.hyperopt_utils import train_with_hyperopt

        # Run hyperparameter optimization
        results = train_with_hyperopt(
            config=config,
            datamodule=datamodule,
            exp_dir=exp_dir
        )

        # Use best configuration
        config.update(results["best_config"])
        print("\nBest hyperparameters found:")
        print(yaml.dump(results["best_config"]))

    # Create model
    model = setup_model(config)

    if args.checkpoint:
        # Load checkpoint
        model = model.load_from_checkpoint(
            args.checkpoint,
            model=model.model,
            task_configs=model.task_configs,
            config=config
        )

    # Setup trainer
    trainer = Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=create_callbacks(config, exp_dir),
        logger=TensorBoardLogger(
            save_dir=exp_dir / "tensorboard",
            name=None,
            version=""
        ),
        gradient_clip_val=config["training"]["grad_clip"],
        deterministic=True
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    # Test model
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
