import os
from typing import Dict, Any
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger


def train_with_hyperopt(
        model_class,
        config: Dict[str, Any],
        datamodule,
        num_samples: int = 10
):
    """Run hyperparameter optimization using Ray Tune"""

    def train_tune(config: Dict[str, Any]):
        model = model_class(config)
        trainer = pl.Trainer(
            max_epochs=config["training"]["epochs"],
            callbacks=[TuneReportCallback(
                metrics={
                    "loss": "val_loss",
                    "accuracy": "val_accuracy"
                },
                on="validation_end"
            )],
            logger=True
        )
        trainer.fit(model, datamodule=datamodule)

    # Define search space
    search_space = {
        "model": {
            "hidden_dim": tune.choice([64, 128, 256]),
            "num_layers": tune.choice([2, 3, 4]),
            "dropout": tune.uniform(0.1, 0.5)
        },
        "training": {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([16, 32, 64])
        }
    }

    analysis = tune.run(
        train_tune,
        config=config,
        num_samples=num_samples,
        search_alg=tune.suggest.hyperopt.HyperOptSearch(),
        scheduler=tune.schedulers.ASHAScheduler(
            max_t=config["training"]["epochs"],
            grace_period=1,
            reduction_factor=2
        )
    )

    return analysis


def train_model(
        model,
        config: Dict[str, Any],
        datamodule,
        experiment_name: str
):
    """Train a single model with given configuration"""

    # Setup logging
    logger = TensorBoardLogger(
        save_dir=config["logging"]["log_dir"],
        name=experiment_name
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(config["logging"]["save_dir"], experiment_name),
            filename="best-model",
            monitor="val_loss",
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"],
            mode="min"
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config["training"]["grad_clip"],
        deterministic=True
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    # Test model
    test_results = trainer.test(model, datamodule=datamodule)

    # Save experiment results
    model.save_experiment(test_results[0])

    return model, test_results[0]
