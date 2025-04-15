import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.lightning_module import LitModel

def train_model(data, config):
    print("Training model with PyTorch Lightning...")

    # Dummy data placeholder (replace with real dataset)
    import torch
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=config.batch_size)

    model = LitModel(input_dim=10, output_dim=2, learning_rate=config.learning_rate)

    logger = TensorBoardLogger(save_dir="logs", name="CambridgeTPC")
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader)
    model.save_model(config.model_path)
    return model