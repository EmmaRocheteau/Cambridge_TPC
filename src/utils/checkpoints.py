from pytorch_lightning.callbacks import ModelCheckpoint

def get_checkpoint_callback(checkpoint_dir="checkpoints", monitor="val_loss"):
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        verbose=True,
        monitor=monitor,
        mode="min"
    )