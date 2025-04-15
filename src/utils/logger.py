import logging
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def get_tensorboard_logger(base_log_dir="logs", name="experiment"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{base_log_dir}/{name}_{timestamp}"
    return TensorBoardLogger(save_dir=base_log_dir, name=f"{name}_{timestamp}")