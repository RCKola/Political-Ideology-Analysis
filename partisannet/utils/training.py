from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

def setup_callbacks():
    early_stop_cb = EarlyStopping(
        monitor="val_loss",  
        min_delta=0.001,
        patience=4,
        verbose=True,
        mode="min"
    )
    return [early_stop_cb]

# def setup_loggers(cfg: DictConfig) -> list[Logger]:
#     """Initialize and setup loggers."""
#     loggers: list[Logger] = [
#         hydra.utils.instantiate(logger)
#         for logger in cfg.get("loggers", dict()).values()
#     ]
#     return loggers

def setup_logger(kind="wandb"):
    if kind == "wandb":
        logger = WandbLogger(
            project="PartisanNet",
            name="ideology-classification",
            log_model="all",
            save_dir="data",
            resume="auto"
        )
    else:
        raise ValueError(f"Logger kind '{kind}' not supported.")
    return logger