from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

def setup_callbacks(log_dir: str) -> list:
    early_stop_cb = EarlyStopping(
        monitor="val_loss",  
        min_delta=0.001,
        patience=4,
        verbose=False,
        mode="min"
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{log_dir}/checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-sbert-{val_acc:.2f}"
    )
    return [early_stop_cb, checkpoint_cb]

# def setup_loggers(cfg: DictConfig) -> list[Logger]:
#     """Initialize and setup loggers."""
#     loggers: list[Logger] = [
#         hydra.utils.instantiate(logger)
#         for logger in cfg.get("loggers", dict()).values()
#     ]
#     return loggers

def setup_logger(log_dir: str, kind="wandb"):
    if kind == "wandb":
        logger = WandbLogger(
            project="PartisanNet",
            log_model="all",
            save_dir=f"{log_dir}"
        )
    else:
        raise ValueError(f"Logger kind '{kind}' not supported.")
    return logger