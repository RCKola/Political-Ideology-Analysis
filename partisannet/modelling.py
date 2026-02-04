import lightning.pytorch as L
import torch.nn as nn
import torch.optim as optim
import torch, math, logging
from transformers import get_cosine_schedule_with_warmup
from .models.loss import disp_loss, ContrastiveCenterLoss
from .eval.metrics import get_classif_meter

logging.basicConfig(level=logging.INFO)

# def disp_loss(x, tau = 0.5):
#     z = torch.flatten(x, 1)
#     dist = nn.functional.pdist(z, p=2).pow(2) / z.shape[1]
#     dist = torch.concat([dist, dist, torch.zeros(z.shape[0]).to(dist.device)]) 
#     loss = torch.logsumexp(-dist/tau, dim=0) - math.log(dist.numel())
#     return loss

class PartisanNetModel(L.LightningModule):
    def __init__(
            self, 
            model, 
            lr=1e-4,
            wd=1e-2,
            lmbda=0.5
        ):
        super(PartisanNetModel, self).__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.criterion = nn.CrossEntropyLoss()
        self.lmbda = lmbda
        self.save_hyperparameters(ignore=['model'])
        self.regularizer = ContrastiveCenterLoss(
            num_classes=model.num_classes,
            feat_dim=model.embed_dim,
            device=model.device
        )
        self.meter = get_classif_meter(num_classes=self.model.num_classes)

    def forward(self, x):
        logits, embeddings = self.model(x)
        return {
            "logits": logits,
            "embeddings": embeddings
        }

    def training_step(self, batch, batch_idx):
        sentences = batch['text']
        targets = batch['label']

        out_dict = self.forward(sentences)
        loss = self.criterion(out_dict["logits"], targets)
        reg = self.regularizer(out_dict["embeddings"], targets)
        self.log('train_loss', loss, prog_bar=True)
        self.log('reg_loss', reg, prog_bar=True)
        return loss + self.lmbda * reg

    def validation_step(self, batch, batch_idx):
        sentences = batch['text']
        targets = batch['label']
        logits = self.forward(sentences)["logits"]

        loss = self.criterion(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        sentences = batch['text']
        targets = batch['label']
        logits = self.forward(sentences)["logits"]
        probs = torch.softmax(logits, dim=1)[:, 1]

        loss = self.criterion(logits, targets)
        try:
            metrics = self.meter(probs, targets)
            self.log_dict(metrics, prog_bar=True)
        except Exception as e:
            logging.warning(f"Error computing metrics: {e}")

        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        sentences = batch['text']
        predictions = self.forward(sentences)["logits"]
        all_logits = torch.cat(predictions) 
        return all_logits.argmax(dim=1).cpu()

    def configure_optimizers(self):
        sbert_params = []
        clf_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "sbert" in name:
                sbert_params.append(param)
            else:
                clf_params.append(param)
        param_groups = [
            {"params": sbert_params, "lr": self.lr},
            {"params": clf_params, "lr": self.lr * 10},
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=self.wd)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

def save_model(trainer: L.Trainer, module: L.LightningModule, path: str = "data/fine_tuned_sbert"):
    logging.info("Loading best model weights...")
    best_path = trainer.checkpoint_callback.best_model_path
    best_model = PartisanNetModel.load_from_checkpoint(best_path, model=module.model)

    logging.info(f"Saving Fine-Tuned SBERT to path {path}")
    finetuned_sbert = best_model.model.sbert 
    finetuned_sbert.save_pretrained(path)

if __name__ == "__main__":
    from data.datamodule import get_dataloaders
    from models.classifier import SBERTClassifier
    from utils.training import setup_logger, setup_callbacks

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    log_dir = "data/outputs"
    L.seed_everything(42, workers=True)

    dataloaders, topic_model = get_dataloaders("DemRep", batch_size=128, split=True, num_topics=None, cluster_in_k=40, renew_cache=True)
    sbert_model = SBERTClassifier(lora_r=32)
    model = PartisanNetModel(sbert_model, lr=5e-5)
    logger = setup_logger(log_dir)
    logger.watch(model, log="all")

    trainer = L.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=50, 
        logger=logger,
        # callbacks=setup_callbacks(log_dir),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    trainer.test(model, dataloaders['test'], ckpt_path="best")

    save_model(trainer, model, path="data/centerloss_sbert")