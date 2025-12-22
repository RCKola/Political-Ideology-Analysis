import lightning as L
import torch.nn as nn
import torch.optim as optim
import torch, math
from transformers import get_cosine_schedule_with_warmup

def disp_loss(x, tau = 0.5):
    z = torch.flatten(x, 1)
    dist = nn.functional.pdist(z, p=2).pow(2) / z.shape[1]

    # Accounts for zero distance to self
    dist = torch.concat([dist, dist, torch.zeros(z.shape[0]).to(dist.device)]) 

    # Log sum exp trick for numerical stability
    loss = torch.logsumexp(-dist/tau, dim=0) - math.log(dist.numel())
    return loss

class PartisanNetModel(L.LightningModule):
    def __init__(
            self, 
            model, 
            learning_rate=1e-3,
            lmbda=0.5
        ):
        super(PartisanNetModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.lmbda = lmbda

    def forward(self, x):
        logits, embeddings = self.model(x)
        return {
            "logits": logits,
            "embeddings": embeddings
        }

    def training_step(self, batch, batch_idx):
        sentences = batch['text']
        targets = batch['label']

        out_dict = self.model(sentences)
        loss = self.criterion(out_dict["logits"], targets)
        disp = disp_loss(out_dict["embeddings"])
        self.log('train_loss', loss, prog_bar=True)
        return loss + self.lmbda * disp

    def validation_step(self, batch, batch_idx):
        sentences = batch['text']
        targets = batch['label']
        logits = self.model(sentences)[0]

        loss = self.criterion(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        self.log('val_accuracy', accuracy)
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        sentences = batch['text'] # Extract just the list of strings
        return self.model(sentences)[0] # Pass only the strings to the SBERT model

    def configure_optimizers(self):
        # 1. Define the Optimizer (AdamW is best for Transformers)
        optimizer = optim.AdamW(self.parameters(), lr=2e-5, weight_decay=0.01)

        # 2. Calculate Total Training Steps
        # (batches_per_epoch * max_epochs)
        # self.trainer.estimated_stepping_batches is a handy PL shortcut
        total_steps = self.trainer.estimated_stepping_batches

        # 3. Define the Scheduler
        # 10% warmup is standard
        warmup_steps = int(0.1 * total_steps) 
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 4. Return dictionary format for Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Update the LR every batch, not every epoch
                "frequency": 1
            }
        }

if __name__ == "__main__":
    from partisannet.data.datamodule import get_dataloaders
    from partisannet.models.classifier import SBERTClassifier
    from lightning.pytorch.callbacks import EarlyStopping

    early_stop_callback = EarlyStopping(
    monitor="val_loss",  # Watch the validation loss
    min_delta=0.001,      # Improvement must be at least this much
    patience=4,          # Stop if no improvement for 3 epochs
    verbose=True,
    mode="min"           # "min" because we want loss to go DOWN
    )


    dataloaders, topic_model = get_dataloaders("LibCon", batch_size=32, split=True, num_topics=None, cluster_in_k=40, renew_cache=False)
    sbert_model = SBERTClassifier()
    model = PartisanNetModel(sbert_model)
    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=50, callbacks=[early_stop_callback])
    trainer.fit(model, dataloaders['train'], dataloaders['val'])

    predictions = trainer.predict(model, dataloaders['test'])
    print("Predictions on test set completed.")
    all_logits = torch.cat(predictions) 
    predicted_classes = all_logits.argmax(dim=1).cpu()

    # 2. Extract the TRUE labels from the test dataloader (in the same order)
    true_labels = []
    for batch in dataloaders['test']:
        true_labels.append(batch['label'])
        
    true_labels = torch.cat(true_labels).cpu()

    # 3. Calculate REAL Accuracy
    # (Check where Prediction equals Truth)
    correct = (predicted_classes == true_labels).float()
    real_accuracy = correct.mean().item()

    print(f"REAL Test Accuracy: {real_accuracy:.4f}")
        



    print("Loading best model weights...")
    # 1. Load the best checkpoint found by EarlyStopping
    best_path = trainer.checkpoint_callback.best_model_path
    # We reload the weights into your existing wrapper
    best_lightning_model = PartisanNetModel.load_from_checkpoint(best_path, model=sbert_model)

    print("Saving Fine-Tuned SBERT to disk...")
    # 2. Extract ONLY the inner SBERT (the "feature extractor")
    # We access: LightningModule -> SBERTClassifier -> SentenceTransformer
    finetuned_sbert = best_lightning_model.model.sbert 

    # 3. Save it as a standard SentenceTransformer
    # This allows you to load it later with one line of code
    finetuned_sbert.save("data/fine_tuned_sbert")

