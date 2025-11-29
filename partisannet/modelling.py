import lightning as L
import torch.nn as nn
import torch.optim as optim


class PartisanNetModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(PartisanNetModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sentences = batch['text']
        targets = batch['label']

        outputs = self.model(sentences)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sentences = batch['text']
        targets = batch['label']
        logits = self.model(sentences)

        loss = self.criterion(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        self.log('val_accuracy', accuracy)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == "__main__":
    from data.datamodule import get_dataloaders
    from models.classifier import SBERTClassifier

    dataloaders = get_dataloaders("mbib-base", batch_size=32)
    sbert_model = SBERTClassifier()
    model = PartisanNetModel(sbert_model)
    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model, dataloaders['train'], dataloaders['val'])

