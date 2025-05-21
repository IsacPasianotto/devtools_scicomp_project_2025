import torch
import torch.nn as nn
import torchmetrics
import lightning.pytorch as pl

class Classifier(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        # Accuracy metrics for training, validation, and testing
        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.model.num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.model.num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.model.num_classes)

    def _classifier_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        loss, true_labels, predicted_labels = self._classifier_step(batch)
        self.train_accuracy(predicted_labels, true_labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        loss, true_labels, predicted_labels = self._classifier_step(batch)
        self.val_accuracy(predicted_labels, true_labels)
        self.log("val_loss", loss)
        self.log(
            "val_acc",
            self.val_accuracy,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, _):
        _, true_labels, predicted_labels = self._classifier_step(batch)
        self.test_accuracy(predicted_labels, true_labels)
        self.log("test_acc", self.test_accuracy, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
