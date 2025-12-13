import pytorch_lightning as pl
import torch
from torch import nn
from typing import Optional, Literal
import torchmetrics
from typing import Any


class ModelLightning(pl.LightningModule):
    def __init__(
        self,
        model,
        idx_to_class: dict,
        class_weights: Optional[torch.Tensor] = None,
        lr: float = 1e-3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

        self.idx_to_class = idx_to_class

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = (
            nn.CrossEntropyLoss()
            if not class_weights
            else nn.CrossEntropyLoss(weight=class_weights)
        )

        self.train_f1 = torchmetrics.F1Score("multiclass", num_classes=6, average=None)
        self.train_recall = torchmetrics.Recall(
            "multiclass", num_classes=6, average=None
        )
        self.train_precision = torchmetrics.Precision(
            "multiclass", num_classes=6, average=None
        )
        self.val_f1 = torchmetrics.F1Score("multiclass", num_classes=6, average=None)
        self.val_recall = torchmetrics.Recall("multiclass", num_classes=6, average=None)
        self.val_precision = torchmetrics.Precision(
            "multiclass", num_classes=6, average=None
        )

        self.train_accuracy = torchmetrics.Accuracy("multiclass", num_classes=6)
        self.val_accuracy = torchmetrics.Accuracy("multiclass", num_classes=6)
        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=6)

        self.test_f1 = torchmetrics.F1Score("multiclass", num_classes=6, average=None)
        self.test_recall = torchmetrics.Recall(
            "multiclass", num_classes=6, average=None
        )
        self.test_precision = torchmetrics.Precision(
            "multiclass", num_classes=6, average=None
        )

    def forward(self, x) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)

        preds = torch.argmax(yhat, 1)
        self.train_accuracy.update(preds, y)
        self.train_f1.update(preds, y)
        self.train_recall.update(preds, y)
        self.train_precision.update(preds, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)

        preds = torch.argmax(yhat, 1)
        self.val_accuracy.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_recall.update(preds, y)
        self.val_precision.update(preds, y)
        loss = self.criterion(yhat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        preds = torch.argmax(yhat, 1)
        self.test_accuracy.update(preds, y)
        self.test_f1.update(preds, y)
        self.test_recall.update(preds, y)
        self.test_precision.update(preds, y)
        loss = self.criterion(yhat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self) -> None:
        self._shared_logging(
            mode="test",
            f1=self.test_f1,
            recall=self.test_recall,
            precision=self.test_precision,
            accuracy=self.test_accuracy,
        )

    def on_train_epoch_end(self) -> None:
        self._shared_logging(
            mode="train",
            f1=self.train_f1,
            recall=self.train_recall,
            precision=self.train_precision,
            accuracy=self.train_accuracy,
        )

    def on_validation_epoch_end(self) -> None:
        self._shared_logging(
            mode="val",
            f1=self.val_f1,
            recall=self.val_recall,
            precision=self.val_precision,
            accuracy=self.val_accuracy,
        )

    def _shared_logging(
        self,
        mode: Literal["train", "val", "test"],
        f1,
        recall,
        precision,
        accuracy,
    ):
        _f1 = f1.compute()
        _recall = recall.compute()
        _precision = precision.compute()
        _accuracy = accuracy.compute()
        self.log(f"{mode}_accuracy", _accuracy, on_epoch=True, prog_bar=True)
        # per class
        for i in range(6):
            self.log(f"{mode}_f1_class_{self.idx_to_class[i]}", _f1[i], on_epoch=True)
            self.log(
                f"{mode}_recall_class_{self.idx_to_class[i]}", _recall[i], on_epoch=True
            )
            self.log(
                f"{mode}_precision_class_{self.idx_to_class[i]}",
                _precision[i],
                on_epoch=True,
            )

        accuracy.reset()
        f1.reset()
        recall.reset()
        precision.reset()

    def configure_optimizers(self):
        return self.optimizer
