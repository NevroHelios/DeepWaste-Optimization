import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from src.models.simple_cnn import SimpleCNN
from src.lightning.model import ModelLightning
from src.data.module import DataLoaderWrapper


def train():
    wandb.init()
    config = wandb.config

    dm = DataLoaderWrapper(
        data_dir=Path(config.data_dir),
        img_size=config.img_size,
        mode="augment" if config.use_augmentation else None,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = SimpleCNN(
        num_layers=config.num_layers,
        filter_size=config.filter_size,
        num_dense=config.num_dense,
        conv_activation=config.conv_activation,
        num_classes=config.num_classes,
        dense_activation=config.dense_activation,
        in_channels=config.in_channels,
        stride=config.stride,
        input_size=config.input_size,
        padding=config.padding,
        dropout=config.dropout if config.dropout > 0 else None,
        strategy=config.strategy,
        include_batchnorm=config.include_batchnorm,
        pooling_k=config.pooling_k,
        base_features=config.base_features,
    )

    pl_model = ModelLightning(
        model=model,
        lr=config.lr,
        idx_to_class=dm.idx_to_class,
    )

    logger = WandbLogger(
        project="garbage-classification-v5",
        name="simple-cnn-v0",
        config=config,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best_model_{epoch:02d}_{val_accuracy:.4f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        verbose=False,
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(pl_model, datamodule=dm)
    logger.experiment.finish()
