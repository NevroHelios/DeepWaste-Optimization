
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import torch

from src.models.pretrained import get_pretrained_model
from src.lightning.model import ModelLightning
from src.data.module import DataLoaderWrapper


def train():
    wandb.init()
    config = wandb.config

    dm = DataLoaderWrapper(
        data_dir=Path(config.data_dir),
        img_size=config.img_size,
        mode="augment",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    freeze_strategy = config.freeze_strategy
    freeze_layers = 0
    
    if freeze_strategy == 'freeze_backbone_half':
        temp_model = get_pretrained_model(config.num_classes, 'finetune_all')
        num_blocks = len(list(temp_model.features.children()))
        freeze_layers = int(num_blocks * 0.5)
        freeze_strategy = 'freeze_k_layers'
        del temp_model

    model = get_pretrained_model(
        num_classes=config.num_classes,
        freeze_strategy=freeze_strategy,
        freeze_layers=freeze_layers
    )

    backbone_lr = None
    head_params = None
    
    if config.use_differential_lr:
        backbone_lr = config.lr * config.backbone_lr_factor
        head_params = list(model.classifier.parameters())

    pl_model = ModelLightning(
        model=model,
        lr=config.lr,
        idx_to_class=dm.idx_to_class,
        backbone_lr=backbone_lr,
        head_params=head_params
    )

    logger = WandbLogger(
        project="garbage-classification-finetune-v1",
        name=f"ft-mobilenetv3small-{config.freeze_strategy}",
        config=config,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="ft_best_model_{epoch:02d}_{val_accuracy:.4f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        verbose=False,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(pl_model, datamodule=dm)
    logger.experiment.finish()
