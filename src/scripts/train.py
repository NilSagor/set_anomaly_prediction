"""
Training script for SetAnomalyPredictor with YAML config and random seed support.
Usage:
    python scripts/train.py --config config/trainer/default.yaml
    python scripts/train.py --config config/trainer/default.yaml --epochs 50 --seed 123
"""

import argparse
import sys
import os
import yaml
import random
from pathlib import Path


# PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = Path(__file__).parent.parent.parent  # three levels up: src/scripts/train.py
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch    

import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.data_module import SetAnomalyDataModule
from src.lightning_module.predictor import SetAnomalyPredictor


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path, overrides=None):
    # Make config_path absolute relative to project root
    config_path = PROJECT_ROOT / config_path
    # print(f"Loading config from: {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
    return config


def main():
    parser = argparse.ArgumentParser(description="Train SetAnomalyPredictor")
    parser.add_argument("--config", type=str, default="config/trainer/default.yaml",
                        help="Path to YAML config file")
    # Override options
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--batch_size", type=int, help="Override batch_size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--num_layers", type=int, help="Override number of layers")
    parser.add_argument("--num_heads", type=int, help="Override number of heads")
    # Add more if needed
    args = parser.parse_args()

    # Overrides
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    config = load_config(args.config, overrides)
    seed = config.get('seed', 42)
    set_seed(seed)

    # Data Module
    dm = SetAnomalyDataModule(
        set_size=int(config.get('set_size', 10)),
        num_classes=int(config.get('num_classes', 10)),
        img_dim=int(config.get('img_dim', 128)),
        batch_size=int(config.get('batch_size', 32)),
        num_train_sets=int(config.get('num_train', 1000)),
        num_val_sets=int(config.get('num_val', 200)),
        num_test_sets=int(config.get('num_test', 200)),
        num_workers=int(config.get('num_workers', 4)),
        anomaly_labels=config.get('anomaly_labels'),  # can be None or list of ints
    )
    
    
    
    
    # 2. Model
    model = SetAnomalyPredictor(
        input_dim=int(config['img_dim']),
        model_dim=int(config.get('model_dim', 128)),
        num_classes=2,
        num_heads=int(config.get('num_heads', 8)),
        num_layers=int(config.get('num_layers', 2)),
        lr=float(config.get('lr', 1e-3)),
        dropout=float(config.get('dropout', 0.1)),
        input_dropout=float(config.get('input_dropout', 0.0)),
    )
    
    # 3. Callbacks
    checkpoint_dir = PROJECT_ROOT / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"set-anomaly-seed{seed}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )
    
    # 4. Logger (TensorBoard)
    # Logger
    log_dir = PROJECT_ROOT / "results" / "logs"
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="set_anomaly",
        version=f"seed_{seed}",
    )
    
    
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config.get('epochs', 20),
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        precision="32",
    )
    
    
    trainer.fit(model, dm)

    
    trainer.test(model, dm)

if __name__ == "__main__":
    main()