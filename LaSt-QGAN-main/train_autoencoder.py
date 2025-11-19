from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import torch.optim as optim
import pennylane as qml
from tqdm import tqdm
import torch.nn as nn
from utils import *
import numpy as np
import warnings
import torch
import wandb
import yaml


class Config:

    def __init__(
            self,
            config_path
            ):
        
        config = parse_config(config_path)
        for key, value in config.items():
            setattr(self, key, value)

    def generate_fields(
            self
            ):

        self.autoencoder = build_model_from_config(self.autoencoder)
        self.l_device = [int(self.device[-1])]


parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()
path_to_config = args.config


config = Config(path_to_config)
config.generate_fields()
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('high')
seed_everything(config.random_state)


dataset = DigitsDataset(
    path_to_csv=config.path_to_mnist,
    label=range(10)
)
if config.batch_size < 0:
    config.batch_size *= -len(dataset)
    config.batch_size = int(config.batch_size)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size, 
    shuffle=True,
    pin_memory=True
)
if config.log_every_n_steps == -1:
    config.log_every_n_steps = len(dataloader)


pushed_config = {k: v for k, v in dict(vars(config)).items() if k[:2] != "__"}
wandb_logger = WandbLogger(
    project="QGAN",
    name=config.run_name,
    config=pushed_config
)
module = AutoencoderModule(
    autoencoder=config.autoencoder,
    optimizer=config.optimizers
)
checkpoint_callback = ModelCheckpoint(
    dirpath='./weights',
    save_last=False,
    filename='autoencoder-{epoch:02d}_epoch-{Loss:.4f}_loss',
    verbose=False,
    every_n_epochs=1,
    save_top_k=1,
    monitor="Loss"
)

trainer = l.Trainer(
    accelerator="cuda",
    devices=config.l_device,
    max_epochs=config.epochs,
    enable_progress_bar=True,
    log_every_n_steps=config.log_every_n_steps,
    logger=wandb_logger,
    num_sanity_val_steps=0,
    fast_dev_run=config.debug,
    callbacks=[checkpoint_callback]
)

trainer.fit(model=module, train_dataloaders=dataloader)
wandb.finish()
