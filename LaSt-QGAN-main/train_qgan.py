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
    def __init__(self, config_path):
        config = parse_config(config_path)
        for key, value in config.items():
            setattr(self, key, value)

    def generate_fields(self):
        self.discriminator = build_model_from_config(self.discriminator)
        self.l_device = [int(self.device[-1])]


class QuantumGenerator(nn.Module):
    """
    Versão independente de argparse / config.
    O device quântico é criado dentro da própria classe.
    """

    def __init__(self, n_qubits, n_rots, n_circuits, dropout):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_rots = n_rots

        # Device do PennyLane atrelado a esse gerador
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_circuit(weights):
            n_circuits_local = weights.size(0)
            n_qubits_local = weights.size(-1)

            for i in range(n_circuits_local):
                for q in range(n_qubits_local):
                    qml.RY(weights[i][0, q], wires=q)
                    qml.RZ(weights[i][1, q], wires=q)
                    qml.RY(weights[i][2, q], wires=q)
                    qml.RZ(weights[i][3, q], wires=q)

                for q in range(n_qubits_local):
                    qml.CRY(weights[i][4, q], wires=[q, (q + 1) % n_qubits_local])
                    qml.CRZ(weights[i][5, q], wires=[q, (q + 1) % n_qubits_local])

            return [qml.expval(qml.PauliX(q)) for q in range(n_qubits_local)] + [
                qml.expval(qml.PauliZ(q)) for q in range(n_qubits_local)
            ]

        self.quantum_circuit = quantum_circuit

        self.rot_params = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=n_qubits,
                        out_features=n_qubits * n_rots,
                        bias=True,
                    ),
                    nn.Dropout(p=dropout),
                )
                for _ in range(n_circuits)
            ]
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.rot_params:
            # o Sequential tem Linear dentro; vamos inicializar o Linear
            for sub in layer:
                if isinstance(sub, nn.Linear):
                    nn.init.uniform_(sub.weight, -0.01, 0.01)
                    nn.init.uniform_(sub.bias, -0.01, 0.01)

    def partial_measure(self, noise):
        rotations = torch.stack(
            [
                linear(noise.unsqueeze(0)).reshape(self.n_rots, self.n_qubits)
                for linear in self.rot_params
            ]
        )
        exps = self.quantum_circuit(rotations)
        exps = torch.stack(exps)
        return exps

    def forward(self, x):
        device = next(self.parameters()).device
        hidden_states = [self.partial_measure(elem) for elem in x]
        hidden_states = torch.stack(hidden_states).to(device)
        return hidden_states


def run_training(config_path: str, autoencoder_config_path: str):
    """
    Função que faz TODO o processo de treino.
    Pode ser chamada tanto via main() (CLI) quanto de outro script/notebook.
    """

    config = Config(config_path)
    config.generate_fields()
    autoencoder_config = Config(autoencoder_config_path)
    base_autoencoder = build_model_from_config(autoencoder_config.autoencoder)

    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_float32_matmul_precision("high")
    seed_everything(config.random_state)

    dataset = DigitsDataset(path_to_csv=config.path_to_mnist, label=range(10))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # fixado como no seu código original
    config.n_rots = 6

    generator = QuantumGenerator(
        n_qubits=config.n_qubits,
        n_rots=config.n_rots,
        n_circuits=config.n_circuits,
        dropout=config.generator_dropout,
    ).double()

    autoencoder = AutoencoderModule.load_from_checkpoint(
        checkpoint_path=config.path_to_autoencoder,
        autoencoder=base_autoencoder,
        optimizer=autoencoder_config.optimizers,
    ).double()

    pushed_config = {
        k: v for k, v in dict(vars(config)).items() if not k.startswith("__")
    }

    wandb_logger = WandbLogger(
        project="QGAN", name=config.run_name, config=pushed_config
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./weights",
        filename="qgan-{epoch}",
        verbose=False,
        every_n_epochs=0,
        save_last=True,
    )

    gan = GANModule(
        alpha=config.alpha,
        n_qubits=config.n_qubits,
        n_rots=config.n_rots,
        autoencoder=autoencoder,
        generator=generator,
        discriminator=config.discriminator,
        optimizers_config=config.optimizers,
        step_disc_every_n_steps=config.step_disc_every_n_steps,
    ).double()

    trainer = l.Trainer(  # l vem do utils (lightning as l)
        accelerator="cuda",
        devices=config.l_device,
        max_epochs=config.epochs,
        enable_progress_bar=True,
        log_every_n_steps=config.log_every_n_steps,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        fast_dev_run=config.debug,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=gan, train_dataloaders=dataloader)
    wandb.finish()
