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
import torch
import wandb
import yaml
import time


class Config:

    def __init__(self, config_path):

        config = parse_config(config_path)
        for key, value in config.items():
            setattr(self, key, value)

    def generate_fields(self):

        self.discriminator = build_model_from_config(self.discriminator)
        self.l_device = [int(self.device[-1])]


parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-a", "--autoencoder-config", type=str, required=True)
args = parser.parse_args()
path_to_config = args.config


config = Config(path_to_config)
config.generate_fields()
autoencoder_config = Config(args.autoencoder_config)
autoencoder = build_model_from_config(autoencoder_config.autoencoder)

def create_lightning_device(n_qubits):

    try:
        return qml.device("lightning.qubit", wires=n_qubits)
    except Exception as exc:
        raise RuntimeError(
            "Pennylane Lightning backend is required but could not be initialized. "
            "Verify that pennylane-lightning is installed and importable from the current environment."
        ) from exc


q_device = create_lightning_device(config.n_qubits)
torch.set_float32_matmul_precision("high")
seed_everything(config.random_state)


@qml.qnode(q_device, interface="torch", diff_method="backprop")
def quantum_circuit(weights):
    n_circuits = weights.size(0)
    n_qubits = weights.size(-1)

    for i in range(n_circuits):
        for q in range(n_qubits):

            qml.RY(weights[i][0, q], wires=q)
            qml.RZ(weights[i][1, q], wires=q)
            qml.RY(weights[i][2, q], wires=q)
            qml.RZ(weights[i][3, q], wires=q)

        for q in range(n_qubits):

            qml.CRY(weights[i][4, q], wires=[q, (q + 1) % n_qubits])
            qml.CRZ(weights[i][5, q], wires=[q, (q + 1) % n_qubits])

    return [qml.expval(qml.PauliX(q)) for q in range(n_qubits)] + [
        qml.expval(qml.PauliZ(q)) for q in range(n_qubits)
    ]


def partial_measure(noise, weights, n_qubits, n_rots):

    rotations = torch.stack(
        [linear(noise.unsqueeze(0)).reshape(n_rots, n_qubits) for linear in weights]
    )
    exps = quantum_circuit(rotations)
    exps = torch.stack(exps)

    return exps


class QuantumGenerator(nn.Module):

    def __init__(self, n_qubits, n_rots, n_circuits, dropout):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_rots = n_rots

        self.rot_params = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=n_qubits, out_features=n_qubits * n_rots, bias=True
                    ),
                    nn.Dropout(p=dropout),
                )
                for _ in range(n_circuits)
            ]
        )
        self.init_weights()

    def init_weights(self):

        for layer in self.rot_params:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.uniform_(layer.bias, -0.01, 0.01)

    def forward(self, x):
        device = next(self.parameters()).device

        hidden_states = [
            partial_measure(elem, self.rot_params, self.n_qubits, self.n_rots)
            for elem in x
        ]
        hidden_states = torch.stack(hidden_states).to(device)

        return hidden_states


class EpochTimingCallback(l.Callback):

    def __init__(self, total_batches):
        super().__init__()
        self.total_batches = total_batches
        self.epoch_start_time = None
        self.batch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()
        self.batch_start_time = time.perf_counter()
        if trainer.is_global_zero:
            trainer.print(
                f"Iniciando época {trainer.current_epoch + 1}/{trainer.max_epochs}..."
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.batch_start_time is None:
            return

        batch_time = time.perf_counter() - self.batch_start_time
        remaining_batches = self.total_batches - batch_idx - 1
        eta_seconds = max(0.0, remaining_batches * batch_time)

        metrics = {
            "batch_time_sec": batch_time,
            "epoch_eta_min": eta_seconds / 60,
        }

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        report_every = max(1, self.total_batches // 5)
        if trainer.is_global_zero and (batch_idx + 1) % report_every == 0:
            trainer.print(
                f"Época {trainer.current_epoch + 1}: passo {batch_idx + 1}/"
                f"{self.total_batches}, ~{eta_seconds / 60:.2f} min restantes"
            )

        self.batch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is None:
            return

        epoch_duration = time.perf_counter() - self.epoch_start_time

        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {"epoch_time_min": epoch_duration / 60}, step=trainer.global_step
            )

        if trainer.is_global_zero:
            trainer.print(
                f"Época {trainer.current_epoch + 1} finalizada em {epoch_duration / 60:.2f} min"
            )


dataset = DigitsDataset(path_to_csv=config.path_to_mnist, label=range(10))
dataloader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)
timing_callback = EpochTimingCallback(len(dataloader))

config.n_rots = 6
generator = QuantumGenerator(
    n_qubits=config.n_qubits,
    n_rots=config.n_rots,
    n_circuits=config.n_circuits,
    dropout=config.generator_dropout,
).double()
autoencoder = AutoencoderModule.load_from_checkpoint(
    checkpoint_path=config.path_to_autoencoder,
    autoencoder=autoencoder,
    optimizer=autoencoder_config.optimizers,
).double()


pushed_config = {k: v for k, v in dict(vars(config)).items() if k[:2] != "__"}
wandb_logger = WandbLogger(project="QGAN", name=config.run_name, config=pushed_config)
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

trainer = l.Trainer(
    accelerator="cuda",
    devices=config.l_device,
    max_epochs=config.epochs,
    enable_progress_bar=True,
    log_every_n_steps=config.log_every_n_steps,
    logger=wandb_logger,
    num_sanity_val_steps=0,
    fast_dev_run=config.debug,
    callbacks=[checkpoint_callback, timing_callback],
)

trainer.fit(model=gan, train_dataloaders=dataloader)
wandb.finish()
