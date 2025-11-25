from models import WassersteinLoss, PenaltyLoss
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import lightning as l
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import models
import torch
import yaml


def type_to_parser(type):

    if type == "Sequential":
        return load_sequential
    return load_torch_module


def load_sequential(config):

    layers = []
    for layer_config in config["layers"]:
        layer = type_to_parser(layer_config["type"])(layer_config)
        layers.append(layer)

    return nn.Sequential(*layers)


def load_torch_module(config):

    try:
        layer_type = getattr(nn, config["type"])
    except:
        layer_type = getattr(models, config["type"])
    parameters = {key: value for key, value in list(config.items())[1:]}
    layer = layer_type(**parameters)
    if layer is None:
        raise ValueError(f"Wrong model type: {config['type']}")

    return layer


def build_model_from_config(config):

    return type_to_parser(config["type"])(config)


def parse_config(config_path):
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def seed_everything(seed):

    torch.backends.cudnn.deterministic = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class DigitsDataset(Dataset):

    def __init__(self, path_to_csv, label=0):

        self.df = pd.read_csv(path_to_csv)
        if label in range(0, 10):
            self.df = self.df.loc[self.df.label == label].iloc[:, 1:]
        else:
            self.df = self.df.loc[self.df.label.isin(label)].iloc[:, 1:]
        self.mean = 0
        self.std = 255

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = self.df.iloc[idx].values
        image = image.astype(np.float32).reshape(28, 28)
        image = torch.from_numpy(image)
        image = (image - self.mean) / self.std

        return image

    def generate_stats(self):

        all_data = np.empty((0, 28, 28))

        for i in range(len(self)):
            x = self[i]
            all_data = np.concatenate((all_data, x[np.newaxis, ...]))

        self.mean = all_data.mean(axis=0)
        self.std = all_data.std(axis=0)


class GANModule(l.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.criterion = WassersteinLoss()
        self.automatic_optimization = False
        self.penalty_loss = PenaltyLoss(alpha=self.alpha)
        self.register_buffer("sample_noise", self.noise(5))

    def noise(self, n):

        mean = torch.zeros((n, self.n_qubits))
        std = torch.ones((n, self.n_qubits))
        noise = torch.normal(mean, std).double()

        return noise.to(self.device)
    
    def generate(self, noise):

        hidden_states = self.generator.eval()(noise)
        samples = self.autoencoder.decode(hidden_states)

        return samples

    def draw_real_hidden_states(self, x):

        return self.autoencoder.encode(x)

    def generate_fake_hidden_states(self, x):

        return self.generator(x)

    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()

        batch_size = batch.size(0)
        x = batch.unsqueeze(1).double()

        x = self.draw_real_hidden_states(x)
        x_gen = self.generate_fake_hidden_states(self.noise(batch_size))

        loss_disc = self.criterion(
            self.discriminator(x), self.discriminator(x_gen.detach())
        )
        loss_disc = loss_disc + self.penalty_loss(x, x_gen.detach(), self.discriminator)
        if (batch_idx + 1) % self.step_disc_every_n_steps == 0:
            disc_opt.zero_grad()
            self.manual_backward(loss_disc)
            disc_opt.step()

        loss_gen = self.criterion(0, -self.discriminator(x_gen))
        gen_opt.zero_grad()
        self.manual_backward(loss_gen)
        gen_opt.step()

        self.log("Gen Loss", loss_gen.item(), prog_bar=True)
        self.log("Disc Loss", loss_disc.item(), prog_bar=True)
        self.log(
            "Gen Disc Losses Diff", (loss_gen - loss_disc).abs().item(), prog_bar=True
        )

    def on_train_epoch_end(self):
        # Skip image generation when no logger is configured (e.g., training with logger=False)
        if self.logger is None:
            return

        samples = self.generate(self.sample_noise).reshape(-1, 1, 28, 28)
        samples = samples.detach().cpu().numpy()
        samples = (samples * 255).astype(np.uint8)
        samples = [x for x in samples]
        self.logger.log_image(key="Samples Generated", images=samples)

    def configure_optimizers(self):

        gen_cfg = self.optimizers_config["generator"]
        disc_cfg = self.optimizers_config["discriminator"]
        gen_opt_type = getattr(optim, gen_cfg["type"])
        disc_opt_type = getattr(optim, disc_cfg["type"])

        gen_opt = gen_opt_type(self.generator.parameters(), **gen_cfg["parameters"])
        disc_opt = disc_opt_type(
            self.discriminator.parameters(), **disc_cfg["parameters"]
        )

        return gen_opt, disc_opt


class AutoencoderModule(l.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.automatic_optimization = True
        self.criterion = nn.MSELoss()
        self.samples = None
        self.reconstructions = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def encode(self, x):

        return self.autoencoder[0](x)

    def decode(self, x):

        return self.autoencoder[1](x)

    def training_step(self, batch):

        x = batch.unsqueeze(1)
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        loss = self.criterion(reconstructed, x)

        self.samples = x[-3:]
        self.reconstructions = reconstructed[-3:]
        self.log("Loss", loss.item(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):

        if (self.current_epoch + 1) % 10 == 0:
            samples = self.samples.detach().cpu().numpy()
            samples = (samples * 255).astype(np.uint8)
            reconstructions = self.reconstructions.detach().cpu().numpy()
            reconstructions = (reconstructions * 255).astype(np.uint8)
            samples = [x for x in samples] + [x for x in reconstructions]
            self.logger.log_image(key="Samples Reconstructed", images=samples)

    def configure_optimizers(self):

        opt_type = getattr(optim, self.optimizer["type"])
        opt = opt_type(self.autoencoder.parameters(), **self.optimizer["parameters"])

        return opt
