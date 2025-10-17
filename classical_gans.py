"""Classical GAN architectures and training utilities for MedMNIST experiments."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class DCGenerator(nn.Module):
    """Standard DCGAN generator for 28x28 grayscale images."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 1) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DCDiscriminator(nn.Module):
    """Standard DCGAN discriminator for 28x28 grayscale images."""

    def __init__(self, img_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_gan_for_class(
    train_loader,
    label_target: int,
    G: nn.Module,
    D: nn.Module,
    latent_dim: int = 100,
    num_epochs: int = 50,
    device: Optional[str] = None,
) -> nn.Module:
    """Train a DCGAN conditioned on a specific class from the dataloader."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    G.train()
    D.train()

    for epoch in range(num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        batches_processed = 0

        for imgs, labels in train_loader:
            mask = labels.squeeze() == label_target
            if mask.sum() == 0:
                continue

            real = imgs[mask].to(device)
            b_size = real.size(0)
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)

            optim_D.zero_grad()
            fake = G(noise)
            loss_real = criterion(D(real), torch.ones(b_size, 1, device=device))
            loss_fake = criterion(D(fake.detach()), torch.zeros(b_size, 1, device=device))
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            optim_G.zero_grad()
            output = D(fake)
            loss_G = criterion(output, torch.ones(b_size, 1, device=device))
            loss_G.backward()
            optim_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            batches_processed += 1

        if batches_processed > 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss D: {epoch_loss_D / batches_processed:.3f} "
                f"| Loss G: {epoch_loss_G / batches_processed:.3f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Nenhum batch da classe {label_target} processado."
            )

    return G


def train_gan_for_class_with_loss(
    train_loader,
    label_target: int,
    G: nn.Module,
    D: nn.Module,
    latent_dim: int = 100,
    num_epochs: int = 50,
    device: Optional[str] = None,
) -> Tuple[nn.Module, list[float], list[float]]:
    """Same as :func:`train_gan_for_class` but also returns loss histories."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    G.train()
    D.train()

    hist_D: list[float] = []
    hist_G: list[float] = []

    for epoch in range(num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        batches = 0

        for imgs, labels in train_loader:
            mask = labels.squeeze() == label_target
            if mask.sum() == 0:
                continue

            real = imgs[mask].to(device)
            b_size = real.size(0)
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)

            optim_D.zero_grad()
            fake = G(noise)
            loss_real = criterion(D(real), torch.ones(b_size, 1, device=device))
            loss_fake = criterion(D(fake.detach()), torch.zeros(b_size, 1, device=device))
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            optim_G.zero_grad()
            output = D(fake)
            loss_G = criterion(output, torch.ones(b_size, 1, device=device))
            loss_G.backward()
            optim_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            batches += 1

        if batches > 0:
            hist_D.append(epoch_loss_D / batches)
            hist_G.append(epoch_loss_G / batches)
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss D: {hist_D[-1]:.3f} | Loss G: {hist_G[-1]:.3f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Nenhum batch da classe {label_target} processado."
            )

    return G, hist_D, hist_G


class CGANGenerator(nn.Module):
    """Conditional GAN generator using label embeddings."""

    def __init__(self, latent_dim: int = 100, num_classes: int = 2, img_channels: int = 1) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = 7
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * self.init_size * self.init_size)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise.squeeze(), label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class CGANDiscriminator(nn.Module):
    """Conditional GAN discriminator with label concatenation."""

    def __init__(self, num_classes: int = 9, img_channels: int = 3) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels + num_classes, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 4, 1)

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_img = (
            self.label_emb(labels)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, img.size(2), img.size(3))
        )
        x = torch.cat([img, label_img], 1)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


def train_cgan(
    train_loader,
    G: nn.Module,
    D: nn.Module,
    latent_dim: int,
    num_classes: int,
    num_epochs: int,
    device: Optional[str],
    label_target: int,
) -> nn.Module:
    """Train a conditional GAN for a specific target class."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    G.train()
    D.train()

    for epoch in range(num_epochs):
        for imgs, labels in train_loader:
            mask = labels.squeeze() == label_target
            if mask.sum() == 0:
                continue
            real = imgs[mask].to(device)
            labels_real = labels[mask].squeeze().long().to(device)
            b_size = real.size(0)

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            gen_labels = torch.full((b_size,), label_target, device=device, dtype=torch.long)

            optim_D.zero_grad()
            fake = G(noise, gen_labels)
            loss_real = criterion(D(real, labels_real), torch.ones(b_size, 1, device=device))
            loss_fake = criterion(D(fake.detach(), gen_labels), torch.zeros(b_size, 1, device=device))
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            optim_G.zero_grad()
            output = D(fake, gen_labels)
            loss_G = criterion(output, torch.ones(b_size, 1, device=device))
            loss_G.backward()
            optim_G.step()

    return G


class WGANGPGenerator(DCGenerator):
    """WGAN-GP generator sharing architecture with the DCGAN generator."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 1) -> None:
        super().__init__(latent_dim=latent_dim, img_channels=img_channels)


class WGANGPCritic(nn.Module):
    """Critic network for WGAN-GP."""

    def __init__(self, img_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Compute the gradient penalty for WGAN-GP."""

    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)
    grad_outputs = torch.ones_like(mixed_scores)

    gradients = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_wgangp(
    train_loader,
    G: nn.Module,
    D: nn.Module,
    latent_dim: int,
    num_epochs: int,
    device: Optional[str],
    label_target: Optional[int] = None,
    critic_iters: int = 5,
    lambda_gp: float = 10.0,
) -> nn.Module:
    """Train a WGAN-GP model optionally conditioned on a label."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    optim_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optim_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    G.train()
    D.train()

    for epoch in range(num_epochs):
        for imgs, labels in train_loader:
            if label_target is not None:
                mask = labels.squeeze() == label_target
                if mask.sum() == 0:
                    continue
                real = imgs[mask].to(device)
            else:
                real = imgs.to(device)

            b_size = real.size(0)
            if b_size == 0:
                continue

            for _ in range(critic_iters):
                noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake = G(noise)

                D_real = D(real).view(-1)
                D_fake = D(fake.detach()).view(-1)

                gp = gradient_penalty(D, real, fake)
                loss_D = -(D_real.mean() - D_fake.mean()) + lambda_gp * gp

                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = G(noise)
            loss_G = -D(fake).view(-1).mean()

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}"
        )

    return G


__all__ = [
    "DCGenerator",
    "DCDiscriminator",
    "train_gan_for_class",
    "train_gan_for_class_with_loss",
    "CGANGenerator",
    "CGANDiscriminator",
    "train_cgan",
    "WGANGPGenerator",
    "WGANGPCritic",
    "gradient_penalty",
    "train_wgangp",
]
