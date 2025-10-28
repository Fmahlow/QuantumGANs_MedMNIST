"""Classical GAN architectures and training loops for MedMNIST experiments.

This module centralises the neural network definitions and helper training
functions that were previously scattered across the
``gan_classical_medmnist.ipynb`` notebook.  Having the implementations in a
Python module keeps the notebook concise while still allowing reuse across
experiments.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor, autograd
from torch.utils.data import DataLoader, Subset, TensorDataset

__all__ = [
    "DCGenerator",
    "DCDiscriminator",
    "CGANGenerator",
    "CGANDiscriminator",
    "WGANGPGenerator",
    "WGANGPCritic",
    "train_gan_for_class",
    "train_gan_for_class_with_loss",
    "train_cgan",
    "train_wgangp",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    """Return a ``torch.device`` using CUDA when available by default."""

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _zero_dim_noise(noise: Tensor) -> Tensor:
    """Flatten convolutional-style noise tensors to ``(batch, latent_dim)``."""

    if noise.dim() <= 2:
        return noise
    return noise.view(noise.size(0), -1)


def _single_class_loader(
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    label_target: int,
) -> DataLoader:
    """Return a ``DataLoader`` that only yields samples from ``label_target``.

    ``train_gan_for_class`` originally filtered the batches on the fly.  When the
    target class is the minority, this leads to tiny batch sizes (often of size
    one), making the batch-normalisation layers inside the generators unstable
    and hurting convergence.  Preparing a dedicated loader up-front gives us
    consistently sized batches and matches the behaviour from the original
    notebook where each class had its own loader.
    """

    base_batch_size = getattr(train_loader, "batch_size", None)
    if base_batch_size is None or base_batch_size <= 0:
        base_batch_size = getattr(train_loader, "_batch_size", None)
    if base_batch_size is None or base_batch_size <= 0:
        base_batch_size = 64

    dataset = getattr(train_loader, "dataset", None)
    indices: Optional[List[int]] = None

    loader_kwargs = {
        "num_workers": getattr(train_loader, "num_workers", 0),
        "pin_memory": getattr(train_loader, "pin_memory", False),
    }
    if loader_kwargs["num_workers"] > 0 and hasattr(train_loader, "persistent_workers"):
        loader_kwargs["persistent_workers"] = getattr(
            train_loader, "persistent_workers", False
        )
    generator = getattr(train_loader, "generator", None)
    if generator is not None:
        loader_kwargs["generator"] = generator

    if dataset is not None:
        labels_attr = None
        for attr in ("targets", "labels", "y"):
            if hasattr(dataset, attr):
                labels_attr = getattr(dataset, attr)
                break

        if labels_attr is not None:
            labels_tensor = torch.as_tensor(labels_attr)
            if labels_tensor.dim() > 1:
                labels_tensor = labels_tensor.squeeze()
            indices_tensor = (labels_tensor == label_target).nonzero(as_tuple=False).view(-1)
            if indices_tensor.numel() > 0:
                indices = indices_tensor.tolist()

    if indices:
        subset = Subset(dataset, indices)
        batch_size = min(base_batch_size, len(indices))
        drop_last = len(indices) >= 2 and len(indices) % batch_size == 1
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            **loader_kwargs,
        )

    class_images: List[Tensor] = []
    for imgs, labels in train_loader:
        mask = labels.squeeze() == label_target
        if mask.any():
            class_images.append(imgs[mask])

    if not class_images:
        raise ValueError(
            f"Nenhuma amostra da classe {label_target} encontrada no train_loader."
        )

    images = torch.cat(class_images, dim=0)
    labels_tensor = torch.full((images.size(0), 1), label_target, dtype=torch.long)
    dataset = TensorDataset(images, labels_tensor)
    batch_size = min(base_batch_size, len(dataset))
    drop_last = len(dataset) >= 2 and len(dataset) % batch_size == 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        **loader_kwargs,
    )


# ---------------------------------------------------------------------------
# DCGAN
# ---------------------------------------------------------------------------


class DCGenerator(nn.Module):
    """Generator architecture used for the DCGAN experiments."""

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

    def forward(self, noise: Tensor, labels: Optional[Tensor] = None) -> Tensor:  # noqa: D401 - short docstring
        """Generate images from ``noise`` ignoring optional class ``labels``.

        Some balancing utilities call generators with a ``labels`` argument
        regardless of whether the underlying architecture is conditional.  The
        original ``DCGenerator`` only accepted noise which raised a ``TypeError``
        when reused in those generic pipelines.  Accepting ``labels`` and
        discarding it keeps backwards compatibility while allowing the
        unconditional generator to plug into the shared code path.
        """

        del labels  # explicitly ignore optional labels for unconditional GANs
        return self.model(noise)


class DCDiscriminator(nn.Module):
    """Discriminator counterpart for the DCGAN experiments."""

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

    def forward(self, image: Tensor) -> Tensor:  # noqa: D401 - short docstring
        features = self.features(image)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Conditional GAN (CGAN)
# ---------------------------------------------------------------------------


class CGANGenerator(nn.Module):
    """Conditional GAN generator supporting arbitrary class counts."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 2,
        img_channels: int = 1,
    ) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = 7  # suitable for 28x28 outputs
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

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        label_input = self.label_emb(labels)
        flattened_noise = _zero_dim_noise(noise)
        gen_input = torch.cat((flattened_noise, label_input), dim=-1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        return self.conv_blocks(out)


class CGANDiscriminator(nn.Module):
    """Conditional GAN discriminator that incorporates label embeddings."""

    def __init__(
        self,
        num_classes: int = 2,
        img_channels: int = 1,
    ) -> None:
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

    def forward(self, image: Tensor, labels: Tensor) -> Tensor:
        label_img = (
            self.label_emb(labels)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, image.size(2), image.size(3))
        )
        x = torch.cat([image, label_img], dim=1)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Wasserstein GAN with Gradient Penalty (WGAN-GP)
# ---------------------------------------------------------------------------


class WGANGPGenerator(DCGenerator):
    """Generator used for the WGAN-GP setup (inherits from ``DCGenerator``)."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 1) -> None:
        super().__init__(latent_dim=latent_dim, img_channels=img_channels)


class WGANGPCritic(nn.Module):
    """Critic network for the WGAN-GP experiments."""

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

    def forward(self, image: Tensor) -> Tensor:  # noqa: D401 - short docstring
        x = self.conv(image)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.fc(x)


def _gradient_penalty(critic: nn.Module, real: Tensor, fake: Tensor) -> Tensor:
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)
    grad_outputs = torch.ones_like(mixed_scores)

    gradients = autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_gan_for_class(
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    label_target: int,
    G: nn.Module,
    D: nn.Module,
    latent_dim: int = 100,
    num_epochs: int = 50,
    device: Optional[torch.device | str] = None,
) -> nn.Module:
    """Train a DCGAN for a specific target class and return the generator."""

    device = _resolve_device(device)
    data_loader = _single_class_loader(train_loader, label_target)
    criterion = nn.BCELoss()
    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    G.train()
    D.train()

    for epoch in range(num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        batches_processed = 0

        for real, _ in data_loader:
            real = real.to(device)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

            optim_D.zero_grad()
            fake = G(noise)
            loss_real = criterion(D(real), torch.ones(batch_size, 1, device=device))
            loss_fake = criterion(
                D(fake.detach()), torch.zeros(batch_size, 1, device=device)
            )
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            optim_G.zero_grad()
            output = D(fake)
            loss_G = criterion(output, torch.ones(batch_size, 1, device=device))
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
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    label_target: int,
    G: nn.Module,
    D: nn.Module,
    latent_dim: int = 100,
    num_epochs: int = 50,
    device: Optional[torch.device | str] = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train a DCGAN while tracking discriminator and generator losses."""

    device = _resolve_device(device)
    data_loader = _single_class_loader(train_loader, label_target)
    criterion = nn.BCELoss()
    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    G.train()
    D.train()

    hist_D: List[float] = []
    hist_G: List[float] = []

    for epoch in range(num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        batches = 0

        for real, _ in data_loader:
            real = real.to(device)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

            optim_D.zero_grad()
            fake = G(noise)
            loss_real = criterion(D(real), torch.ones(batch_size, 1, device=device))
            loss_fake = criterion(
                D(fake.detach()), torch.zeros(batch_size, 1, device=device)
            )
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            optim_G.zero_grad()
            output = D(fake)
            loss_G = criterion(output, torch.ones(batch_size, 1, device=device))
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


def train_cgan(
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    G: nn.Module,
    D: nn.Module,
    latent_dim: int,
    num_classes: int,
    num_epochs: int,
    device: Optional[torch.device | str] = None,
    label_target: Optional[int] = None,
) -> nn.Module:
    """Train a conditional GAN; optionally filter to a single target label."""

    device = _resolve_device(device)
    criterion = nn.BCELoss()
    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    G.train()
    D.train()

    data_loader = (
        _single_class_loader(train_loader, label_target)
        if label_target is not None
        else train_loader
    )

    for epoch in range(num_epochs):
        for imgs, labels in data_loader:
            real = imgs.to(device)
            if label_target is not None:
                labels_real = torch.full(
                    (real.size(0),), label_target, device=device, dtype=torch.long
                )
            else:
                labels_real = labels.squeeze().long().to(device)

            batch_size = real.size(0)
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            gen_labels = (
                torch.full((batch_size,), label_target, device=device, dtype=torch.long)
                if label_target is not None
                else labels_real
            )

            optim_D.zero_grad()
            fake = G(noise, gen_labels)
            loss_real = criterion(
                D(real, labels_real), torch.ones(batch_size, 1, device=device)
            )
            loss_fake = criterion(
                D(fake.detach(), gen_labels), torch.zeros(batch_size, 1, device=device)
            )
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            optim_G.zero_grad()
            output = D(fake, gen_labels)
            loss_G = criterion(output, torch.ones(batch_size, 1, device=device))
            loss_G.backward()
            optim_G.step()

    return G


def train_wgangp(
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    G: nn.Module,
    D: nn.Module,
    latent_dim: int,
    num_epochs: int,
    device: Optional[torch.device | str] = None,
    label_target: Optional[int] = None,
    critic_iters: int = 5,
    lambda_gp: float = 10.0,
) -> nn.Module:
    """Train a WGAN-GP on the provided ``train_loader`` and return the generator."""

    device = _resolve_device(device)
    optim_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optim_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    G.train()
    D.train()

    data_loader = (
        _single_class_loader(train_loader, label_target)
        if label_target is not None
        else train_loader
    )

    for epoch in range(num_epochs):
        for imgs, _ in data_loader:
            real = imgs.to(device)

            batch_size = real.size(0)
            if batch_size == 0:
                continue

            for _ in range(critic_iters):
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake = G(noise)

                D_real = D(real).view(-1)
                D_fake = D(fake.detach()).view(-1)

                gp = _gradient_penalty(D, real, fake)
                loss_D = -(D_real.mean() - D_fake.mean()) + lambda_gp * gp

                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake = G(noise)
            loss_G = -D(fake).view(-1).mean()

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}"
        )

    return G
