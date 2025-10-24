"""Utility classes and functions for training quantum GANs on MedMNIST datasets."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pennylane as qml
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.models import resnet18


__all__ = [
    "Discriminator",
    "PatchQuantumGenerator",
    "train_quantum_gan",
    "custom_collate_fn",
    "upscale_to_28",
    "SyntheticDataset",
    "run_experiments",
]


class Discriminator(nn.Module):
    """Simple convolution-free discriminator for flattened grayscale images."""

    def __init__(self, img_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class PatchQuantumGenerator(nn.Module):
    """Quantum patch generator that stitches patches into a full image."""

    def __init__(
        self,
        n_generators: int,
        target_img_size: int,
        *,
        n_qubits: int,
        n_a_qubits: int,
        q_depth: int,
        q_delta: float = 1.0,
        backend: str = "lightning.qubit",
        diff_method: str = "parameter-shift",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_a_qubits = n_a_qubits
        self.q_depth = q_depth
        self.target_img_size = target_img_size
        self.latent_dim = n_qubits
        self.patch_size = 2 ** (n_qubits - n_a_qubits)

        if (target_img_size**2) % self.patch_size != 0:
            raise ValueError(
                "target_img_size**2 deve ser múltiplo de patch_size para montar a imagem completa"
            )

        self.q_params = nn.ParameterList(
            [nn.Parameter(q_delta * torch.rand(q_depth * n_qubits)) for _ in range(n_generators)]
        )

        dev = qml.device(backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(noise: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            reshaped = weights.reshape(self.q_depth, self.n_qubits)

            for i in range(self.n_qubits):
                qml.RY(noise[i], wires=i)

            for layer in reshaped:
                for qubit, value in enumerate(layer):
                    qml.RY(value, wires=qubit)
                for qubit in range(self.n_qubits - 1):
                    qml.CZ(wires=[qubit, qubit + 1])

            return qml.probs(wires=list(range(self.n_qubits)))

        self._circuit = circuit

    def partial_measure(self, noise: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        probs = self._circuit(noise, weights).clone().detach()
        probs_given = probs[: self.patch_size]
        probs_given = probs_given / probs_given.sum()
        probs_given = probs_given / probs_given.max()
        return probs_given

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        device = x.device
        batch_size = x.size(0)

        patch_outputs: List[torch.Tensor] = []
        for params in self.q_params:
            patches: List[torch.Tensor] = []
            for elem in x:
                q_out = self.partial_measure(elem, params).float().unsqueeze(0).to(device)
                patches.append(q_out)
            patch_outputs.append(torch.cat(patches, dim=0))

        images = torch.cat(patch_outputs, dim=1)
        images = images.view(batch_size, 1, self.target_img_size, self.target_img_size)
        images = (images - 0.5) / 0.5
        return images


def train_quantum_gan(
    loader: DataLoader,
    generator: PatchQuantumGenerator,
    discriminator: Discriminator,
    *,
    epochs: int = 50,
    device: Optional[str] = None,
    lr_discriminator: float = 1e-2,
    lr_generator: float = 3e-1,
) -> Tuple[List[float], List[float]]:
    """Train a quantum GAN and return discriminator and generator loss histories."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    opt_d = torch.optim.SGD(discriminator.parameters(), lr=lr_discriminator)
    opt_g = torch.optim.SGD(generator.parameters(), lr=lr_generator)

    hist_d: List[float] = []
    hist_g: List[float] = []

    for _ in range(epochs):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0

        for real, _ in loader:
            real = real.to(device)
            batch_size = real.size(0)

            real_label = torch.ones((batch_size, 1), device=device)
            fake_label = torch.zeros((batch_size, 1), device=device)

            noise = torch.rand(batch_size, generator.latent_dim, device=device) * torch.pi / 2
            fake = generator(noise)

            opt_d.zero_grad()
            out_real = discriminator(real)
            out_fake = discriminator(fake.detach())
            loss_d = criterion(out_real, real_label) + criterion(out_fake, fake_label)
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            out_fake = discriminator(fake)
            loss_g = criterion(out_fake, real_label)
            loss_g.backward()
            opt_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()

        hist_d.append(epoch_loss_d / len(loader))
        hist_g.append(epoch_loss_g / len(loader))

    return hist_d, hist_g


def custom_collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    xs_tensor = torch.stack(xs, dim=0)

    labels: List[int] = []
    for y in ys:
        if isinstance(y, torch.Tensor):
            value = y.item() if y.numel() == 1 else int(y.argmax().item())
        else:
            value = int(y)
        labels.append(value)

    ys_tensor = torch.tensor(labels, dtype=torch.long)
    return xs_tensor, ys_tensor


def upscale_to_28(imgs: torch.Tensor, mode: str = "bicubic") -> torch.Tensor:
    return F.interpolate(imgs, size=(28, 28), mode=mode, align_corners=False)


class SyntheticDataset(Dataset):
    """Dataset that samples synthetic images from trained generators."""

    def __init__(
        self,
        generator_dict: Dict[str, PatchQuantumGenerator],
        num_per_class: int,
        latent_dim: int,
        device: Optional[str] = None,
    ) -> None:
        self.samples: List[Tuple[torch.Tensor, int]] = []
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        for label_name, generator in generator_dict.items():
            label = 0 if label_name == "malignant" else 1
            noise = torch.rand(num_per_class, latent_dim, device=device) * torch.pi / 2
            with torch.no_grad():
                imgs = generator(noise).cpu()
                imgs = upscale_to_28(imgs)
            for img in imgs:
                self.samples.append((img, label))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        return self.samples[idx]


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: Optional[str] = None,
) -> Tuple[float, float, float, float, float, int, int, int, int]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return acc, prec, rec, f1, auc, int(tn), int(fp), int(fn), int(tp)


def _train_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int = 5,
    device: Optional[str] = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()


def run_experiments(
    train_dataset,
    test_dataset,
    generator_dict: Dict[str, PatchQuantumGenerator],
    *,
    latent_dim: int,
    batch_size: int = 32,
    epochs: int = 5,
    device: Optional[str] = None,
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ratios: Sequence[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    results: List[Dict[str, object]] = []

    for ratio in ratios:
        if ratio == 0.0:
            dataset = train_dataset
        else:
            num_syn = int(len(train_dataset) * ratio)
            syn_ds = SyntheticDataset(generator_dict, num_syn // 2, latent_dim, device)
            dataset = ConcatDataset([train_dataset, syn_ds])

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

        model = resnet18(num_classes=2)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _train_classifier(model, loader, epochs=epochs, device=device)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        acc, prec, rec, f1, auc, tn, fp, fn, tp = _evaluate(model, test_loader, device=device)
        results.append(
            {
                "ratio": ratio,
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1,
                "auc": auc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    syn_only_ds = SyntheticDataset(generator_dict, len(train_dataset) // 2, latent_dim, device)
    syn_loader = DataLoader(
        syn_only_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    model = resnet18(num_classes=2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    _train_classifier(model, syn_loader, epochs=epochs, device=device)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    acc, prec, rec, f1, auc, tn, fp, fn, tp = _evaluate(model, test_loader, device=device)
    results.append(
        {
            "ratio": "100%_sintético→real",
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "auc": auc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    )

    total_syn = len(syn_only_ds)
    n_train = int(total_syn * 0.7)
    n_test = total_syn - n_train
    syn_train_ds, syn_test_ds = random_split(syn_only_ds, [n_train, n_test])

    train_loader = DataLoader(
        syn_train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        syn_test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    model = resnet18(num_classes=2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    _train_classifier(model, train_loader, epochs=epochs, device=device)

    acc, prec, rec, f1, auc, tn, fp, fn, tp = _evaluate(model, test_loader, device=device)
    results.append(
        {
            "ratio": "100%_sintético_70/30_selftest",
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "auc": auc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    )

    return pd.DataFrame(results)
