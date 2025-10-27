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
from sklearn.decomposition import PCA
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.models import resnet18

import numpy as np


__all__ = [
    "Discriminator",
    "PatchQuantumGenerator",
    "train_quantum_gan",
    "custom_collate_fn",
    "upscale_to_28",
    "SyntheticDataset",
    "run_experiments",
    "MosaiqDiscriminator",
    "MosaiqQuantumGenerator",
    "train_mosaiq_gan",
    "scale_data",
    "prepare_mosaiq_pca_data",
    "create_mosaiq_pca_loaders",
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


def scale_data(
    data: np.ndarray,
    scale: Sequence[float] = (-1.0, 1.0),
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Scale the input array to the provided range.

    Parameters
    ----------
    data:
        Array containing the values to be scaled.
    scale:
        Two-element sequence with the minimum and maximum of the desired range.
    dtype:
        Target dtype of the returned array.
    """

    if len(scale) != 2:
        raise ValueError("scale must contain exactly two elements: (min, max)")

    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(dtype)

    mn, mx = float(arr.min()), float(arr.max())
    a, b = float(scale[0]), float(scale[1])

    if np.isclose(mx, mn):
        return np.full_like(arr, (a + b) / 2.0, dtype=dtype)

    scaled = (arr - mn) / (mx - mn)
    scaled = scaled * (b - a) + a
    return scaled.astype(dtype)


def _extract_dataset_images(dataset) -> torch.Tensor:
    if hasattr(dataset, "imgs"):
        base_imgs = dataset.imgs
    elif hasattr(dataset, "data"):
        base_imgs = dataset.data
    else:
        raise AttributeError("Dataset must expose an 'imgs' or 'data' attribute with the images")

    if isinstance(base_imgs, torch.Tensor):
        imgs_tensor = base_imgs.clone().float()
    else:
        imgs_tensor = torch.as_tensor(base_imgs).float()

    if imgs_tensor.ndim == 4 and imgs_tensor.shape[-1] == 1:
        imgs_tensor = imgs_tensor.permute(0, 3, 1, 2)
    elif imgs_tensor.ndim == 3:
        imgs_tensor = imgs_tensor.unsqueeze(1)

    if imgs_tensor.max() > 1:
        imgs_tensor = imgs_tensor / 255.0

    return imgs_tensor


def prepare_mosaiq_pca_data(
    dataset,
    *,
    target_size: int = 8,
    pca_components: int = 40,
) -> Tuple[torch.Tensor, torch.Tensor, PCA]:
    """Return PCA-compressed tensors for the MOSAIQ GAN pipeline.

    The dataset is resized to ``target_size`` before being flattened and compressed
    with Principal Component Analysis. The resulting tensor is already scaled to
    ``[-1, 1]`` so it can be directly consumed by the MOSAIQ discriminator.
    """

    imgs_tensor = _extract_dataset_images(dataset)
    lowres_tensor = F.interpolate(
        imgs_tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
    )

    flat_imgs = lowres_tensor.reshape(lowres_tensor.size(0), -1).cpu().numpy()
    scaled_inputs = scale_data(flat_imgs, (0.0, 1.0))

    pca = PCA(n_components=pca_components)
    pca_data = pca.fit_transform(scaled_inputs)
    scaled_pca = scale_data(pca_data)
    tensor_pca = torch.from_numpy(scaled_pca).float()

    if hasattr(dataset, "labels"):
        labels = torch.as_tensor(dataset.labels).view(-1)
    elif hasattr(dataset, "targets"):
        labels = torch.as_tensor(dataset.targets).view(-1)
    else:
        raise AttributeError("Dataset must expose 'labels' or 'targets' with the class ids")

    labels = labels.to(torch.long)
    return tensor_pca, labels, pca


def create_mosaiq_pca_loaders(
    dataset,
    *,
    batch_size: int,
    target_size: int = 8,
    pca_dims: int = 40,
    drop_last: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> Tuple[Dict[int, DataLoader], torch.Tensor, torch.Tensor, PCA]:
    """Build per-class dataloaders with PCA-compressed samples for MOSAIQ training."""

    tensor_pca, labels, pca = prepare_mosaiq_pca_data(
        dataset, target_size=target_size, pca_components=pca_dims
    )

    loaders: Dict[int, DataLoader] = {}
    unique_labels = torch.unique(labels).tolist()

    for label in sorted(int(l) for l in unique_labels):
        indices = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        subset = tensor_pca[indices]
        loaders[label] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    return loaders, tensor_pca, labels, pca


class MosaiqDiscriminator(nn.Module):
    """Feed-forward discriminator operating on PCA-compressed vectors."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class MosaiqQuantumGenerator(nn.Module):
    """Quantum generator that assembles a MOSAIQ-style latent representation."""

    def __init__(
        self,
        n_generators: int,
        n_qubits: int,
        q_depth: int,
        *,
        q_delta: float = 1.0,
        backend: str = "lightning.qubit",
        diff_method: str = "parameter-shift",
    ) -> None:
        super().__init__()
        self.n_generators = n_generators
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.latent_dim = n_qubits
        self.output_dim = n_generators * n_qubits
        self._circuit_device = torch.device("cpu")

        self.q_params = nn.ParameterList(
            [nn.Parameter(q_delta * torch.rand(q_depth, n_qubits)) for _ in range(n_generators)]
        )

        dev = qml.device(backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(noise: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            for q in range(self.n_qubits):
                qml.RY(noise[q], wires=q)
                qml.RX(noise[q], wires=q)

            for layer in weights:
                for qubit, value in enumerate(layer):
                    qml.RY(value, wires=qubit)
                for qubit in range(self.n_qubits - 1):
                    qml.CZ(wires=[qubit, qubit + 1])

            return [qml.expval(qml.PauliX(q)) for q in range(self.n_qubits)]

        self._circuit = circuit

    def to(self, *args, **kwargs):  # type: ignore[override]
        obj = super().to(*args, **kwargs)
        for param in self.q_params:
            if param.device != self._circuit_device:
                param.data = param.data.to(self._circuit_device)
        return obj

    def forward(self, noise: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if noise.dim() != 2 or noise.size(1) != self.n_qubits:
            raise ValueError(
                f"Expected noise with shape (batch, {self.n_qubits}), got {tuple(noise.shape)}"
            )

        orig_device = noise.device
        noise_cpu = noise.to(self._circuit_device)
        batch_size = noise_cpu.size(0)
        outputs = torch.zeros(batch_size, self.output_dim, device=self._circuit_device)

        for gen_idx, params in enumerate(self.q_params):
            params_cpu = params
            patches: List[torch.Tensor] = []
            for sample in noise_cpu:
                circuit_out = self._circuit(sample, params_cpu)
                if isinstance(circuit_out, (list, tuple)):
                    circuit_out = torch.stack(tuple(circuit_out))
                if not isinstance(circuit_out, torch.Tensor):
                    circuit_out = torch.tensor(circuit_out)
                circuit_out = circuit_out.to(self._circuit_device)
                patches.append(circuit_out)
            patch_tensor = torch.stack(patches, dim=0)
            start = gen_idx * self.n_qubits
            outputs[:, start : start + self.n_qubits] = patch_tensor

        return outputs.to(orig_device).float()


def _ensure_tensor_batch(batch: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if not batch:
        raise ValueError("Received an empty batch from the dataloader")
    if isinstance(batch, (list, tuple)):
        first = batch[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError("Expected the dataloader to yield tensors or sequences of tensors")


def train_mosaiq_gan(
    loader: DataLoader,
    generator: MosaiqQuantumGenerator,
    discriminator: MosaiqDiscriminator,
    *,
    epochs: int = 50,
    device: Optional[str | torch.device] = None,
    lr_discriminator: float = 1e-2,
    lr_generator: float = 3e-1,
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
    """Train the MOSAIQ quantum GAN on PCA-compressed features."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    discriminator = discriminator.to(device)
    generator = generator.to(device)
    generator.train()
    discriminator.train()

    criterion = nn.BCELoss()
    opt_d = torch.optim.SGD(discriminator.parameters(), lr=lr_discriminator)
    opt_g = torch.optim.SGD(generator.parameters(), lr=lr_generator)

    hist_d: List[float] = []
    hist_g: List[float] = []

    for epoch in range(epochs):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0

        for batch in loader:
            real = _ensure_tensor_batch(batch).to(device)
            batch_size = real.size(0)

            real_label = torch.ones((batch_size, 1), device=device)
            fake_label = torch.zeros((batch_size, 1), device=device)

            noise = torch.rand(batch_size, generator.latent_dim, device=device) * (torch.pi / 2)
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

        avg_d = epoch_loss_d / max(len(loader), 1)
        avg_g = epoch_loss_g / max(len(loader), 1)
        hist_d.append(avg_d)
        hist_g.append(avg_g)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: D={avg_d:.4f}, G={avg_g:.4f}")

    return hist_d, hist_g
