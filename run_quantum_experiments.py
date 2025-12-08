"""Pipeline unificado para treinar e avaliar os geradores quânticos.

O módulo espelha o fluxo definido em ``run_classical_experiments.py`` para
produzir as mesmas tabelas de saída (qualidade, eficiência e impacto na
classificação), mas aplicando as rotinas das arquiteturas PatchQGAN e MOSAIQ
documentadas nos notebooks ``gans_quantum_resources.ipynb``,
``gans_quantum_fid_is.ipynb`` e ``mosaiq_resources.ipynb``.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import models
from torchvision import transforms

from medmnist_data import load_medmnist_data
from quantum_gan_medmnist import (
    Discriminator,
    MosaiqDiscriminator,
    MosaiqQuantumGenerator,
    PatchQuantumGenerator,
    create_mosaiq_pca_loaders,
    scale_data,
    train_mosaiq_gan,
    train_quantum_gan,
)


class ProgressTracker:
    """Estimativa simples de tempo restante com base em etapas concluídas."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.completed = 0
        self.accumulated = 0.0

    def step(self, elapsed: float) -> None:
        self.completed += 1
        self.accumulated += elapsed
        avg = self.accumulated / max(self.completed, 1)
        remaining = max(self.total_steps - self.completed, 0) * avg
        print(
            f"[ETA] {self.completed}/{self.total_steps} etapas completas. "
            f"Tempo médio {avg:.1f}s | Estimativa restante {remaining:.1f}s",
            flush=True,
        )


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunConfig:
    data_flag: str = "breastmnist"
    latent_dim: int = 5
    img_channels: int = 1
    num_classes: int = 2
    gan_epochs: int = 50
    clf_epochs: int = 3
    batch_size: int = 128
    mosaiq_batch_size: int = 8
    num_workers: int = 0
    repeats: int = 1
    synth_ratio_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5)
    balance_ratio: float = 0.5
    target_img_size: int = 8
    n_a_qubits: int = 1
    q_depth: int = 6
    pca_dims: int = 40
    output_dir: Path = Path("experiments_outputs")
    qml_backend: str = "lightning.qubit"
    qml_diff_method: str = "parameter-shift"
    qml_batch_obs: Optional[int] = None
    qml_mpi: bool = False
    qml_circuit_device: Optional[str] = None


@dataclass
class MosaiqPCA:
    loaders: Dict[int, DataLoader]
    pca_data: torch.Tensor
    labels: torch.Tensor
    mean_: np.ndarray
    min_: float
    max_: float
    pca_min_: float
    pca_max_: float
    input_min_: float
    input_max_: float
    pca_model: object


@dataclass
class DataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    train_dataset: Dataset
    test_dataset: Dataset
    label_loaders: Dict[int, DataLoader]
    mosaiq: Optional[MosaiqPCA] = None

    @property
    def label_names(self) -> Dict[int, str]:
        dataset = getattr(self.train_dataset, "_dataset", self.train_dataset)
        if hasattr(dataset, "info") and "label" in dataset.info:
            return {int(k): v for k, v in dataset.info["label"].items()}
        return {0: "class0", 1: "class1"}



def timed(fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return wrapper


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_inference_latency(generator: nn.Module, cfg: RunConfig, device: torch.device) -> float:
    generator.eval()
    batch_size = 8
    gen_device = next(generator.parameters()).device
    noise = torch.rand(batch_size, cfg.latent_dim, device=gen_device) * (torch.pi / 2)

    with torch.no_grad():
        start = time.perf_counter()
        generator(noise)
        if gen_device.type == "cuda":
            torch.cuda.synchronize(gen_device)
    return (time.perf_counter() - start) / batch_size


def build_patch_models(cfg: RunConfig) -> Tuple[nn.Module, nn.Module]:
    patch_size = 2 ** (cfg.latent_dim - cfg.n_a_qubits)
    n_generators = (cfg.target_img_size ** 2) // patch_size
    device_kwargs: Dict[str, object] = {}
    if cfg.qml_batch_obs is not None:
        device_kwargs["batch_obs"] = cfg.qml_batch_obs
    if cfg.qml_mpi:
        device_kwargs["mpi"] = True
    generator = PatchQuantumGenerator(
        n_generators,
        cfg.target_img_size,
        n_qubits=cfg.latent_dim,
        n_a_qubits=cfg.n_a_qubits,
        q_depth=cfg.q_depth,
        backend=cfg.qml_backend,
        diff_method=cfg.qml_diff_method,
        circuit_device=cfg.qml_circuit_device,
        device_kwargs=device_kwargs,
    )
    discriminator = Discriminator(img_size=cfg.target_img_size)
    return generator, discriminator


def build_mosaiq_models(cfg: RunConfig) -> Tuple[nn.Module, nn.Module]:
    if cfg.pca_dims % cfg.latent_dim != 0:
        raise ValueError(
            "MOSAIQ requer que pca_dims seja divisível por latent_dim para alinhar as dimensões"
        )

    n_generators = cfg.pca_dims // cfg.latent_dim
    device_kwargs: Dict[str, object] = {}
    if cfg.qml_batch_obs is not None:
        device_kwargs["batch_obs"] = cfg.qml_batch_obs
    if cfg.qml_mpi:
        device_kwargs["mpi"] = True
    generator = MosaiqQuantumGenerator(
        n_generators,
        cfg.latent_dim,
        cfg.q_depth,
        backend=cfg.qml_backend,
        diff_method=cfg.qml_diff_method,
        circuit_device=cfg.qml_circuit_device,
        device_kwargs=device_kwargs,
    )
    discriminator = MosaiqDiscriminator(input_dim=cfg.pca_dims)
    return generator, discriminator


def create_label_image_loaders(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    drop_last: bool = False,
    shuffle: bool = True,
) -> Dict[int, DataLoader]:
    """Create per-label dataloaders mirroring the notebook's class-wise training."""

    labels = torch.as_tensor(getattr(dataset, "labels")).view(-1).long()
    unique_labels = torch.unique(labels).tolist()

    loaders: Dict[int, DataLoader] = {}
    for label in sorted(int(l) for l in unique_labels):
        indices = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        subset = Subset(dataset, indices.tolist())
        loaders[label] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    return loaders


def prepare_data(cfg: RunConfig) -> DataBundle:
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.target_img_size, cfg.target_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    med_data = load_medmnist_data(
        data_flag=cfg.data_flag,
        batch_size=cfg.batch_size,
        transform=transform,
        shuffle_train=True,
        shuffle_test=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    label_loaders = create_label_image_loaders(
        med_data.train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
    )

    # MOSAIQ usa PCA sobre as imagens já reescaladas
    dataset_class = med_data.train_dataset.__class__
    highres_train = dataset_class(split="train", transform=transform, download=True)
    (
        loaders,
        tensor_pca,
        labels,
        pca_model,
        pca_min,
        pca_max,
        input_min,
        input_max,
    ) = create_mosaiq_pca_loaders(
        highres_train,
        batch_size=cfg.mosaiq_batch_size,
        target_size=cfg.target_img_size,
        pca_dims=cfg.pca_dims,
    )
    mosaiq = MosaiqPCA(
        loaders=loaders,
        pca_data=tensor_pca,
        labels=labels,
        mean_=getattr(pca_model, "mean_", np.zeros(cfg.pca_dims)),
        min_=float(tensor_pca.min().item()),
        max_=float(tensor_pca.max().item()),
        pca_min_=pca_min,
        pca_max_=pca_max,
        input_min_=input_min,
        input_max_=input_max,
        pca_model=pca_model,
    )

    return DataBundle(
        train_loader=med_data.train_loader,
        test_loader=med_data.test_loader,
        train_dataset=med_data.train_dataset,
        test_dataset=med_data.test_dataset,
        label_loaders=label_loaders,
        mosaiq=mosaiq,
    )


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_average_csv(rows: List[Dict[str, object]], path: Path, group_keys: List[str]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(path, index=False)
        return

    numeric_cols = [
        col for col in df.select_dtypes(include=["number"]).columns.tolist() if col not in group_keys and col != "Run"
    ]
    grouped = df.groupby(group_keys)[numeric_cols].mean().reset_index()
    grouped.to_csv(path, index=False)


def reverse_scale_data(
    data: np.ndarray, orig_min: float, orig_max: float, scale: Tuple[float, float] = (-1.0, 1.0)
) -> np.ndarray:
    if len(scale) != 2:
        raise ValueError("scale must contain exactly two elements: (min, max)")

    a, b = float(scale[0]), float(scale[1])
    if np.isclose(orig_max, orig_min):
        return np.full_like(data, (orig_min + orig_max) / 2.0)

    return ((data - a) / (b - a)) * (orig_max - orig_min) + orig_min


def mosaiq_pca_to_images(pca_tensor: torch.Tensor, mosaiq: MosaiqPCA, cfg: RunConfig) -> torch.Tensor:
    pca_np = pca_tensor.detach().cpu().numpy()
    unscaled_pca = reverse_scale_data(pca_np, mosaiq.pca_min_, mosaiq.pca_max_)
    reconstructed = mosaiq.pca_model.inverse_transform(unscaled_pca)
    rescaled_inputs = reverse_scale_data(
        reconstructed, mosaiq.input_min_, mosaiq.input_max_, scale=(0.0, 1.0)
    )
    rescaled_inputs = np.clip(rescaled_inputs, 0.0, 1.0)

    images = torch.from_numpy(rescaled_inputs).float()
    images = images.view(-1, cfg.img_channels, cfg.target_img_size, cfg.target_img_size)
    normalized = torch.clamp(images * 2.0 - 1.0, -1.0, 1.0)
    return normalized


class MosaiqImageGenerator(nn.Module):
    def __init__(self, generator: nn.Module, mosaiq: MosaiqPCA, cfg: RunConfig) -> None:
        super().__init__()
        self.generator = generator
        self.mosaiq = mosaiq
        self.cfg = cfg

    def forward(self, noise: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        gen_device = next(self.generator.parameters()).device
        noise = noise.to(gen_device)
        pca_out = self.generator(noise)
        return mosaiq_pca_to_images(pca_out, self.mosaiq, self.cfg)


class LabelledMosaiqImageGenerator(nn.Module):
    """Agrupa geradores MOSAIQ treinados por rótulo e suporta amostragem condicionada."""

    supports_labels = True

    def __init__(self, generators: Dict[int, MosaiqImageGenerator]) -> None:
        super().__init__()
        self.generators = nn.ModuleDict({str(label): gen for label, gen in generators.items()})
        self.labels = sorted(generators.keys())

    def forward(
        self, noise: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # type: ignore[override]
        if labels is None:
            labels = self._balanced_labels(noise.size(0), device=noise.device)

        outputs: List[torch.Tensor] = []
        for label in self.labels:
            mask = labels == label
            if not torch.any(mask):
                continue
            label_noise = noise[mask]
            gen = self.generators[str(label)]
            outputs.append(gen(label_noise))

        if not outputs:
            raise ValueError("Nenhuma amostra foi gerada; verifique os rótulos fornecidos.")

        return torch.cat(outputs, dim=0)

    def _balanced_labels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        reps = int(np.ceil(batch_size / max(len(self.labels), 1)))
        tiled = torch.tensor(self.labels, device=device).repeat(reps)[:batch_size]
        return tiled


class LabelledPatchImageGenerator(nn.Module):
    """Agrupa geradores PatchQGAN treinados por rótulo e suporta amostragem condicionada."""

    supports_labels = True

    def __init__(self, generators: Dict[int, nn.Module]) -> None:
        super().__init__()
        self.generators = nn.ModuleDict({str(label): gen for label, gen in generators.items()})
        self.labels = sorted(generators.keys())

    def forward(
        self, noise: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # type: ignore[override]
        if labels is None:
            labels = self._balanced_labels(noise.size(0), device=noise.device)

        outputs: List[torch.Tensor] = []
        for label in self.labels:
            mask = labels == label
            if not torch.any(mask):
                continue
            label_noise = noise[mask]
            gen = self.generators[str(label)]
            outputs.append(gen(label_noise))

        if not outputs:
            raise ValueError("Nenhuma amostra foi gerada; verifique os rótulos fornecidos.")

        return torch.cat(outputs, dim=0)

    def _balanced_labels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        reps = int(np.ceil(batch_size / max(len(self.labels), 1)))
        tiled = torch.tensor(self.labels, device=device).repeat(reps)[:batch_size]
        return tiled


def evaluate_fid_is(
    real_loader: Iterable,
    generator: nn.Module,
    cfg: RunConfig,
    device: torch.device,
) -> Dict[str, float]:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception = InceptionScore(normalize=True).to(device)

    def ensure_three_channels(imgs: torch.Tensor) -> torch.Tensor:
        """Replicate single-channel images to match Inception's RGB expectation."""

        if imgs.size(1) == 1:
            return imgs.repeat(1, 3, 1, 1)
        return imgs

    # acumula amostras reais
    for batch, _ in real_loader:
        batch = ensure_three_channels(batch.to(device))
        fid.update(batch, real=True)

    generator.eval()
    synth_batches = []
    with torch.no_grad():
        for _ in range(10):
            noise = torch.rand(cfg.batch_size, cfg.latent_dim, device=device) * (torch.pi / 2)
            fake = ensure_three_channels(generator(noise).to(device))
            synth_batches.append(fake)
            fid.update(fake, real=False)
            inception.update(fake)

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    return {"FID": fid_score, "IS_mean": float(is_mean), "IS_std": float(is_std)}


def _generate_with_label(
    generator: nn.Module, noise: torch.Tensor, label: Optional[int]
) -> torch.Tensor:
    if label is None or not getattr(generator, "supports_labels", False):
        return generator(noise)

    labels = torch.full((noise.size(0),), label, dtype=torch.long, device=noise.device)
    return generator(noise, labels=labels)


def evaluate_balancing_strategies(
    real_loader: Iterable,
    generator: nn.Module,
    test_loader: Iterable,
    cfg: RunConfig,
    device: torch.device,
    model_name: str,
    run_idx: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    generator.eval()
    gen_device = next(generator.parameters()).device

    real_images, real_labels = [], []
    for images, labels in real_loader:
        real_images.append(images)
        real_labels.append(labels)
    real_images_t = torch.cat(real_images)
    real_labels_t = torch.cat(real_labels).view(-1).long()

    num_pos = (real_labels_t == 1).sum().item()
    num_neg = (real_labels_t == 0).sum().item()

    needed_pos = max(num_neg - num_pos, 0)
    needed_neg = max(num_pos - num_neg, 0)

    synth_images = []
    synth_labels = []
    if needed_pos > 0:
        labels = torch.ones(needed_pos, dtype=torch.long)
        noise = torch.rand(needed_pos, cfg.latent_dim, device=gen_device) * (torch.pi / 2)
        synth_images.append(_generate_with_label(generator, noise, 1).cpu())
        synth_labels.append(labels)
    if needed_neg > 0:
        labels = torch.zeros(needed_neg, dtype=torch.long)
        noise = torch.rand(needed_neg, cfg.latent_dim, device=gen_device) * (torch.pi / 2)
        synth_images.append(_generate_with_label(generator, noise, 0).cpu())
        synth_labels.append(labels)

    if synth_images:
        synth_images_t = torch.cat(synth_images)
        synth_labels_t = torch.cat(synth_labels).view(-1)
        balanced_images = torch.cat([real_images_t, synth_images_t])
        balanced_labels = torch.cat([real_labels_t, synth_labels_t])
    else:
        balanced_images = real_images_t
        balanced_labels = real_labels_t

    dataset = TensorDataset(balanced_images, balanced_labels)
    row, _ = train_and_evaluate(dataset, test_loader, cfg, device)
    row.update({"Model": model_name, "Run": run_idx, "Strategy": "balance_50_50"})
    rows.append(row)
    return rows


def vary_synth_ratio(
    real_loader: Iterable,
    generator: nn.Module,
    test_loader: Iterable,
    cfg: RunConfig,
    device: torch.device,
    preserve_original_ratio: bool,
    model_name: str,
    run_idx: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    real_images, real_labels = [], []
    for images, labels in real_loader:
        real_images.append(images)
        real_labels.append(labels)
    real_images_t = torch.cat(real_images)
    real_labels_t = torch.cat(real_labels).view(-1).long()

    total_real = len(real_labels_t)
    gen_device = next(generator.parameters()).device

    for ratio in cfg.synth_ratio_grid:
        synth_images = []
        synth_labels = []
        synth_total = int(total_real * ratio)
        half = synth_total // 2

        noise = torch.rand(half, cfg.latent_dim, device=gen_device) * (torch.pi / 2)
        synth_images.append(_generate_with_label(generator, noise, 0).cpu())
        synth_labels.append(torch.zeros(half, dtype=torch.long))

        noise = torch.rand(synth_total - half, cfg.latent_dim, device=gen_device) * (torch.pi / 2)
        synth_images.append(_generate_with_label(generator, noise, 1).cpu())
        synth_labels.append(torch.ones(synth_total - half, dtype=torch.long))

        mixed_images = torch.cat([real_images_t, torch.cat(synth_images)])
        mixed_labels = torch.cat([real_labels_t, torch.cat(synth_labels)])

        dataset = TensorDataset(mixed_images, mixed_labels)
        row, _ = train_and_evaluate(dataset, test_loader, cfg, device)
        row.update(
            {
                "Model": model_name,
                "Run": run_idx,
                "Ratio": ratio,
                "PreserveOriginal": preserve_original_ratio,
            }
        )
        rows.append(row)

    return rows


def train_classifier(
    train_dataset: Dataset, test_loader: DataLoader, cfg: RunConfig, device: torch.device
) -> Tuple[Dict[str, float], float]:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    start = time.perf_counter()
    for _ in range(cfg.clf_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - start

    model.eval()
    all_preds: List[int] = []
    all_probs: List[float] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_targets.extend(labels.view(-1).tolist())

    acc = metrics.accuracy_score(all_targets, all_preds)
    prec = metrics.precision_score(all_targets, all_preds, zero_division=0)
    rec = metrics.recall_score(all_targets, all_preds, zero_division=0)
    f1 = metrics.f1_score(all_targets, all_preds, zero_division=0)
    auc = metrics.roc_auc_score(all_targets, all_probs)

    return (
        {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc},
        train_time,
    )


def train_and_evaluate(
    train_dataset: Dataset, test_loader: DataLoader, cfg: RunConfig, device: torch.device
) -> Tuple[Dict[str, float], float]:
    metrics_row, train_time = train_classifier(train_dataset, test_loader, cfg, device)
    metrics_row["TrainTime"] = train_time
    return metrics_row, train_time


def run_experiments(cfg: RunConfig) -> None:
    device = _default_device()
    data_bundle = prepare_data(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    fid_rows: List[Dict[str, float]] = []
    balance_rows: List[Dict[str, object]] = []
    ratio_bal_rows: List[Dict[str, object]] = []
    ratio_orig_rows: List[Dict[str, object]] = []

    total_models = 1 + (1 if data_bundle.mosaiq is not None else 0)
    progress = ProgressTracker(total_steps=cfg.repeats * total_models)

    def make_epoch_callback(model_name: str, total_epochs: int):
        epoch_tracker = ProgressTracker(total_epochs)
        last = time.perf_counter()

        def _callback(epoch: int, _: int, loss_d: float, loss_g: float) -> None:
            nonlocal last
            now = time.perf_counter()
            epoch_tracker.step(now - last)
            last = now
            print(
                f"[{model_name}] Epoch {epoch}/{total_epochs} "
                f"D_loss={loss_d:.4f} G_loss={loss_g:.4f}",
                flush=True,
            )

        return _callback

    for run_idx in range(cfg.repeats):
        # PatchQGAN
        iter_start = time.perf_counter()
        label_generators: Dict[int, nn.Module] = {}
        total_train_time = 0.0
        for label, loader in data_bundle.label_loaders.items():
            patch_gen, patch_disc = build_patch_models(cfg)
            (patch_hist, _), train_time = timed(train_quantum_gan)(
                loader,
                patch_gen,
                patch_disc,
                epochs=cfg.gan_epochs,
                device=str(device),
                progress_callback=make_epoch_callback(
                    f"patchqgan_label_{label}", cfg.gan_epochs
                ),
            )
            total_train_time += train_time
            label_generators[label] = patch_gen

        patch_img_gen = LabelledPatchImageGenerator(label_generators)
        avg_inf = measure_inference_latency(patch_img_gen, cfg, device)
        summary_rows.append(
            {
                "Model": "patchqgan",
                "Run": run_idx,
                "Train_time_sec": total_train_time,
                "Params": count_parameters(patch_img_gen),
                "Inference_time_per_img_sec": avg_inf,
            }
        )
        fid_rows.append(
            {
                "Model": "patchqgan",
                "Run": run_idx,
                **evaluate_fid_is(data_bundle.test_loader, patch_img_gen, cfg, device),
            }
        )
        balance_rows.extend(
            evaluate_balancing_strategies(
                data_bundle.train_loader,
                patch_img_gen,
                data_bundle.test_loader,
                cfg,
                device,
                "patchqgan",
                run_idx,
            )
        )
        ratio_bal_rows.extend(
            vary_synth_ratio(
                data_bundle.train_loader,
                patch_img_gen,
                data_bundle.test_loader,
                cfg,
                device,
                preserve_original_ratio=False,
                model_name="patchqgan",
                run_idx=run_idx,
            )
        )
        ratio_orig_rows.extend(
            vary_synth_ratio(
                data_bundle.train_loader,
                patch_img_gen,
                data_bundle.test_loader,
                cfg,
                device,
                preserve_original_ratio=True,
                model_name="patchqgan",
                run_idx=run_idx,
            )
        )

        progress.step(time.perf_counter() - iter_start)

        # MOSAIQ
        if data_bundle.mosaiq is None:
            continue
        iter_start = time.perf_counter()
        mosaiq = data_bundle.mosaiq
        label_generators: Dict[int, MosaiqImageGenerator] = {}
        total_train_time = 0.0
        for label, loader in mosaiq.loaders.items():
            mos_gen, mos_disc = build_mosaiq_models(cfg)
            (_, _), train_time = timed(train_mosaiq_gan)(
                loader,
                mos_gen,
                mos_disc,
                epochs=cfg.gan_epochs,
                device=str(device),
                progress_callback=make_epoch_callback(f"mosaiq_label_{label}", cfg.gan_epochs),
            )
            total_train_time += train_time
            label_generators[label] = MosaiqImageGenerator(mos_gen, mosaiq, cfg)

        mosaiq_img_gen = LabelledMosaiqImageGenerator(label_generators)
        avg_inf = measure_inference_latency(mosaiq_img_gen, cfg, device)
        summary_rows.append(
            {
                "Model": "mosaiq",
                "Run": run_idx,
                "Train_time_sec": total_train_time,
                "Params": count_parameters(mosaiq_img_gen),
                "Inference_time_per_img_sec": avg_inf,
            }
        )

        fid_rows.append(
            {
                "Model": "mosaiq",
                "Run": run_idx,
                **evaluate_fid_is(data_bundle.test_loader, mosaiq_img_gen, cfg, device),
            }
        )

        balance_rows.extend(
            evaluate_balancing_strategies(
                data_bundle.train_loader,
                mosaiq_img_gen,
                data_bundle.test_loader,
                cfg,
                device,
                "mosaiq",
                run_idx,
            )
        )

        ratio_bal_rows.extend(
            vary_synth_ratio(
                data_bundle.train_loader,
                mosaiq_img_gen,
                data_bundle.test_loader,
                cfg,
                device,
                preserve_original_ratio=False,
                model_name="mosaiq",
                run_idx=run_idx,
            )
        )

        ratio_orig_rows.extend(
            vary_synth_ratio(
                data_bundle.train_loader,
                mosaiq_img_gen,
                data_bundle.test_loader,
                cfg,
                device,
                preserve_original_ratio=True,
                model_name="mosaiq",
                run_idx=run_idx,
            )
        )

        progress.step(time.perf_counter() - iter_start)

    def csv_path(base_name: str) -> Path:
        return cfg.output_dir / f"{base_name}_{cfg.repeats}.csv"

    save_csv(summary_rows, csv_path("quantum_efficiency"))
    save_csv(fid_rows, csv_path("quantum_synthetic_quality"))
    save_csv(balance_rows, csv_path("quantum_balancing_strategies"))
    save_csv(ratio_bal_rows, csv_path("quantum_balanced_ratios"))
    save_csv(ratio_orig_rows, csv_path("quantum_original_ratio_with_synth"))

    save_average_csv(summary_rows, csv_path("average_quantum_efficiency"), ["Model"])
    save_average_csv(fid_rows, csv_path("average_quantum_synthetic_quality"), ["Model"])
    save_average_csv(
        balance_rows,
        csv_path("average_quantum_balancing_strategies"),
        ["Model", "Strategy"],
    )
    save_average_csv(
        ratio_bal_rows,
        csv_path("average_quantum_balanced_ratios"),
        ["Model", "Ratio"],
    )
    save_average_csv(
        ratio_orig_rows,
        csv_path("average_quantum_original_ratio_with_synth"),
        ["Model", "Ratio"],
    )

    cfg_dict = asdict(cfg)
    cfg_dict["output_dir"] = str(cfg.output_dir)
    (cfg.output_dir / "quantum_config_used.json").write_text(json.dumps(cfg_dict, indent=2))



def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Experimentos com GANs quânticas")
    parser.add_argument("--data-flag", type=str, default="breastmnist")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gan-epochs", type=int, default=50)
    parser.add_argument("--clf-epochs", type=int, default=3)
    parser.add_argument("--target-img-size", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--n-a-qubits", type=int, default=1)
    parser.add_argument("--q-depth", type=int, default=6)
    parser.add_argument("--pca-dims", type=int, default=40)
    parser.add_argument("--mosaiq-batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments_outputs"))
    parser.add_argument("--qml-backend", type=str, default="lightning.qubit")
    parser.add_argument("--qml-diff-method", type=str, default="parameter-shift")
    parser.add_argument("--qml-batch-obs", type=int, default=None)
    parser.add_argument("--qml-mpi", action="store_true", help="Enable MPI backend for lightning.gpu")
    parser.add_argument(
        "--qml-circuit-device",
        type=str,
        default=None,
        help="Force the torch device used to run QNodes (e.g., 'cuda' to align with lightning.gpu)",
    )
    args = parser.parse_args()

    return RunConfig(
        data_flag=args.data_flag,
        batch_size=args.batch_size,
        gan_epochs=args.gan_epochs,
        clf_epochs=args.clf_epochs,
        target_img_size=args.target_img_size,
        latent_dim=args.latent_dim,
        n_a_qubits=args.n_a_qubits,
        q_depth=args.q_depth,
        pca_dims=args.pca_dims,
        mosaiq_batch_size=args.mosaiq_batch_size,
        repeats=args.repeats,
        output_dir=args.output_dir,
        qml_backend=args.qml_backend,
        qml_diff_method=args.qml_diff_method,
        qml_batch_obs=args.qml_batch_obs,
        qml_mpi=args.qml_mpi,
        qml_circuit_device=args.qml_circuit_device,
    )


if __name__ == "__main__":
    run_experiments(parse_args())
