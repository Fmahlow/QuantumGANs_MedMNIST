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
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
    pca_model: object


@dataclass
class DataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    train_dataset: Dataset
    test_dataset: Dataset
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
    patch_size = 2 ** (cfg.latent_dim - cfg.n_a_qubits)
    n_generators = (cfg.target_img_size ** 2) // patch_size
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

    # MOSAIQ usa PCA sobre as imagens já reescaladas
    dataset_class = med_data.train_dataset.__class__
    highres_train = dataset_class(split="train", transform=transform, download=True)
    loaders, tensor_pca, labels, pca_model = create_mosaiq_pca_loaders(
        highres_train,
        batch_size=cfg.batch_size,
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
        pca_model=pca_model,
    )

    return DataBundle(
        train_loader=med_data.train_loader,
        test_loader=med_data.test_loader,
        train_dataset=med_data.train_dataset,
        test_dataset=med_data.test_dataset,
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

    # acumula amostras reais
    for batch, _ in real_loader:
        fid.update(batch.to(device), real=True)

    generator.eval()
    synth_batches = []
    with torch.no_grad():
        for _ in range(10):
            noise = torch.rand(cfg.batch_size, cfg.latent_dim, device=device) * (torch.pi / 2)
            fake = generator(noise).to(device)
            synth_batches.append(fake)
            fid.update(fake, real=False)
            inception.update(fake)

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    return {"FID": fid_score, "IS_mean": float(is_mean), "IS_std": float(is_std)}


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
    real_labels_t = torch.cat(real_labels)

    num_pos = (real_labels_t == 1).sum().item()
    num_neg = (real_labels_t == 0).sum().item()

    needed_pos = max(num_neg - num_pos, 0)
    needed_neg = max(num_pos - num_neg, 0)

    synth_images = []
    synth_labels = []
    if needed_pos > 0:
        labels = torch.ones(needed_pos, dtype=torch.long)
        noise = torch.rand(needed_pos, cfg.latent_dim, device=gen_device) * (torch.pi / 2)
        synth_images.append(generator(noise).cpu())
        synth_labels.append(labels)
    if needed_neg > 0:
        labels = torch.zeros(needed_neg, dtype=torch.long)
        noise = torch.rand(needed_neg, cfg.latent_dim, device=gen_device) * (torch.pi / 2)
        synth_images.append(generator(noise).cpu())
        synth_labels.append(labels)

    if synth_images:
        synth_images_t = torch.cat(synth_images)
        synth_labels_t = torch.cat(synth_labels)
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
    real_labels_t = torch.cat(real_labels)

    total_real = len(real_labels_t)
    gen_device = next(generator.parameters()).device

    for ratio in cfg.synth_ratio_grid:
        synth_images = []
        synth_labels = []
        synth_total = int(total_real * ratio)
        half = synth_total // 2

        noise = torch.rand(half, cfg.latent_dim, device=gen_device) * (torch.pi / 2)
        synth_images.append(generator(noise).cpu())
        synth_labels.append(torch.zeros(half, dtype=torch.long))

        noise = torch.rand(synth_total - half, cfg.latent_dim, device=gen_device) * (torch.pi / 2)
        synth_images.append(generator(noise).cpu())
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

    for run_idx in range(cfg.repeats):
        # PatchQGAN
        iter_start = time.perf_counter()
        patch_gen, patch_disc = build_patch_models(cfg)
        (patch_hist, _), train_time = timed(train_quantum_gan)(
            data_bundle.train_loader, patch_gen, patch_disc, epochs=cfg.gan_epochs, device=str(device)
        )
        avg_inf = measure_inference_latency(patch_gen, cfg, device)
        summary_rows.append(
            {
                "Model": "patchqgan",
                "Run": run_idx,
                "Train_time_sec": train_time,
                "Params": count_parameters(patch_gen),
                "Inference_time_per_img_sec": avg_inf,
            }
        )
        fid_rows.append(
            {
                "Model": "patchqgan",
                "Run": run_idx,
                **evaluate_fid_is(data_bundle.test_loader, patch_gen, cfg, device),
            }
        )
        balance_rows.extend(
            evaluate_balancing_strategies(
                data_bundle.train_loader,
                patch_gen,
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
                patch_gen,
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
                patch_gen,
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
        mos_gen, mos_disc = build_mosaiq_models(cfg)
        combined_loader = DataLoader(
            mosaiq.pca_data,
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        (_, _), train_time = timed(train_mosaiq_gan)(
            combined_loader, mos_gen, mos_disc, epochs=cfg.gan_epochs, device=str(device)
        )
        avg_inf = measure_inference_latency(mos_gen, cfg, device)
        summary_rows.append(
            {
                "Model": "mosaiq",
                "Run": run_idx,
                "Train_time_sec": train_time,
                "Params": count_parameters(mos_gen),
                "Inference_time_per_img_sec": avg_inf,
            }
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
        ["Model", "Strategy", "Ratio"],
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
