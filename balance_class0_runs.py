"""Batch experiments for class-0 balancing with synthetic data.

This script reproduces the workflow from the notebook section
"Balanceamento de Classes com Dados Sintéticos da Classe 0" while scaling it
up for repeated executions.  The pipeline executes three stochastic stages—
GAN training, synthetic data generation and ResNet-18 classification—multiple
times (10× by default) to capture variability and compute aggregated metrics.
Results are stored as CSV files with both per-run metrics and aggregated
statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision.models import resnet18

from classical_gans import DCDiscriminator, DCGenerator, train_gan_for_class
from medmnist_data import load_medmnist_data

# ---------------------------------------------------------------------------
# Constants and helper dataclasses
# ---------------------------------------------------------------------------

CLASSIFICATION_METRICS = ["acc", "prec", "rec", "f1", "auc", "tn", "fp", "fn", "tp"]
TIME_METRICS = [
    "gan_training_time_sec",
    "synthetic_generation_time_sec",
    "classifier_training_time_sec",
    "classifier_eval_time_sec",
]
ADDITIONAL_NUMERIC_FIELDS = [
    "synthetic_class0_count",
    "balanced_dataset_size",
    "total_real_samples",
    "real_class0_count",
    "real_class1_count",
]
AGGREGATION_FIELDS = CLASSIFICATION_METRICS + TIME_METRICS + ADDITIONAL_NUMERIC_FIELDS


@dataclass
class ExperimentConfig:
    """Configuration controlling the multi-stage experiment."""

    data_flag: str = "breastmnist"
    data_batch_size: int = 128
    latent_dim: int = 100
    gan_epochs: int = 50
    classifier_epochs: int = 5
    num_gan_runs: int = 10
    num_generation_runs: int = 10
    num_classifier_runs: int = 10
    classifier_batch_size: int = 64
    device: Optional[str] = None
    base_seed: int = 2024
    output_dir: Path = Path("balance_class0_runs")


@dataclass
class StageSeeds:
    """Hold the seeds used for a particular (GAN, generation, classifier) trio."""

    gan_seed: int
    generation_seed: int
    classifier_seed: int


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_scalar(y: Tensor | np.ndarray | int | float) -> int:
    if isinstance(y, torch.Tensor):
        return int(y.detach().view(-1)[0].item())
    y_np = np.asarray(y)
    return int(y_np.reshape(-1)[0].item())


def count_class_samples(dataset: Dataset) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for _, label in dataset:
        cls = _to_scalar(label)
        counts[cls] = counts.get(cls, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Data handling helpers
# ---------------------------------------------------------------------------


def custom_collate_fn(batch: Sequence[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    images, labels = zip(*batch)
    xs = torch.stack(images, dim=0)
    ys_list: List[int] = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            if label.numel() == 1:
                ys_list.append(int(label.item()))
            else:
                ys_list.append(int(label.argmax().item()))
        else:
            ys_list.append(int(label))
    ys = torch.tensor(ys_list, dtype=torch.long)
    return xs, ys


def build_balanced_dataset(
    train_dataset: Dataset,
    generator: nn.Module,
    *,
    latent_dim: int,
    synthetic_count: int,
    device: torch.device,
    seed: int,
) -> Tuple[Dataset, int]:
    """Return a dataset with additional synthetic class-0 samples."""

    if synthetic_count <= 0:
        return train_dataset, 0

    _seed_everything(seed)
    generator = generator.to(device).eval()

    noise = torch.randn(synthetic_count, latent_dim, 1, 1, device=device, dtype=torch.float32)
    with torch.no_grad():
        synth_imgs = generator(noise)
    if synth_imgs.dim() == 3:
        synth_imgs = synth_imgs.unsqueeze(1)
    synth_imgs = synth_imgs.to(torch.float32).cpu()
    synth_labels = torch.zeros(synthetic_count, dtype=torch.long)

    synthetic_dataset = TensorDataset(synth_imgs, synth_labels)
    combined_dataset = ConcatDataset([train_dataset, synthetic_dataset])
    return combined_dataset, synthetic_count


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def train_class0_gan(
    train_loader: DataLoader,
    *,
    latent_dim: int,
    gan_epochs: int,
    device: torch.device,
    seed: int,
    img_channels: int,
) -> Tuple[nn.Module, float]:
    """Train a DCGAN for class 0 and return the generator with elapsed time."""

    _seed_everything(seed)
    start = time.time()

    generator = DCGenerator(latent_dim=latent_dim, img_channels=img_channels).to(device)
    discriminator = DCDiscriminator(img_channels=img_channels).to(device)

    generator = train_gan_for_class(
        train_loader=train_loader,
        label_target=0,
        G=generator,
        D=discriminator,
        latent_dim=latent_dim,
        num_epochs=gan_epochs,
        device=device,
    ).eval()

    elapsed = time.time() - start
    return generator, elapsed


def train_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    device: torch.device,
) -> None:
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


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> Tuple[float, float, float, float, float, int, int, int, int]:
    model.to(device)
    model.eval()

    preds: List[Tensor] = []
    labels: List[Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.append(out.argmax(dim=1).cpu())
            labels.append(y)

    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()

    acc = float((y_true == y_pred).mean())

    def _safe_metric(func, default=np.nan):
        try:
            return float(func(y_true, y_pred))
        except Exception:
            return float(default)

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    acc = _safe_metric(accuracy_score)
    prec = _safe_metric(lambda yt, yp: precision_score(yt, yp, zero_division=0))
    rec = _safe_metric(lambda yt, yp: recall_score(yt, yp, zero_division=0))
    f1 = _safe_metric(lambda yt, yp: f1_score(yt, yp, zero_division=0))
    auc = _safe_metric(roc_auc_score, default=np.nan)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return acc, prec, rec, f1, auc, int(tn), int(fp), int(fn), int(tp)


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


def compute_stage_seeds(config: ExperimentConfig, gan_id: int, gen_id: int, clf_id: int) -> StageSeeds:
    gan_seed = config.base_seed + gan_id
    generation_seed = config.base_seed + 1000 * gan_id + gen_id
    classifier_seed = config.base_seed + 1_000_000 * gan_id + 1_000 * gen_id + clf_id
    return StageSeeds(gan_seed=gan_seed, generation_seed=generation_seed, classifier_seed=classifier_seed)


def prepare_resnet(device: torch.device) -> nn.Module:
    model = resnet18(num_classes=2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model.to(device)


def run_balance_experiments(config: ExperimentConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, int]]:
    device = _resolve_device(config.device)
    bundle = load_medmnist_data(
        data_flag=config.data_flag,
        batch_size=config.data_batch_size,
        download=True,
    )

    train_dataset = bundle.train_dataset
    test_dataset = bundle.test_dataset
    train_loader = bundle.train_loader

    counts = count_class_samples(train_dataset)
    real_class0 = counts.get(0, 0)
    real_class1 = counts.get(1, 0)
    deficit = max(0, real_class1 - real_class0)

    results: List[Dict[str, object]] = []

    img_channels = train_dataset[0][0].shape[0]
    total_real_samples = len(train_dataset)

    for gan_run in range(1, config.num_gan_runs + 1):
        seeds_for_gan = compute_stage_seeds(config, gan_run, 0, 0)
        generator, gan_time = train_class0_gan(
            train_loader,
            latent_dim=config.latent_dim,
            gan_epochs=config.gan_epochs,
            device=device,
            seed=seeds_for_gan.gan_seed,
            img_channels=img_channels,
        )

        try:
            for gen_run in range(1, config.num_generation_runs + 1):
                seeds_for_generation = compute_stage_seeds(config, gan_run, gen_run, 0)
                gen_start = time.time()
                balanced_dataset, synth_count = build_balanced_dataset(
                    train_dataset,
                    generator,
                    latent_dim=config.latent_dim,
                    synthetic_count=deficit,
                    device=device,
                    seed=seeds_for_generation.generation_seed,
                )
                generation_time = time.time() - gen_start

                for clf_run in range(1, config.num_classifier_runs + 1):
                    seeds = compute_stage_seeds(config, gan_run, gen_run, clf_run)
                    _seed_everything(seeds.classifier_seed)

                    loader = DataLoader(
                        balanced_dataset,
                        batch_size=config.classifier_batch_size,
                        shuffle=True,
                        pin_memory=device.type == "cuda",
                        drop_last=False,
                        collate_fn=custom_collate_fn,
                    )

                    model = prepare_resnet(device)
                    train_start = time.time()
                    train_classifier(
                        model,
                        loader,
                        epochs=config.classifier_epochs,
                        device=device,
                    )
                    train_time = time.time() - train_start

                    eval_loader = DataLoader(
                        test_dataset,
                        batch_size=config.classifier_batch_size,
                        shuffle=False,
                        pin_memory=device.type == "cuda",
                        drop_last=False,
                        collate_fn=custom_collate_fn,
                    )
                    eval_start = time.time()
                    acc, prec, rec, f1, auc, tn, fp, fn, tp = evaluate_classifier(
                        model, eval_loader, device=device
                    )
                    eval_time = time.time() - eval_start

                    result = {
                        "gan_run_id": gan_run,
                        "generation_run_id": gen_run,
                        "classifier_run_id": clf_run,
                        "ratio": 0.0,
                        "acc": acc,
                        "prec": prec,
                        "rec": rec,
                        "f1": f1,
                        "auc": auc,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "tp": tp,
                        "synthetic_class0_count": synth_count,
                        "balanced_dataset_size": len(balanced_dataset),
                        "total_real_samples": total_real_samples,
                        "real_class0_count": real_class0,
                        "real_class1_count": real_class1,
                        "gan_training_time_sec": gan_time,
                        "synthetic_generation_time_sec": generation_time,
                        "classifier_training_time_sec": train_time,
                        "classifier_eval_time_sec": eval_time,
                        "gan_seed": seeds_for_gan.gan_seed,
                        "generation_seed": seeds_for_generation.generation_seed,
                        "classifier_seed": seeds.classifier_seed,
                    }
                    results.append(result)
        finally:
            # free GPU memory between GAN runs
            generator.cpu()
            del generator
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df_results = pd.DataFrame(results)

    summary_tables = {
        "overall": aggregate_metrics(df_results, []),
        "by_gan": aggregate_metrics(df_results, ["gan_run_id"]),
        "by_gan_generation": aggregate_metrics(df_results, ["gan_run_id", "generation_run_id"]),
    }

    metadata = {
        "data_flag": config.data_flag,
        "latent_dim": config.latent_dim,
        "gan_epochs": config.gan_epochs,
        "classifier_epochs": config.classifier_epochs,
        "num_gan_runs": config.num_gan_runs,
        "num_generation_runs": config.num_generation_runs,
        "num_classifier_runs": config.num_classifier_runs,
        "classifier_batch_size": config.classifier_batch_size,
        "data_batch_size": config.data_batch_size,
        "device": str(device),
        "base_seed": config.base_seed,
        "real_class0_count": real_class0,
        "real_class1_count": real_class1,
        "synthetic_needed_for_balance": deficit,
        "total_real_samples": total_real_samples,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    return df_results, summary_tables, metadata


def aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + [f"{m}_mean" for m in AGGREGATION_FIELDS])

    grouped = df.groupby(group_cols) if group_cols else [((), df)]
    rows: List[Dict[str, object]] = []

    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = {}
        for idx, col in enumerate(group_cols):
            row[col] = key[idx]
        row["num_rows"] = len(group)
        for metric in AGGREGATION_FIELDS:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run repeated class-0 balancing experiments with DCGAN and ResNet-18",
    )
    parser.add_argument("--data-flag", default="breastmnist", help="MedMNIST dataset flag")
    parser.add_argument("--data-batch-size", type=int, default=128, help="Batch size for the GAN data loader")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent dimension for the GAN")
    parser.add_argument("--gan-epochs", type=int, default=50, help="Number of epochs for GAN training")
    parser.add_argument(
        "--classifier-epochs",
        type=int,
        default=5,
        help="Number of epochs for each ResNet-18 training run",
    )
    parser.add_argument("--num-gan-runs", type=int, default=10, help="How many times to retrain the GAN")
    parser.add_argument(
        "--num-generation-runs",
        type=int,
        default=10,
        help="How many synthetic datasets to generate per GAN training",
    )
    parser.add_argument(
        "--num-classifier-runs",
        type=int,
        default=10,
        help="How many classifier trainings per synthetic dataset",
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=64,
        help="Batch size for the ResNet-18 training and evaluation",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (cuda, cpu or auto)",
    )
    parser.add_argument("--base-seed", type=int, default=2024, help="Base seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("balance_class0_runs"),
        help="Directory where CSV and metadata files will be stored",
    )

    args = parser.parse_args()
    return ExperimentConfig(
        data_flag=args.data_flag,
        data_batch_size=args.data_batch_size,
        latent_dim=args.latent_dim,
        gan_epochs=args.gan_epochs,
        classifier_epochs=args.classifier_epochs,
        num_gan_runs=args.num_gan_runs,
        num_generation_runs=args.num_generation_runs,
        num_classifier_runs=args.num_classifier_runs,
        classifier_batch_size=args.classifier_batch_size,
        device=args.device,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
    )


def save_results(
    df_results: pd.DataFrame,
    summary_tables: Dict[str, pd.DataFrame],
    metadata: Dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "balance_class0_results.csv"
    df_results.to_csv(results_path, index=False)

    for name, df in summary_tables.items():
        df.to_csv(output_dir / f"summary_{name}.csv", index=False)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    config = parse_args()
    df_results, summary_tables, metadata = run_balance_experiments(config)
    save_results(df_results, summary_tables, metadata, config.output_dir)


if __name__ == "__main__":
    main()
