"""Executa apenas o fluxo PatchQGAN separado do pipeline quântico completo."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch

from run_quantum_experiments import (
    ProgressTracker,
    RunConfig,
    LabelledPatchImageGenerator,
    build_patch_models,
    count_parameters,
    evaluate_balancing_strategies,
    evaluate_fid_is,
    measure_inference_latency,
    prepare_data,
    save_average_csv,
    save_csv,
    timed,
    train_quantum_gan,
    vary_synth_ratio,
    _default_device,
)


def _csv_path(cfg: RunConfig, base_name: str) -> Path:
    return cfg.output_dir / f"{base_name}_{cfg.repeats}.csv"


def run_patch_experiments(cfg: RunConfig) -> None:
    device = _default_device()
    data_bundle = prepare_data(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    fid_rows: List[Dict[str, float]] = []
    balance_rows: List[Dict[str, object]] = []
    ratio_bal_rows: List[Dict[str, object]] = []
    ratio_orig_rows: List[Dict[str, object]] = []

    progress = ProgressTracker(total_steps=cfg.repeats)

    def make_epoch_callback(total_epochs: int):
        epoch_tracker = ProgressTracker(total_epochs)
        last = time.perf_counter()

        def _callback(epoch: int, _: int, loss_d: float, loss_g: float) -> None:
            nonlocal last
            now = time.perf_counter()
            epoch_tracker.step(now - last)
            last = now
            print(
                f"[patchqgan] Epoch {epoch}/{total_epochs} "
                f"D_loss={loss_d:.4f} G_loss={loss_g:.4f}",
                flush=True,
            )

        return _callback

    for run_idx in range(cfg.repeats):
        iter_start = time.perf_counter()
        label_generators: Dict[int, torch.nn.Module] = {}
        total_train_time = 0.0
        for label, loader in data_bundle.label_loaders.items():
            patch_gen, patch_disc = build_patch_models(cfg)
            (_, _), train_time = timed(train_quantum_gan)(
                loader,
                patch_gen,
                patch_disc,
                epochs=cfg.gan_epochs,
                device=str(device),
                progress_callback=make_epoch_callback(cfg.gan_epochs),
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

    save_csv(summary_rows, _csv_path(cfg, "quantum_efficiency_patchqgan"))
    save_csv(fid_rows, _csv_path(cfg, "quantum_synthetic_quality_patchqgan"))
    save_csv(balance_rows, _csv_path(cfg, "quantum_balancing_strategies_patchqgan"))
    save_csv(ratio_bal_rows, _csv_path(cfg, "quantum_balanced_ratios_patchqgan"))
    save_csv(ratio_orig_rows, _csv_path(cfg, "quantum_original_ratio_with_synth_patchqgan"))

    save_average_csv(summary_rows, _csv_path(cfg, "average_quantum_efficiency_patchqgan"), ["Model"])
    save_average_csv(
        fid_rows,
        _csv_path(cfg, "average_quantum_synthetic_quality_patchqgan"),
        ["Model"],
    )
    save_average_csv(
        balance_rows,
        _csv_path(cfg, "average_quantum_balancing_strategies_patchqgan"),
        ["Model", "Strategy"],
    )
    save_average_csv(
        ratio_bal_rows,
        _csv_path(cfg, "average_quantum_balanced_ratios_patchqgan"),
        ["Model", "Ratio"],
    )
    save_average_csv(
        ratio_orig_rows,
        _csv_path(cfg, "average_quantum_original_ratio_with_synth_patchqgan"),
        ["Model", "Ratio"],
    )

    cfg_dict = asdict(cfg)
    cfg_dict["output_dir"] = str(cfg.output_dir)
    (cfg.output_dir / "quantum_config_used_patchqgan.json").write_text(json.dumps(cfg_dict, indent=2))


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Experimentos com PatchQGAN isolado")
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
    run_patch_experiments(parse_args())
