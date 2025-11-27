from __future__ import annotations

import argparse  # parser de argumentos de linha de comando
import json  # exportar configurações usadas
import time  # medir tempos de execução
from dataclasses import dataclass, asdict  # estruturas de configuração
from pathlib import Path  # manipulação conveniente de caminhos
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np  # operações vetoriais usadas em métricas
import torch  # base para treino e geração das GANs
import torch.nn as nn  # definição de modelos
import torch.nn.functional as F  # funções auxiliares de perda
from sklearn import metrics  # métricas clássicas do classificador
from torchvision import models  # resnet de referência
from torch.utils.data import DataLoader, TensorDataset  # loaders simples

from classical_gans import (  # arquiteturas e loops de treino existentes
    CGANDiscriminator,
    CGANGenerator,
    DCDiscriminator,
    DCGenerator,
    WGANGPCritic,
    WGANGPGenerator,
    _resolve_device,
    train_cgan,
    train_gan_for_class,
    train_wgangp,
)
from medmnist_data import load_medmnist_data  # carregamento do dataset base


class ProgressTracker:
    """Controla o progresso para estimar tempo restante."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.completed = 0
        self.start_time = time.perf_counter()
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


@dataclass
class RunConfig:
    """Configuração de alto nível para cada rodada de experimento."""

    data_flag: str = "breastmnist"  # dataset padrão usado nos notebooks
    latent_dim: int = 100  # dimensão do ruído para todas as GANs
    img_channels: int = 1  # número de canais do MedMNIST
    num_classes: int = 2  # quantidade de classes alvo
    gan_epochs: int = 50  # épocas reduzidas para rodadas rápidas
    clf_epochs: int = 3  # épocas do classificador de referência
    batch_size: int = 128  # tamanho de batch padrão
    num_workers: int = 0  # workers zero evita travamentos em ambientes simples
    repeats: int = 1  # quantas vezes repetir cada GAN e classificador
    synth_ratio_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5)  # rácios testados
    balance_ratio: float = 0.5  # alvo de balanceamento 50/50
    output_dir: Path = Path("experiments_outputs")  # pasta para CSVs


def timed(fn):
    """Wrapper simples para medir tempo de execução de funções."""

    def wrapper(*args, **kwargs):  # recebe quaisquer argumentos
        start = time.perf_counter()  # marca início
        result = fn(*args, **kwargs)  # executa função alvo
        elapsed = time.perf_counter() - start  # calcula tempo total
        return result, elapsed  # retorna resultado e tempo

    return wrapper  # devolve função decorada


def build_gan(model_name: str, cfg: RunConfig) -> Tuple[nn.Module, nn.Module]:
    """Instancia gerador e discriminador/critic conforme o modelo solicitado."""

    if model_name == "dcgan":  # configuração para DCGAN
        generator = DCGenerator(latent_dim=cfg.latent_dim, img_channels=cfg.img_channels)
        discriminator = DCDiscriminator(img_channels=cfg.img_channels)
    elif model_name == "cgan":  # configuração para CGAN
        generator = CGANGenerator(
            latent_dim=cfg.latent_dim, num_classes=cfg.num_classes, img_channels=cfg.img_channels
        )
        discriminator = CGANDiscriminator(num_classes=cfg.num_classes, img_channels=cfg.img_channels)
    elif model_name == "wgan":  # configuração para WGAN-GP
        generator = WGANGPGenerator(latent_dim=cfg.latent_dim, img_channels=cfg.img_channels)
        discriminator = WGANGPCritic(img_channels=cfg.img_channels)
    else:  # falha para modelos desconhecidos
        raise ValueError(f"Modelo desconhecido: {model_name}")

    return generator, discriminator  # retorna par de módulos prontos


def count_parameters(module: nn.Module) -> int:
    """Conta parâmetros treináveis de um módulo."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)  # soma total


def measure_inference_latency(generator: nn.Module, cfg: RunConfig, device: torch.device) -> float:
    """Calcula tempo médio de inferência por imagem para o gerador."""

    generator.eval()
    batch_size = 32
    noise = torch.randn(batch_size, cfg.latent_dim, 1, 1, device=device)

    with torch.no_grad():
        if isinstance(generator, CGANGenerator):
            # gera labels dummy só pra medir tempo (por ex. aleatórias)
            labels = torch.randint(0, cfg.num_classes, (batch_size,), device=device)
            start = time.perf_counter()
            images = generator(noise, labels)
        else:
            start = time.perf_counter()
            images = generator(noise)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start

    per_image = elapsed / images.size(0)
    return per_image



def generate_synthetic(
    generator: nn.Module, labels: torch.Tensor, cfg: RunConfig, device: torch.device
) -> torch.Tensor:
    """Gera lote de imagens sintéticas alinhadas às labels fornecidas."""

    generator.eval()  # evita atualizações acidentais
    batch_size = labels.size(0)  # define quantidade a ser gerada
    noise = torch.randn(batch_size, cfg.latent_dim, 1, 1, device=device)  # ruído
    with torch.no_grad():  # desativa gradiente
        if isinstance(generator, CGANGenerator):  # caminho condicional
            images = generator(noise, labels.to(device))  # usa labels para CGAN
        else:  # demais modelos ignoram labels
            images = generator(noise)  # geração simples
    return images.detach().cpu()  # retorna no CPU para pós-processamento


def prepare_flattened_dataset(loader: Iterable, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Converte um loader Torch em arrays numpy achatados (imblearn friendly)."""

    all_images: List[np.ndarray] = []  # armazenamento acumulado
    all_labels: List[int] = []  # labels correspondentes
    for images, labels in loader:  # itera sobre batches
        all_images.append(images.view(images.size(0), -1).numpy())  # achata
        all_labels.append(labels.view(-1).numpy())  # guarda labels
    X = np.concatenate(all_images, axis=0)  # junta imagens
    y = np.concatenate(all_labels, axis=0)  # junta labels
    return X, y  # retorna matrizes finais


def build_resnet(num_classes: int, in_channels: int, device: torch.device) -> nn.Module:
    """Cria uma ResNet18 adaptada para entrada de canal único."""

    model = models.resnet18(weights=None)  # inicializa sem pesos pré-treinados
    model.conv1 = nn.Conv2d(  # substitui primeira camada para 1 canal
        in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # ajusta saída
    return model.to(device)  # envia para dispositivo


def train_classifier(
    train_dataset: TensorDataset, test_loader: DataLoader, cfg: RunConfig, device: torch.device
) -> Tuple[Dict[str, float], float]:
    """Treina ResNet compacta e devolve métricas no conjunto de teste."""

    model = build_resnet(cfg.num_classes, cfg.img_channels, device)  # modelo base
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # otimizador padrão
    criterion = nn.CrossEntropyLoss()  # perda de classificação
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)  # loader

    model.train()  # habilita treinamento
    start = time.perf_counter()  # marca início
    for _ in range(cfg.clf_epochs):  # laço de épocas
        for images, labels in train_loader:  # iteração sobre batches
            images, labels = images.to(device), labels.to(device)  # envia para device
            optimizer.zero_grad()  # limpa gradientes
            logits = model(images)  # forward
            loss = criterion(logits, labels)  # calcula perda
            loss.backward()  # backprop
            optimizer.step()  # atualização de pesos
    train_time = time.perf_counter() - start  # mede tempo total

    model.eval()  # modo avaliação
    all_preds: List[int] = []  # armazenará previsões
    all_probs: List[float] = []  # probabilidades para AUC
    all_targets: List[int] = []  # labels reais
    with torch.no_grad():  # sem gradiente
        for images, labels in test_loader:  # percorre teste
            images = images.to(device)  # move imagens
            logits = model(images)  # forward
            probs = F.softmax(logits, dim=1)[:, 1]  # prob classe positiva
            preds = torch.argmax(logits, dim=1)  # classe prevista
            all_preds.extend(preds.cpu().tolist())  # salva previsões
            all_probs.extend(probs.cpu().tolist())  # salva probabilidades
            all_targets.extend(labels.view(-1).tolist())  # salva alvos

    acc = metrics.accuracy_score(all_targets, all_preds)  # acurácia
    prec = metrics.precision_score(all_targets, all_preds, zero_division=0)  # precisão
    rec = metrics.recall_score(all_targets, all_preds, zero_division=0)  # revocação
    f1 = metrics.f1_score(all_targets, all_preds, zero_division=0)  # f1
    auc = metrics.roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5  # AUC segura
    return {"Acc": acc, "Prec": prec, "Rec": rec, "F1": f1, "AUC": auc}, train_time  # devolve métricas


def build_balanced_dataset(
    real_loader: Iterable, generator: nn.Module, cfg: RunConfig, device: torch.device
) -> TensorDataset:
    """Cria dataset balanceado 50/50 usando amostras reais + sintéticas."""

    X_real, y_real = prepare_flattened_dataset(real_loader, device)  # dados reais achatados
    minority = 0 if (y_real == 0).sum() < (y_real == 1).sum() else 1  # identifica classe minoritária
    majority = 1 - minority  # classe majoritária

    num_majority = int((y_real == majority).sum())  # tamanho da classe dominante
    num_minority = int((y_real == minority).sum())  # tamanho da classe rara
    needed = num_majority - num_minority  # quantos exemplos faltam

    labels_needed = torch.full((needed,), minority, dtype=torch.long)  # labels sintéticas
    synthetic = generate_synthetic(generator, labels_needed, cfg, device)  # gera imagens
    synth_labels = torch.full((synthetic.size(0),), minority, dtype=torch.long)  # labels tensor

    real_tensor = torch.tensor(X_real).view(-1, cfg.img_channels, 28, 28)  # reconstrói tensores
    real_labels = torch.tensor(y_real, dtype=torch.long)  # labels reais

    all_images = torch.cat([real_tensor, synthetic], dim=0)  # concatena dados
    all_labels = torch.cat([real_labels, synth_labels], dim=0)  # concatena labels
    return TensorDataset(all_images, all_labels)  # dataset final


def apply_sampling_strategy(X: np.ndarray, y: np.ndarray, strategy: str) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica SMOTE/undersampling/oversampling usando imblearn."""

    from imblearn.over_sampling import RandomOverSampler, SMOTE  # import lazily
    from imblearn.under_sampling import RandomUnderSampler  # idem

    if strategy == "smote":  # SMOTE sintético
        sampler = SMOTE()
    elif strategy == "undersampling":  # reduz classe majoritária
        sampler = RandomUnderSampler()
    elif strategy == "oversampling":  # replica exemplos minoritários
        sampler = RandomOverSampler()
    else:  # estratégia desconhecida
        raise ValueError(f"Estratégia inválida: {strategy}")
    X_res, y_res = sampler.fit_resample(X, y)  # aplica transformação
    return X_res, y_res  # devolve arrays balanceados


def numpy_to_dataset(X: np.ndarray, y: np.ndarray, cfg: RunConfig) -> TensorDataset:
    """Converte arrays numpy achatados para TensorDataset de imagens."""

    images = torch.tensor(X).view(-1, cfg.img_channels, 28, 28).float()  # reconstrói imagens
    labels = torch.tensor(y, dtype=torch.long)  # labels long
    return TensorDataset(images, labels)  # dataset pronto


def evaluate_balancing_strategies(
    base_loader: Iterable,
    generator: Optional[nn.Module],
    test_loader: DataLoader,
    cfg: RunConfig,
    device: torch.device,
    model_name: str,
    run_idx: int,
    include_classical: bool = True,
) -> List[Dict[str, float]]:
    """Executa avaliação comparativa das estratégias de balanceamento."""

    X_real, y_real = prepare_flattened_dataset(base_loader, device)  # extrai base real
    rows: List[Dict[str, float]] = []  # acumula resultados

    if include_classical:
        real_dataset = numpy_to_dataset(X_real, y_real, cfg)  # dataset sem balanceamento
        metrics_base, train_time = train_classifier(real_dataset, test_loader, cfg, device)  # métrica baseline
        rows.append(
            {
                "Model": model_name,
                "Run": run_idx,
                "Strategy": "None",
                **metrics_base,
                "Ratio": 0.0,
                "Synth": 0,
                "TrainTime": train_time,
            }
        )  # registra linha

        for strategy in ("smote", "undersampling", "oversampling"):  # laço sobre técnicas clássicas
            X_bal, y_bal = apply_sampling_strategy(X_real, y_real, strategy)  # aplica sampler
            dataset_bal = numpy_to_dataset(X_bal, y_bal, cfg)  # cria dataset
            metrics_bal, train_time = train_classifier(dataset_bal, test_loader, cfg, device)  # mede desempenho
            synth_count = len(y_bal) - len(y_real)  # quantidade extra
            rows.append(
                {
                    "Model": model_name,
                    "Run": run_idx,
                    "Strategy": strategy,
                    **metrics_bal,
                    "Ratio": cfg.balance_ratio,
                    "Synth": synth_count,
                    "TrainTime": train_time,
                }
            )

    if generator is not None:
        balanced_dataset = build_balanced_dataset(base_loader, generator, cfg, device)  # dataset via GAN
        metrics_gan, train_time = train_classifier(balanced_dataset, test_loader, cfg, device)  # avalia
        synth_count = len(balanced_dataset) - len(X_real)  # calcula extras
        rows.append(
            {
                "Model": model_name,
                "Run": run_idx,
                "Strategy": "gan_balanced",
                **metrics_gan,
                "Ratio": cfg.balance_ratio,
                "Synth": synth_count,
                "TrainTime": train_time,
            }
        )
    return rows  # devolve todas as linhas


def vary_synth_ratio(
    base_loader: Iterable,
    generator: nn.Module,
    test_loader: DataLoader,
    cfg: RunConfig,
    device: torch.device,
    preserve_original_ratio: bool,
    model_name: str,
    run_idx: int,
) -> List[Dict[str, float]]:
    """Avalia diferentes rácios de dados sintéticos mantendo ou não balanceamento."""

    X_real, y_real = prepare_flattened_dataset(base_loader, device)  # base real
    rows: List[Dict[str, float]] = []  # acumula resultados
    for ratio in cfg.synth_ratio_grid:  # percorre rácios
        synth_size = int(len(X_real) * ratio)  # quantidade sintética desejada
        labels = torch.tensor(np.random.choice(cfg.num_classes, synth_size), dtype=torch.long)  # labels aleatórias
        synthetic = generate_synthetic(generator, labels, cfg, device)  # gera imagens
        synth_dataset = TensorDataset(synthetic, labels)  # dataset sintético

        real_dataset = numpy_to_dataset(X_real, y_real, cfg)  # dataset real
        if preserve_original_ratio:  # mantém proporção real
            combined_images = torch.cat([real_dataset.tensors[0], synthetic], dim=0)  # concatena imagens
            combined_labels = torch.cat([real_dataset.tensors[1], labels], dim=0)  # concatena labels
        else:  # força balanceamento 50/50 antes de misturar
            balanced_dataset = build_balanced_dataset(base_loader, generator, cfg, device)  # dataset balanceado
            combined_images = torch.cat([balanced_dataset.tensors[0], synthetic], dim=0)  # imagens balanceadas + sintéticas
            combined_labels = torch.cat([balanced_dataset.tensors[1], labels], dim=0)  # labels correspondentes

        dataset = TensorDataset(combined_images, combined_labels)  # dataset final
        metrics_row, train_time = train_classifier(dataset, test_loader, cfg, device)  # avalia
        rows.append(
            {
                "Model": model_name,
                "Run": run_idx,
                "Ratio": ratio,
                **metrics_row,
                "TrainTime": train_time,
            }
        )  # registra linha
    return rows  # retorna experimentos


def compute_fid_is(
    real_loader: Iterable,
    generator: nn.Module,
    cfg: RunConfig,
    device: torch.device,
    num_images: int = 512,
) -> Tuple[float, float]:
    """Calcula FID e IS usando torchmetrics com normalização adequada."""

    from torchmetrics.image.fid import FrechetInceptionDistance  # métrica FID
    from torchmetrics.image.inception import InceptionScore  # métrica IS

    fid = FrechetInceptionDistance(normalize=True).to(device)  # instancia métrica
    isc = InceptionScore(normalize=True).to(device)  # instancia IS

    collected = 0  # contador de amostras
    for images, _ in real_loader:  # acumula estatísticas reais
        batch = images.to(device).repeat(1, 3, 1, 1)  # duplica canais para 3
        fid.update(batch, real=True)  # atualiza FID real
        collected += batch.size(0)  # soma amostras
        if collected >= num_images:  # limita quantidade
            break  # sai do loop

    remaining = num_images  # controla quantas imagens sintéticas usar
    while remaining > 0:  # gera até completar
        current = min(cfg.batch_size, remaining)  # define tamanho do lote
        labels = torch.randint(0, cfg.num_classes, (current,), device=device)  # labels aleatórias
        fake = generate_synthetic(generator, labels.cpu(), cfg, device).to(device)  # gera lote
        batch = fake.repeat(1, 3, 1, 1)  # ajusta canais
        fid.update(batch, real=False)  # atualiza FID fake
        isc.update(batch)  # atualiza IS
        remaining -= current  # decrementa contador

    fid_score = fid.compute().item()  # extrai valor
    is_score = isc.compute()[0].item()  # primeira saída do InceptionScore
    return fid_score, is_score  # devolve métricas


def train_single_gan(
    model_name: str,
    data_bundle,
    cfg: RunConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Treina uma GAN específica e retorna métricas de eficiência."""

    generator, discriminator = build_gan(model_name, cfg)  # instancia modelos
    generator.to(device)  # envia gerador
    discriminator.to(device)  # envia discriminador

    if model_name == "dcgan":  # treina DCGAN clássica
        train_fn = lambda: train_gan_for_class(
            data_bundle.train_loader,  # loader de treino
            label_target=1,  # classe positiva
            G=generator,
            D=discriminator,
            latent_dim=cfg.latent_dim,
            num_epochs=cfg.gan_epochs,
            device=device,
        )
    elif model_name == "cgan":  # treina CGAN multi-classe
        train_fn = lambda: train_cgan(
            data_bundle.train_loader,
            G=generator,
            D=discriminator,
            latent_dim=cfg.latent_dim,
            num_classes=cfg.num_classes,
            num_epochs=cfg.gan_epochs,
            device=device,
        )
    else:  # WGAN-GP
        train_fn = lambda: train_wgangp(
            data_bundle.train_loader,
            G=generator,
            D=discriminator,
            latent_dim=cfg.latent_dim,
            num_epochs=cfg.gan_epochs,
            device=device,
        )

    trained_generator, train_time = timed(train_fn)()  # executa e mede tempo
    params = count_parameters(trained_generator)  # conta parâmetros
    latency = measure_inference_latency(trained_generator, cfg, device)  # mede latência
    fid_score, is_score = compute_fid_is(data_bundle.train_loader, trained_generator, cfg, device)  # métricas de imagem

    return {
        "Model": model_name,
        "TrainTime": train_time,
        "Params": params,
        "LatencyPerImage": latency,
        "FID": fid_score,
        "IS": is_score,
        "Generator": trained_generator,
    }


def save_csv(rows: List[Dict[str, float]], path: Path) -> None:
    """Salva lista de dicionários como CSV simples."""

    import pandas as pd  # carregamento tardio

    pd.DataFrame(rows).to_csv(path, index=False)  # exporta direto


def save_average_csv(rows: List[Dict[str, float]], path: Path, group_keys: List[str]) -> None:
    """Agrega colunas numéricas por chaves e salva versão média."""

    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(path, index=False)
        return

    numeric_cols = [
        col for col in df.select_dtypes(include=["number"]).columns.tolist() if col not in group_keys and col != "Run"
    ]
    grouped = df.groupby(group_keys)[numeric_cols].mean().reset_index()
    grouped.to_csv(path, index=False)


def main() -> None:
    """Ponto de entrada CLI para orquestrar todos os experimentos."""

    parser = argparse.ArgumentParser(description="Experimentos clássicos automatizados")  # CLI
    parser.add_argument("--config", type=str, default=None, help="Caminho para JSON opcional de configuração")  # config externa
    args = parser.parse_args()  # parseia argumentos

    if args.config:  # se usuário forneceu config
        cfg = RunConfig(**json.loads(Path(args.config).read_text()))  # carrega JSON
    else:  # caso contrário usa padrão
        cfg = RunConfig()  # instância default

    device = _resolve_device(None)  # escolhe CPU ou CUDA
    cfg.output_dir.mkdir(parents=True, exist_ok=True)  # garante diretório

    data_bundle = load_medmnist_data(  # carrega dataset base
        data_flag=cfg.data_flag,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    summary_rows: List[Dict[str, float]] = []  # CSV 1: tempos/params/latência
    fid_rows: List[Dict[str, float]] = []  # CSV 2: FID/IS
    balance_rows: List[Dict[str, float]] = []  # CSV 3: impacto do balanceamento
    ratio_bal_rows: List[Dict[str, float]] = []  # CSV 4: rácios mantendo balanceamento
    ratio_orig_rows: List[Dict[str, float]] = []  # CSV 5: rácios mantendo proporção original

    balance_rows.extend(
        evaluate_balancing_strategies(
            data_bundle.train_loader,
            generator=None,
            test_loader=data_bundle.test_loader,
            cfg=cfg,
            device=device,
            model_name="baseline",
            run_idx=0,
            include_classical=True,
        )
    )

    progress = ProgressTracker(total_steps=cfg.repeats * 3)  # três modelos

    for model_name in ("dcgan", "cgan", "wgan"):  # percorre modelos
        for run_idx in range(cfg.repeats):  # repetições
            iter_start = time.perf_counter()
            metrics_row = train_single_gan(model_name, data_bundle, cfg, device)  # executa treino
            summary_rows.append(
                {
                    "Model": metrics_row["Model"],
                    "Run": run_idx,
                    "TrainTime": metrics_row["TrainTime"],
                    "Params": metrics_row["Params"],
                    "LatencyPerImage": metrics_row["LatencyPerImage"],
                }
            )
            fid_rows.append(
                {
                    "Model": metrics_row["Model"],
                    "Run": run_idx,
                    "FID": metrics_row["FID"],
                    "IS": metrics_row["IS"],
                }
            )

            generator = metrics_row["Generator"]  # recupera gerador treinado

            balance_rows.extend(
                evaluate_balancing_strategies(
                    data_bundle.train_loader,
                    generator,
                    data_bundle.test_loader,
                    cfg,
                    device,
                    model_name,
                    run_idx,
                    include_classical=False,
                )
            )
            ratio_bal_rows.extend(
                vary_synth_ratio(
                    data_bundle.train_loader,
                    generator,
                    data_bundle.test_loader,
                    cfg,
                    device,
                    preserve_original_ratio=False,
                    model_name=model_name,
                    run_idx=run_idx,
                )
            )
            ratio_orig_rows.extend(
                vary_synth_ratio(
                    data_bundle.train_loader,
                    generator,
                    data_bundle.test_loader,
                    cfg,
                    device,
                    preserve_original_ratio=True,
                    model_name=model_name,
                    run_idx=run_idx,
                )
            )

            progress.step(time.perf_counter() - iter_start)

    def csv_path(base_name: str) -> Path:
        return cfg.output_dir / f"{base_name}_{cfg.repeats}.csv"

    save_csv(summary_rows, csv_path("classical_efficiency"))  # salva CSV 1
    save_csv(fid_rows, csv_path("classical_synthetic_quality"))  # salva CSV 2
    save_csv(balance_rows, csv_path("classical_balancing_strategies"))  # salva CSV 3
    save_csv(ratio_bal_rows, csv_path("classical_balanced_ratios"))  # salva CSV 4
    save_csv(ratio_orig_rows, csv_path("classical_original_ratio_with_synth"))  # salva CSV 5

    save_average_csv(summary_rows, csv_path("average_classical_efficiency"), ["Model"])
    save_average_csv(fid_rows, csv_path("average_classical_synthetic_quality"), ["Model"])
    save_average_csv(
        balance_rows,
        csv_path("average_classical_balancing_strategies"),
        ["Model", "Strategy", "Ratio"],
    )
    save_average_csv(
        ratio_bal_rows,
        csv_path("average_classical_balanced_ratios"),
        ["Model", "Ratio"],
    )
    save_average_csv(
        ratio_orig_rows,
        csv_path("average_classical_original_ratio_with_synth"),
        ["Model", "Ratio"],
    )

    cfg_dict = asdict(cfg)  # converte dataclass em dicionário simples
    cfg_dict["output_dir"] = str(cfg.output_dir)  # transforma Path em string para JSON
    (cfg.output_dir / "config_used.json").write_text(json.dumps(cfg_dict, indent=2))  # salva config


if __name__ == "__main__":  # execução direta
    main()  # chama rotina principal
