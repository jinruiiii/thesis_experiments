import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.seed import SEED
from models.bayesian_cnn import BayesianCNN


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _model_arch_metadata(model):
    return {
        "conv_channels": list(model.conv_channels),
        "in_channels": model.in_channels,
        "num_classes": model.num_classes,
        "kernel_size": model.kernel_size,
        "padding": model.padding,
        "stride": model.stride,
        "fc_hidden": int(getattr(model, "fc_hidden", model.fc.out_features)),
    }


def save_checkpoint(
    path,
    state_dict,
    epoch,
    model=None,
    conv_channels=None,
    selection_metric=None,
    selection_value=None,
):
    payload = {
        "state_dict": state_dict,
        "epoch": int(epoch),
    }
    if model is not None:
        payload.update(_model_arch_metadata(model))
    if conv_channels is not None:
        payload["conv_channels"] = list(conv_channels)
    if selection_metric is not None:
        payload["selection_metric"] = str(selection_metric)
    if selection_value is not None:
        payload["selection_value"] = float(selection_value)
    torch.save(payload, path)


def load_checkpoint(path, device=None):
    """Load a BayesianCNN from a checkpoint saved by save_checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device)
    conv_channels = payload.get("conv_channels")
    if conv_channels is None:
        raise ValueError(f"checkpoint at {path!r} is missing conv_channels")
    model = BayesianCNN(
        in_channels=payload.get("in_channels", 3),
        conv_channels=list(conv_channels),
        num_classes=payload.get("num_classes", 10),
        kernel_size=payload.get("kernel_size", 3),
        padding=payload.get("padding", 1),
        stride=payload.get("stride", 1),
        fc_hidden=payload.get("fc_hidden", 128),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    metadata = {
        "epoch": payload.get("epoch"),
        "selection_metric": payload.get("selection_metric"),
        "selection_value": payload.get("selection_value"),
    }
    return model, list(conv_channels), metadata


def write_metrics_csv(output_dir, metrics):
    metrics_df = pd.DataFrame({
        "epoch": range(1, 1 + len(metrics["train_loss_total"])),
        "train_loss_total": metrics["train_loss_total"],
        "train_loss_nll": metrics["train_loss_nll"],
        "train_loss_kl": metrics["train_loss_kl"],
        "train_loss_penalty": metrics["train_loss_penalty"],
        "train_acc": metrics["train_acc"],
        "train_brier": metrics["train_brier"],
        "val_loss_total": metrics["val_loss_total"],
        "val_loss_nll": metrics["val_loss_nll"],
        "val_loss_kl": metrics["val_loss_kl"],
        "val_loss_penalty": metrics["val_loss_penalty"],
        "val_acc": metrics["val_acc"],
        "val_brier": metrics["val_brier"],
        "param_count": metrics["param_count_history"],
    })
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)


def write_experiment_summary_csv(
    output_dir,
    model_label,
    params,
    trainable_params,
    conv_channels,
    best_epoch,
    best_val_total,
    best_val_acc,
    best_val_brier,
    test_acc,
    test_brier,
    lambda_penalty,
    selected_checkpoint_metric="val_total",
    junctures_mode="both",
):
    summary_df = pd.DataFrame([{
        "Model": model_label,
        "Parameters": int(params),
        "Trainable Params": int(trainable_params),
        "Best Val Acc": float(best_val_acc),
        "Best Val Brier": float(best_val_brier),
        "Test Acc": float(test_acc),
        "Test Brier": float(test_brier),
        "Conv Channels": str(list(conv_channels)),
        "Lambda Penalty": float(lambda_penalty),
        "Junctures Mode": str(junctures_mode),
        "Selected checkpoint metric": str(selected_checkpoint_metric),
        "Selected epoch": int(best_epoch),
        "Selected val_total": float(best_val_total),
    }])
    summary_df.to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)


def _normalize_checkpoint_metric(checkpoint_metric):
    if checkpoint_metric in ("val_loss_nll", "val_nll"):
        return "val_loss_nll"
    if checkpoint_metric in ("val_loss_total", "val_total"):
        return "val_loss_total"
    raise ValueError(
        f"checkpoint_metric must be 'val_loss_nll' or 'val_loss_total', got {checkpoint_metric!r}"
    )


def _checkpoint_metric_label(checkpoint_metric, phase2=False):
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    prefix = "phase2_mixed_" if phase2 else ""
    if checkpoint_metric == "val_loss_nll":
        return f"{prefix}val_nll"
    return f"{prefix}val_total"


def _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric):
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    if checkpoint_metric == "val_loss_nll":
        return val_loss_nll
    return val_loss_total


def _is_better_checkpoint_score(candidate, best, checkpoint_metric):
    return candidate < best


def loss_function(outputs, labels, kl_loss, beta):
    criterion = nn.CrossEntropyLoss()
    nll = criterion(outputs, labels)
    return nll + kl_loss * beta, nll, kl_loss * beta


def count_params(model):
    """Total variational parameters |theta| for the penalised ELBO."""
    if hasattr(model, "get_param_stats"):
        return model.get_param_stats()["total_params"]
    return sum(p.numel() for p in model.parameters())


def penalised_loss_function(outputs, labels, kl_loss, beta, lambda_penalty, param_count):
    """Minimised objective: NLL + beta*KL + lambda*|theta| (negative penalised ELBO)."""
    _, nll, kl_scaled = loss_function(outputs, labels, kl_loss, beta)
    penalty = lambda_penalty * param_count
    total = nll + kl_scaled + penalty
    return total, nll, kl_scaled, penalty


def penalised_elbo_on_batch(model, inputs, labels, beta_scaled, lambda_penalty):
    """Penalised ELBO L_pen on a single batch (higher is better)."""
    model.eval()
    param_count = count_params(model)
    with torch.no_grad():
        outputs = model(inputs)
        loss, nll, kl_scaled, penalty = penalised_loss_function(
            outputs, labels, model.kl_loss(), beta_scaled, lambda_penalty, param_count
        )
    return -(loss.item())


def penalised_elbo_on_batches(model, batches_val, beta_scaled, lambda_penalty):
    """Mean penalised ELBO over several validation batches (higher is better)."""
    vals = []
    for inputs, labels in batches_val:
        vals.append(penalised_elbo_on_batch(model, inputs, labels, beta_scaled, lambda_penalty))
    return float(np.mean(vals))


def sample_batch(loader, device):
    """Draw one random mini-batch (shuffle=True for stochastic B_ws / B_val)."""
    g = torch.Generator()
    g.manual_seed(SEED + random.randint(0, 1_000_000))
    shuffled_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
    )
    inputs, labels = next(iter(shuffled_loader))
    return inputs.to(device), labels.to(device)


def sample_batches(loader, device, num_batches):
    """
    Draw num_batches consecutive mini-batches from a single shuffled pass
    (no replacement within this chunk).
    """
    g = torch.Generator()
    g.manual_seed(SEED + random.randint(0, 1_000_000))
    shuffled_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
        num_workers=loader.num_workers,
        worker_init_fn=getattr(loader, "worker_init_fn", None),
    )
    batches = []
    it = iter(shuffled_loader)
    for _ in range(num_batches):
        inputs, labels = next(it)
        batches.append((inputs.to(device), labels.to(device)))
    return batches


def train(model, train_dataloader, optimizer, epoch, device, beta_scaled, lambda_penalty=0):
    model.train()
    running_loss_total = 0.0
    running_loss_nll = 0.0
    running_loss_kl = 0.0
    running_loss_penalty = 0.0
    running_brier = 0.0
    correct = 0
    total = 0
    param_count = count_params(model)

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        if lambda_penalty > 0:
            loss, nll, kl, penalty = penalised_loss_function(
                outputs, labels, model.kl_loss(), beta_scaled, lambda_penalty, param_count
            )
        else:
            loss, nll, kl = loss_function(outputs, labels, model.kl_loss(), beta_scaled)
            penalty = torch.tensor(0.0)
        loss.backward()
        optimizer.step()

        running_loss_total += loss.item()
        running_loss_nll += nll.item()
        running_loss_kl += kl.item()
        running_loss_penalty += penalty.item() if hasattr(penalty, "item") else penalty

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        probs = torch.softmax(outputs, dim=1)
        num_classes = outputs.size(1)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        brier = torch.sum((probs - one_hot) ** 2, dim=1).sum()
        running_brier += brier.item()

        progress_bar.set_postfix({
            "loss": running_loss_total / (progress_bar.n + 1),
            "acc": 100.0 * correct / total,
            "brier": running_brier / total,
        })

    train_loss_total = running_loss_total / len(train_dataloader)
    train_acc = 100.0 * correct / total
    train_loss_nll = running_loss_nll / len(train_dataloader)
    train_loss_kl = running_loss_kl / len(train_dataloader)
    train_loss_penalty = running_loss_penalty / len(train_dataloader)
    train_brier = running_brier / total

    return train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty


def validate(model, val_dataloader, device, beta_scaled, lambda_penalty=0):
    model.eval()
    val_loss_total = 0.0
    val_loss_nll = 0.0
    val_loss_kl = 0.0
    val_loss_penalty = 0.0
    running_brier = 0.0
    correct = 0
    total = 0
    param_count = count_params(model)

    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc="Validating"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            if lambda_penalty > 0:
                loss, nll, kl, penalty = penalised_loss_function(
                    outputs, labels, model.kl_loss(), beta_scaled, lambda_penalty, param_count
                )
            else:
                loss, nll, kl = loss_function(outputs, labels, model.kl_loss(), beta_scaled)
                penalty = torch.tensor(0.0)

            val_loss_total += loss.item()
            val_loss_nll += nll.item()
            val_loss_kl += kl.item()
            val_loss_penalty += penalty.item() if hasattr(penalty, "item") else penalty

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            probs = torch.softmax(outputs, dim=1)
            num_classes = outputs.size(1)
            one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            brier = torch.sum((probs - one_hot) ** 2, dim=1).sum()
            running_brier += brier.item()

    val_loss_total = val_loss_total / len(val_dataloader)
    val_acc = 100.0 * correct / total
    val_loss_nll = val_loss_nll / len(val_dataloader)
    val_loss_kl = val_loss_kl / len(val_dataloader)
    val_loss_penalty = val_loss_penalty / len(val_dataloader)
    val_brier = running_brier / total

    return val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty


def _initialise_standard_metrics():
    return {
        "train_loss_total": [],
        "train_loss_nll": [],
        "train_loss_kl": [],
        "train_loss_penalty": [],
        "train_acc": [],
        "train_brier": [],
        "val_loss_total": [],
        "val_loss_nll": [],
        "val_loss_kl": [],
        "val_loss_penalty": [],
        "val_acc": [],
        "val_brier": [],
        "param_count_history": [],
    }
