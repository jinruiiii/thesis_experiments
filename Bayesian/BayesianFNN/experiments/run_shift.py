"""
Fashion-MNIST prior-shift experiment (class-proportion flip).

Phase 1 train/val: ~90% on group A (Trouser, Sandal, Sneaker, Bag, Ankle boot),
                   ~10% on group B (T-shirt, Pullover, Dress, Coat, Shirt).
Phase 2: proportions swapped.

Protocol:
  - Train Phase 1 for a conservative number of epochs.
  - Select best Phase-1 checkpoint on Phase-1-matched val.
  - Rewind to that checkpoint, switch loaders, continue Phase 2.
  - Optional (phase2_vcl_prior): freeze Phase-1 posterior as the KL prior for Phase 2.
  - Optional (phase2_regrow_to_init): expand hidden layers back to the original
    init widths before Phase-2 training (copy old weights; new neurons random).
  - Select best Phase-2 checkpoint on Phase-2-matched val for final eval.
  - Report balanced-test, Phase-2-matched, and Phase-1-matched (@P1/@P2) metrics for forgetting.

run_mode:
  - "plasticity": adaptive grow/prune (lambda sweep for Pareto).
  - "baseline": same-start static (no structural junctures, lambda=0).
  - "static_replay": fixed widths from a plasticity experiment_summary.csv.
"""

from __future__ import annotations

import copy
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_plasticity import (
    _experiment_dir_suffix,
    _format_lambda_dir,
    _parse_hidden_sizes_from_summary,
    _resolve_plasticity_dir_for_static_replay,
    _static_replay_output_dir,
    write_static_replay_provenance,
)
from lib.plot import _parse_experiment_dir_name
from lib.plasticity import expand_and_load_encoder_layer, structural_decision_juncture
from lib.seed import SEED, device
from lib.train import (
    _checkpoint_metric_label,
    _is_better_checkpoint_score,
    _normalize_checkpoint_metric,
    _val_score_for_checkpoint,
    count_params,
    ensure_output_dir,
    load_checkpoint,
    save_checkpoint,
    train,
    validate,
)
from models.bayesian_fnn import BayesianFNN
from torchvision import datasets

INPUT_DIM = 784
NUM_CLASSES = 10
DATASET_ROOT = "../../Datasets"

GROUP_A = (1, 5, 7, 8, 9) 
GROUP_B = (0, 2, 3, 4, 6)   
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def _dataset_to_flat_tensors(raw_dataset):
    x = raw_dataset.data.float().div_(255.0).flatten(1)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def _make_loader(dataset, batch_size, shuffle, seed):
    pin_memory = torch.cuda.is_available()
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        pin_memory=pin_memory,
        generator=g,
    )


def _indices_for_prior(
    labels,
    candidate_idx,
    majority_group,
    majority_frac,
    rng,
):
    """
    Subsample candidate_idx so majority_group classes form ~majority_frac of the set.

    Takes all majority-group samples available in candidate_idx, then samples enough
    minority-group samples so minority_frac = 1 - majority_frac.
    """
    majority_group = set(int(c) for c in majority_group)
    cand = np.asarray(candidate_idx, dtype=np.int64)
    y = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)

    maj_mask = np.array([int(y[i]) in majority_group for i in cand], dtype=bool)
    maj_idx = cand[maj_mask]
    min_idx = cand[~maj_mask]

    if len(maj_idx) == 0:
        raise ValueError("No majority-group samples available for prior subsample")
    if len(min_idx) == 0:
        raise ValueError("No minority-group samples available for prior subsample")

    majority_frac = float(majority_frac)
    if not (0.0 < majority_frac < 1.0):
        raise ValueError(f"majority_frac must be in (0, 1), got {majority_frac}")

    n_maj = len(maj_idx)
    n_min_target = int(round(n_maj * (1.0 - majority_frac) / majority_frac))
    n_min_target = max(1, min(n_min_target, len(min_idx)))

    chosen_min = rng.choice(min_idx, size=n_min_target, replace=False)
    out = np.concatenate([maj_idx, chosen_min])
    rng.shuffle(out)
    return out.astype(np.int64)


def _group_counts(labels, indices, group_a=GROUP_A, group_b=GROUP_B):
    y = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
    idx = np.asarray(indices, dtype=np.int64)
    labs = y[idx]
    group_a = set(int(c) for c in group_a)
    n_a = int(sum(int(c) in group_a for c in labs))
    n_b = int(len(labs) - n_a)
    return {
        "n_total": int(len(labs)),
        "n_group_a": n_a,
        "n_group_b": n_b,
        "frac_group_a": float(n_a / max(1, len(labs))),
        "frac_group_b": float(n_b / max(1, len(labs))),
        "per_class": {int(c): int((labs == c).sum()) for c in range(NUM_CLASSES)},
    }


@torch.no_grad()
def evaluate_group_accuracy(model, dataloader, group_a=GROUP_A, group_b=GROUP_B):
    """Return overall / group-A / group-B accuracy (%) and Brier scores."""
    model.eval()
    group_a = set(int(c) for c in group_a)
    group_b = set(int(c) for c in group_b)

    correct = 0
    total = 0
    correct_a = 0
    total_a = 0
    correct_b = 0
    total_b = 0
    brier_sum = 0.0
    brier_sum_a = 0.0
    brier_sum_b = 0.0
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        match = preds.eq(labels)
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()
        brier_per = torch.sum((probs - one_hot) ** 2, dim=1)

        total += labels.size(0)
        correct += int(match.sum().item())
        brier_sum += float(brier_per.sum().item())

        for c in range(NUM_CLASSES):
            mask = labels == c
            n = int(mask.sum().item())
            if n == 0:
                continue
            per_class_total[c] += n
            per_class_correct[c] += int(match[mask].sum().item())

        for c in group_a:
            mask = labels == c
            n = int(mask.sum().item())
            if n == 0:
                continue
            total_a += n
            correct_a += int(match[mask].sum().item())
            brier_sum_a += float(brier_per[mask].sum().item())
        for c in group_b:
            mask = labels == c
            n = int(mask.sum().item())
            if n == 0:
                continue
            total_b += n
            correct_b += int(match[mask].sum().item())
            brier_sum_b += float(brier_per[mask].sum().item())

    def _pct(num, den):
        return 100.0 * num / den if den > 0 else float("nan")

    def _mean(num, den):
        return float(num / den) if den > 0 else float("nan")

    return {
        "overall": _pct(correct, total),
        "group_a": _pct(correct_a, total_a),
        "group_b": _pct(correct_b, total_b),
        "brier": _mean(brier_sum, total),
        "brier_group_a": _mean(brier_sum_a, total_a),
        "brier_group_b": _mean(brier_sum_b, total_b),
        "per_class": {
            int(c): _pct(int(per_class_correct[c]), int(per_class_total[c]))
            for c in range(NUM_CLASSES)
        },
        "n_total": int(total),
        "n_group_a": int(total_a),
        "n_group_b": int(total_b),
    }


def build_prior_shift_dataloaders(
    batch_size=256,
    train_frac=0.8,
    majority_frac=0.9,
    seed=SEED,
    group_a=GROUP_A,
    group_b=GROUP_B,
):
    """
    Build Phase-1 / Phase-2 prior-shifted Fashion-MNIST loaders plus balanced,
    Phase-1-matched (majority A), and Phase-2-matched (majority B) held-out tests.

    Train/val are split first on the official training set, then each split is
    subsampled independently so Phase-1 and Phase-2 share no leaked indices beyond
    the original train/val partition.
    """
    raw_train = datasets.FashionMNIST(root=DATASET_ROOT, train=True, download=True)
    raw_test = datasets.FashionMNIST(root=DATASET_ROOT, train=False, download=True)
    x_all, y_all = _dataset_to_flat_tensors(raw_train)
    x_test, y_test = _dataset_to_flat_tensors(raw_test)

    n = len(raw_train)
    train_size = int(train_frac * n)
    rng_split = np.random.default_rng(seed)
    perm = rng_split.permutation(n)
    train_pool = perm[:train_size]
    val_pool = perm[train_size:]

    rng_p1 = np.random.default_rng(seed + 1)
    rng_p2 = np.random.default_rng(seed + 2)
    rng_test_p2 = np.random.default_rng(seed + 3)
    rng_test_p1 = np.random.default_rng(seed + 4)

    phase1_train_idx = _indices_for_prior(
        y_all, train_pool, majority_group=group_a, majority_frac=majority_frac, rng=rng_p1
    )
    phase1_val_idx = _indices_for_prior(
        y_all, val_pool, majority_group=group_a, majority_frac=majority_frac, rng=rng_p1
    )
    phase2_train_idx = _indices_for_prior(
        y_all, train_pool, majority_group=group_b, majority_frac=majority_frac, rng=rng_p2
    )
    phase2_val_idx = _indices_for_prior(
        y_all, val_pool, majority_group=group_b, majority_frac=majority_frac, rng=rng_p2
    )
    test_pool = np.arange(len(y_test), dtype=np.int64)
    phase2_matched_test_idx = _indices_for_prior(
        y_test, test_pool, majority_group=group_b, majority_frac=majority_frac, rng=rng_test_p2
    )
    phase1_matched_test_idx = _indices_for_prior(
        y_test, test_pool, majority_group=group_a, majority_frac=majority_frac, rng=rng_test_p1
    )

    def _tensor_ds(idx):
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        return TensorDataset(x_all[idx_t], y_all[idx_t])

    phase2_matched_test_idx_t = torch.as_tensor(phase2_matched_test_idx, dtype=torch.long)
    phase1_matched_test_idx_t = torch.as_tensor(phase1_matched_test_idx, dtype=torch.long)
    loaders = {
        "phase1_train": _make_loader(_tensor_ds(phase1_train_idx), batch_size, True, seed),
        "phase1_val": _make_loader(_tensor_ds(phase1_val_idx), batch_size, False, seed + 10),
        "phase2_train": _make_loader(_tensor_ds(phase2_train_idx), batch_size, True, seed + 20),
        "phase2_val": _make_loader(_tensor_ds(phase2_val_idx), batch_size, False, seed + 30),
        "balanced_test": _make_loader(
            TensorDataset(x_test, y_test), batch_size, False, seed + 40
        ),
        "phase2_matched_test": _make_loader(
            TensorDataset(x_test[phase2_matched_test_idx_t], y_test[phase2_matched_test_idx_t]),
            batch_size,
            False,
            seed + 50,
        ),
        "phase1_matched_test": _make_loader(
            TensorDataset(x_test[phase1_matched_test_idx_t], y_test[phase1_matched_test_idx_t]),
            batch_size,
            False,
            seed + 60,
        ),
    }

    manifest = {
        "seed": int(seed),
        "train_frac": float(train_frac),
        "majority_frac": float(majority_frac),
        "group_a": list(group_a),
        "group_b": list(group_b),
        "group_a_names": [CLASS_NAMES[c] for c in group_a],
        "group_b_names": [CLASS_NAMES[c] for c in group_b],
        "train_pool_size": int(len(train_pool)),
        "val_pool_size": int(len(val_pool)),
        "phase1_train": _group_counts(y_all, phase1_train_idx, group_a, group_b),
        "phase1_val": _group_counts(y_all, phase1_val_idx, group_a, group_b),
        "phase2_train": _group_counts(y_all, phase2_train_idx, group_a, group_b),
        "phase2_val": _group_counts(y_all, phase2_val_idx, group_a, group_b),
        "balanced_test_size": int(len(y_test)),
        "phase2_matched_test": _group_counts(
            y_test, phase2_matched_test_idx, group_a, group_b
        ),
        "phase1_matched_test": _group_counts(
            y_test, phase1_matched_test_idx, group_a, group_b
        ),
        "indices": {
            "train_pool": train_pool.tolist(),
            "val_pool": val_pool.tolist(),
            "phase1_train": phase1_train_idx.tolist(),
            "phase1_val": phase1_val_idx.tolist(),
            "phase2_train": phase2_train_idx.tolist(),
            "phase2_val": phase2_val_idx.tolist(),
            "phase2_matched_test": phase2_matched_test_idx.tolist(),
            "phase1_matched_test": phase1_matched_test_idx.tolist(),
        },
    }
    return loaders, manifest


def write_shift_manifest(output_dir, manifest):
    path = os.path.join(output_dir, "shift_manifest.json")
    slim = {k: v for k, v in manifest.items() if k != "indices"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)
    full_path = os.path.join(output_dir, "shift_indices.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(manifest["indices"], f)
    return path, full_path


def write_shift_metrics_csv(output_dir, metrics, phase1_epochs):
    metrics_df = pd.DataFrame({
        "epoch": range(1, 1 + len(metrics["train_loss_total"])),
        "phase": metrics["phase"],
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
        "val_group_a_acc": metrics["val_group_a_acc"],
        "val_group_b_acc": metrics["val_group_b_acc"],
        "param_count": metrics["param_count_history"],
        "phase1_epochs": phase1_epochs,
        "rewound_to_phase1_best": metrics.get("rewound_to_phase1_best_epoch"),
    })
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)


def write_shift_summary_csv(
    output_dir,
    model_label,
    params,
    trainable_params,
    hidden_sizes,
    best_phase1_epoch,
    best_phase2_epoch,
    best_phase2_val_total,
    best_phase2_val_acc,
    best_phase2_val_brier,
    test_acc,
    test_brier,
    test_group_a_acc,
    test_group_b_acc,
    test_group_a_brier,
    test_group_b_brier,
    phase2_matched_test_acc,
    phase2_matched_test_brier,
    phase2_matched_test_group_a_acc,
    phase2_matched_test_group_b_acc,
    phase2_matched_test_group_a_brier,
    phase2_matched_test_group_b_brier,
    phase1_matched_at_p1_acc,
    phase1_matched_at_p1_brier,
    phase1_matched_at_p1_group_a_acc,
    phase1_matched_at_p1_group_b_acc,
    phase1_matched_at_p1_group_a_brier,
    phase1_matched_at_p1_group_b_brier,
    phase1_matched_at_p2_acc,
    phase1_matched_at_p2_brier,
    phase1_matched_at_p2_group_a_acc,
    phase1_matched_at_p2_group_b_acc,
    phase1_matched_at_p2_group_a_brier,
    phase1_matched_at_p2_group_b_brier,
    phase1_matched_forget_acc,
    phase1_matched_forget_group_a_acc,
    lambda_penalty,
    phase1_epochs,
    phase2_epochs,
    junctures_mode="both",
    selected_checkpoint_metric="phase2_val_total",
    run_mode="plasticity",
):
    summary_df = pd.DataFrame([{
        "Model": model_label,
        "Run Mode": str(run_mode),
        "Parameters": int(params),
        "Trainable Params": int(trainable_params),
        "Best Phase 1 Epoch": int(best_phase1_epoch),
        "Best Phase 2 Epoch": int(best_phase2_epoch),
        "Best Phase 2 Val Acc": float(best_phase2_val_acc),
        "Best Phase 2 Val Brier": float(best_phase2_val_brier),
        "Test Acc": float(test_acc),
        "Test Brier": float(test_brier),
        "Test Group A Acc": float(test_group_a_acc),
        "Test Group B Acc": float(test_group_b_acc),
        "Test Group A Brier": float(test_group_a_brier),
        "Test Group B Brier": float(test_group_b_brier),
        "Phase2 Matched Test Acc": float(phase2_matched_test_acc),
        "Phase2 Matched Test Brier": float(phase2_matched_test_brier),
        "Phase2 Matched Test Group A Acc": float(phase2_matched_test_group_a_acc),
        "Phase2 Matched Test Group B Acc": float(phase2_matched_test_group_b_acc),
        "Phase2 Matched Test Group A Brier": float(phase2_matched_test_group_a_brier),
        "Phase2 Matched Test Group B Brier": float(phase2_matched_test_group_b_brier),
        "Phase1 Matched @P1 Acc": float(phase1_matched_at_p1_acc),
        "Phase1 Matched @P1 Brier": float(phase1_matched_at_p1_brier),
        "Phase1 Matched @P1 Group A Acc": float(phase1_matched_at_p1_group_a_acc),
        "Phase1 Matched @P1 Group B Acc": float(phase1_matched_at_p1_group_b_acc),
        "Phase1 Matched @P1 Group A Brier": float(phase1_matched_at_p1_group_a_brier),
        "Phase1 Matched @P1 Group B Brier": float(phase1_matched_at_p1_group_b_brier),
        "Phase1 Matched @P2 Acc": float(phase1_matched_at_p2_acc),
        "Phase1 Matched @P2 Brier": float(phase1_matched_at_p2_brier),
        "Phase1 Matched @P2 Group A Acc": float(phase1_matched_at_p2_group_a_acc),
        "Phase1 Matched @P2 Group B Acc": float(phase1_matched_at_p2_group_b_acc),
        "Phase1 Matched @P2 Group A Brier": float(phase1_matched_at_p2_group_a_brier),
        "Phase1 Matched @P2 Group B Brier": float(phase1_matched_at_p2_group_b_brier),
        "Phase1 Matched Forget Acc": float(phase1_matched_forget_acc),
        "Phase1 Matched Forget Group A Acc": float(phase1_matched_forget_group_a_acc),
        "Hidden Sizes": str(list(hidden_sizes)),
        "Lambda Penalty": float(lambda_penalty),
        "Junctures Mode": str(junctures_mode),
        "Phase 1 Epochs": int(phase1_epochs),
        "Phase 2 Epochs": int(phase2_epochs),
        "Selected checkpoint metric": str(selected_checkpoint_metric),
        "Selected phase2 val_total": float(best_phase2_val_total),
    }])
    summary_df.to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)


def plot_shift_metrics(metrics, phase1_epochs, save_path):
    epochs = range(1, 1 + len(metrics["train_loss_total"]))
    phase_line = phase1_epochs + 0.5
    fig, axs = plt.subplots(4, 3, figsize=(20, 15))

    plot_specs = [
        (0, 0, "train_loss_total", "Training Loss (nll + kl)", "Loss", "blue"),
        (0, 1, "train_loss_nll", "Training Loss (nll)", "Loss", "blue"),
        (0, 2, "train_loss_kl", "Training Loss (kl)", "Loss", "blue"),
        (1, 0, "train_acc", "Training Accuracy", "Accuracy (%)", "blue"),
        (1, 1, "train_brier", "Training Brier", "Brier", "blue"),
        (1, 2, "val_group_a_acc", "Val Group A Acc", "Accuracy (%)", "purple"),
        (2, 0, "val_loss_total", "Validation Loss (nll + kl)", "Loss", "red"),
        (2, 1, "val_loss_nll", "Validation Loss (nll)", "Loss", "red"),
        (2, 2, "val_loss_kl", "Validation Loss (kl)", "Loss", "red"),
        (3, 0, "val_acc", "Validation Accuracy", "Accuracy (%)", "red"),
        (3, 1, "val_brier", "Validation Brier", "Brier", "red"),
    ]

    for row, col, key, title, ylabel, color in plot_specs:
        ax = axs[row, col]
        ax.plot(epochs, metrics[key], color=color, label=key)
        ax.axvline(phase_line, color="black", linestyle="--", linewidth=1.2, label="phase switch")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax = axs[3, 2]
    ax.plot(epochs, metrics["val_group_a_acc"], color="purple", label="group A")
    ax.plot(epochs, metrics["val_group_b_acc"], color="orange", label="group B")
    ax.axvline(phase_line, color="black", linestyle="--", linewidth=1.2, label="phase switch")
    ax.set_title("Val Group Accuracies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, metrics["param_count_history"], color="green", label="param count")
    ax.axvline(phase_line, color="black", linestyle="--", linewidth=1.2, label="phase switch")
    ax.set_title("Parameter Count")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameters")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(
        os.path.join(os.path.dirname(save_path), "param_count.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig2)


def _resolve_phase2_juncture_warmup_epochs(phase2_juncture_warmup_epochs, phase2_only_junctures):
    if phase2_juncture_warmup_epochs is None:
        return 2 if phase2_only_junctures else 0
    return int(phase2_juncture_warmup_epochs)


def _phase2_juncture_start_epoch(phase1_epochs, phase2_juncture_warmup_epochs):
    return phase1_epochs + max(0, int(phase2_juncture_warmup_epochs)) + 1


def regrow_model_to_target(model, target_hidden_sizes):
    target = [int(w) for w in target_hidden_sizes]
    current = [layer.mu_w.shape[0] for layer in model.layers]
    if len(target) != len(current):
        raise ValueError(
            f"target_hidden_sizes length {len(target)} != model depth {len(current)}"
        )
    for i, (cur, tgt) in enumerate(zip(current, target)):
        if tgt < cur:
            raise ValueError(
                f"Cannot regrow layer {i}: target width {tgt} < current width {cur}"
            )
    if target == current:
        return model, current

    model_device = next(model.parameters()).device
    grown = BayesianFNN(model.in_features, target, model.out_features).to(model_device)
    expand_and_load_encoder_layer(model.state_dict(), grown)
    return grown, target


def run_prior_shift_experiment(
    experiment_name,
    model,
    hidden_sizes,
    loaders,
    phase1_epochs,
    phase2_epochs,
    learning_rate,
    beta,
    lambda_penalty,
    gamma,
    rho,
    warm_start_steps,
    warm_start_lr,
    decision_interval=None,
    decision_interval_min=1,
    decision_interval_max=1,
    decision_interval_power=1.0,
    output_dir=None,
    growth_cooldown_junctures=0,
    uncertainty_combine="mean",
    growth_layer_score="mad",
    growth_mad_percentile=90.0,
    snr_combine="geometric",
    junctures_mode="both",
    phase2_only_junctures=False,
    phase2_juncture_warmup_epochs=None,
    prune_mode="global_param",
    global_prune_budget="neurons",
    global_prune_normalize="mad",
    checkpoint_metric="val_loss_total",
    enable_structural=True,
    run_mode="plasticity",
    shift_manifest=None,
    initial_hidden_sizes=None,
    phase2_regrow_to_init=False,
    phase2_vcl_prior=False,
    summary_lambda_penalty=None,
):
    """
    Two-phase Fashion-MNIST prior-shift run.

    At the Phase-1 -> Phase-2 boundary the model is rewound to the best Phase-1
    validation checkpoint before continuing on the flipped prior.
    If phase2_vcl_prior is True, the Phase-1 posterior is frozen as the KL prior
    (VCL-style) before optional regrow / Phase-2 training.
    If phase2_regrow_to_init is True, every hidden layer is then expanded back to
    initial_hidden_sizes (original init widths) before Phase-2 training.
    phase2_juncture_warmup_epochs=K defers all grow/prune until epoch
    (phase1_epochs + K + 1), i.e. the first K Phase-2 epochs are weight-only.
    Final evaluation uses the best Phase-2 validation checkpoint on balanced and
    Phase-2-matched tests, and scores the Phase-1-matched test on both best Phase-1
    and best Phase-2 checkpoints to measure forgetting.
    summary_lambda_penalty:
      Optional value written to experiment_summary.csv as Lambda Penalty while
      training still uses lambda_penalty (e.g. static_replay trains at 0 but
      reports the source plasticity lambda for Pareto coloring).
    """
    if output_dir is None:
        output_dir = os.path.join("./results", experiment_name)
    ensure_output_dir(output_dir)

    csv_lambda_penalty = (
        float(summary_lambda_penalty)
        if summary_lambda_penalty is not None
        else float(lambda_penalty)
    )

    if shift_manifest is not None:
        write_shift_manifest(output_dir, shift_manifest)

    if initial_hidden_sizes is None:
        initial_hidden_sizes = list(hidden_sizes)
    else:
        initial_hidden_sizes = [int(w) for w in initial_hidden_sizes]
    phase2_regrow_to_init = bool(phase2_regrow_to_init)
    phase2_vcl_prior = bool(phase2_vcl_prior)

    total_epochs = int(phase1_epochs) + int(phase2_epochs)
    phase1_epochs = int(phase1_epochs)
    phase2_epochs = int(phase2_epochs)
    if phase1_epochs <= 0 or phase2_epochs <= 0:
        raise ValueError("phase1_epochs and phase2_epochs must be positive")

    phase2_juncture_warmup_epochs = _resolve_phase2_juncture_warmup_epochs(
        phase2_juncture_warmup_epochs, phase2_only_junctures
    )
    phase2_start = _phase2_juncture_start_epoch(phase1_epochs, phase2_juncture_warmup_epochs)

    train_loader = loaders["phase1_train"]
    val_loader = loaders["phase1_val"]

    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric, phase2=True)

    print("\n" + "=" * 50)
    print(
        f"Training {experiment_name.upper()} prior-shift "
        f"(phase1={phase1_epochs}, phase2={phase2_epochs}, "
        f"lambda={lambda_penalty}, junctures_mode={junctures_mode}, "
        f"enable_structural={enable_structural}, "
        f"phase2_only_junctures={phase2_only_junctures}, "
        f"phase2_juncture_warmup_epochs={phase2_juncture_warmup_epochs}, "
        f"phase2_regrow_to_init={phase2_regrow_to_init}, "
        f"phase2_vcl_prior={phase2_vcl_prior})"
    )
    print("=" * 50)
    print(
        f"Phase 1 sizes: train={len(loaders['phase1_train'].dataset):,}, "
        f"val={len(loaders['phase1_val'].dataset):,}"
    )
    print(
        f"Phase 2 sizes: train={len(loaders['phase2_train'].dataset):,}, "
        f"val={len(loaders['phase2_val'].dataset):,}"
    )

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    beta_scaled = (1 / len(train_loader.dataset)) * beta

    metrics = {
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
        "val_group_a_acc": [],
        "val_group_b_acc": [],
        "phase": [],
        "structural_epochs": [],
        "structural_actions": [],
        "structural_delta_grow": [],
        "structural_delta_prune": [],
        "structural_L_before": [],
        "structural_L_after_none": [],
        "structural_hidden_sizes": [],
        "structural_prune_mode": [],
        "structural_prune_target_remove": [],
        "structural_prune_params_removed": [],
        "structural_prune_neurons_pruned": [],
        "structural_global_prune_budget": [],
        "structural_global_prune_normalize": [],
        "param_count_history": [],
        "junctures_mode": junctures_mode,
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "rewound_to_phase1_best_epoch": None,
        "phase2_regrew_to_init": False,
        "phase2_hidden_sizes_before_regrow": None,
        "phase2_hidden_sizes_after_regrow": None,
        "initial_hidden_sizes": list(initial_hidden_sizes),
        "phase2_vcl_prior_applied": False,
    }

    growth_cooldown = {}

    # Phase-1 best (for rewind at switch)
    best_p1_score = float("inf")
    best_p1_epoch = 0
    best_p1_state = None
    best_p1_hidden = list(hidden_sizes)

    # Phase-2 best (for final eval)
    best_p2_score = float("inf")
    best_p2_epoch = 0
    best_p2_state = None
    best_p2_hidden = list(hidden_sizes)
    best_p2_val_acc = 0.0
    best_p2_val_brier = float("inf")

    def _annealed_decision_interval(epoch: int) -> int:
        e = max(1, total_epochs)
        frac = min(1.0, max(0.0, epoch / e))
        raw = decision_interval_min + (
            decision_interval_max - decision_interval_min
        ) * (frac ** decision_interval_power)
        return int(math.floor(raw))

    def _advance_decision_epoch(epoch: int) -> int:
        if decision_interval is not None:
            return min(total_epochs, epoch + int(decision_interval))
        return min(total_epochs, epoch + _annealed_decision_interval(epoch))

    if enable_structural:
        next_decision_epoch = (
            decision_interval if decision_interval is not None else decision_interval_min
        )
        # Only skip Phase-1 junctures when explicitly requested. Phase-2 warmup must
        # NOT delay Phase-1 decisions (that was a prior bug).
        if phase2_only_junctures:
            next_decision_epoch = max(next_decision_epoch, phase2_start)
    else:
        next_decision_epoch = total_epochs + 1

    if enable_structural and phase2_juncture_warmup_epochs > 0:
        print(
            f"Phase-2 structural warmup: no grow/prune for the first "
            f"{phase2_juncture_warmup_epochs} Phase-2 epoch(s); "
            f"first juncture at epoch {phase2_start}"
        )

    for epoch in range(1, total_epochs + 1):
        if epoch == phase1_epochs + 1:
            print("\n" + "=" * 50)
            print(f"PHASE 2 START (epoch {epoch}): rewind to best Phase-1 ckpt + flip prior")
            print("=" * 50)

            if best_p1_state is None:
                raise RuntimeError("No Phase-1 checkpoint available to rewind from")

            # Persist end-of-phase-1 state (pre-rewind) for diagnostics.
            save_checkpoint(
                os.path.join(output_dir, "phase1_end_checkpoint.pth"),
                state_dict=copy.deepcopy(model.state_dict()),
                epoch=phase1_epochs,
                hidden_sizes=[layer.mu_w.shape[0] for layer in model.layers],
                selection_metric="phase1_end",
            )

            model = BayesianFNN(INPUT_DIM, list(best_p1_hidden), NUM_CLASSES).to(device)
            model.load_state_dict(best_p1_state, strict=True)
            hidden_sizes = list(best_p1_hidden)
            metrics["rewound_to_phase1_best_epoch"] = int(best_p1_epoch)
            print(
                f"Rewound to Phase-1 best epoch {best_p1_epoch} "
                f"(hidden_sizes={hidden_sizes})"
            )

            if phase2_vcl_prior:
                model.set_prior_from_posterior()
                metrics["phase2_vcl_prior_applied"] = True
                print(
                    "Phase-2 VCL prior: froze Phase-1 posterior as KL prior "
                    f"(kl_after_set={model.kl_loss().item():.4e})"
                )
                save_checkpoint(
                    os.path.join(output_dir, "phase2_vcl_prior_checkpoint.pth"),
                    state_dict=copy.deepcopy(model.state_dict()),
                    epoch=best_p1_epoch,
                    hidden_sizes=list(hidden_sizes),
                    selection_metric="phase2_vcl_prior",
                )

            if phase2_regrow_to_init:
                sizes_before = list(hidden_sizes)
                model, hidden_sizes = regrow_model_to_target(model, initial_hidden_sizes)
                metrics["phase2_regrew_to_init"] = True
                metrics["phase2_hidden_sizes_before_regrow"] = sizes_before
                metrics["phase2_hidden_sizes_after_regrow"] = list(hidden_sizes)
                print(
                    f"Phase-2 regrow to init: {sizes_before} -> {hidden_sizes} "
                    f"(params={count_params(model):,})"
                )
                save_checkpoint(
                    os.path.join(output_dir, "phase2_regrown_checkpoint.pth"),
                    state_dict=copy.deepcopy(model.state_dict()),
                    epoch=phase1_epochs,
                    hidden_sizes=list(hidden_sizes),
                    selection_metric="phase2_regrow_to_init",
                )

            train_loader = loaders["phase2_train"]
            val_loader = loaders["phase2_val"]
            beta_scaled = (1 / len(train_loader.dataset)) * beta
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            growth_cooldown = {}

            if enable_structural:
                if phase2_juncture_warmup_epochs > 0:
                    # Weight-only adaptation for the first K Phase-2 epochs.
                    next_decision_epoch = phase2_start
                    print(
                        f"Deferring structural junctures until epoch {phase2_start} "
                        f"({phase2_juncture_warmup_epochs} Phase-2 warmup epoch(s))"
                    )
                elif phase2_only_junctures:
                    next_decision_epoch = epoch
                elif next_decision_epoch <= phase1_epochs:
                    # Phase-1 schedule ended; resume junctures from the first Phase-2 epoch.
                    next_decision_epoch = epoch
                # else: keep existing schedule from phase 1 if it already points into phase 2

            print(
                f"Phase 2 dataset sizes: train={len(train_loader.dataset):,}, "
                f"val={len(val_loader.dataset):,}"
            )

        phase_id = 1 if epoch <= phase1_epochs else 2
        metrics["phase"].append(phase_id)

        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty = train(
            model, train_loader, optimizer, epoch, device, beta_scaled, lambda_penalty
        )
        metrics["train_loss_total"].append(train_loss_total)
        metrics["train_loss_nll"].append(train_loss_nll)
        metrics["train_loss_kl"].append(train_loss_kl)
        metrics["train_loss_penalty"].append(train_loss_penalty)
        metrics["train_acc"].append(train_acc)
        metrics["train_brier"].append(train_brier)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        group_stats = evaluate_group_accuracy(model, val_loader)
        metrics["val_loss_total"].append(val_loss_total)
        metrics["val_loss_nll"].append(val_loss_nll)
        metrics["val_loss_kl"].append(val_loss_kl)
        metrics["val_loss_penalty"].append(val_loss_penalty)
        metrics["val_acc"].append(val_acc)
        metrics["val_brier"].append(val_brier)
        metrics["val_group_a_acc"].append(group_stats["group_a"])
        metrics["val_group_b_acc"].append(group_stats["group_b"])
        metrics["param_count_history"].append(count_params(model))

        print(
            f"Epoch {epoch} [phase {phase_id}]: "
            f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, "
            f"Val A={group_stats['group_a']:.2f}%, Val B={group_stats['group_b']:.2f}%, "
            f"Params={count_params(model):,}"
        )

        val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)

        if phase_id == 1 and _is_better_checkpoint_score(val_score, best_p1_score, checkpoint_metric):
            best_p1_score = val_score
            best_p1_epoch = epoch
            best_p1_hidden = [layer.mu_w.shape[0] for layer in model.layers]
            best_p1_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "phase1_best_checkpoint.pth"),
                state_dict=best_p1_state,
                epoch=best_p1_epoch,
                hidden_sizes=best_p1_hidden,
                selection_metric=_checkpoint_metric_label(checkpoint_metric, phase2=False),
                selection_value=best_p1_score,
            )

        if phase_id == 2 and _is_better_checkpoint_score(val_score, best_p2_score, checkpoint_metric):
            best_p2_score = val_score
            best_p2_epoch = epoch
            best_p2_hidden = [layer.mu_w.shape[0] for layer in model.layers]
            best_p2_val_acc = val_acc
            best_p2_val_brier = val_brier
            best_p2_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_p2_state,
                epoch=best_p2_epoch,
                hidden_sizes=best_p2_hidden,
                selection_metric=checkpoint_metric_label,
                selection_value=best_p2_score,
            )

        if enable_structural and epoch == next_decision_epoch and epoch != total_epochs:
            if phase2_only_junctures:
                juncture_allowed = epoch >= phase2_start
            else:
                juncture_allowed = epoch <= phase1_epochs or epoch >= phase2_start

            if not juncture_allowed:
                next_decision_epoch = max(_advance_decision_epoch(epoch), phase2_start)
            else:
                print("-" * 20 + f" Decision juncture (epoch {epoch}, phase {phase_id}) " + "-" * 20)
                exclude_layers = [i for i, rem in growth_cooldown.items() if rem > 0]
                if exclude_layers:
                    print(f"Growth cooldown: excluding layer indices {exclude_layers} (0-based)")
                action, model, hidden_sizes, info = structural_decision_juncture(
                    model,
                    hidden_sizes,
                    train_loader,
                    val_loader,
                    device,
                    beta,
                    lambda_penalty,
                    gamma,
                    rho,
                    warm_start_steps,
                    warm_start_lr,
                    grow_exclude_layers=exclude_layers,
                    uncertainty_combine=uncertainty_combine,
                    growth_layer_score=growth_layer_score,
                    growth_mad_percentile=growth_mad_percentile,
                    snr_combine=snr_combine,
                    junctures_mode=junctures_mode,
                    prune_mode=prune_mode,
                    global_prune_budget=global_prune_budget,
                    global_prune_normalize=global_prune_normalize,
                )
                metrics["structural_epochs"].append(epoch)
                metrics["structural_actions"].append(action)
                metrics["structural_delta_grow"].append(info.get("delta_grow"))
                metrics["structural_delta_prune"].append(info.get("delta_prune"))
                metrics["structural_L_before"].append(info.get("L_before"))
                metrics["structural_L_after_none"].append(info.get("L_after_none"))
                metrics["structural_hidden_sizes"].append(list(hidden_sizes))
                metrics["structural_prune_mode"].append(info.get("prune_mode"))
                metrics["structural_prune_target_remove"].append(info.get("prune_target_remove"))
                metrics["structural_prune_params_removed"].append(info.get("prune_params_removed"))
                metrics["structural_prune_neurons_pruned"].append(info.get("prune_neurons_pruned"))
                metrics["structural_global_prune_budget"].append(info.get("global_prune_budget"))
                metrics["structural_global_prune_normalize"].append(
                    info.get("global_prune_normalize")
                )

                if action != "none":
                    optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
                    )

                if (
                    action == "grow"
                    and growth_cooldown_junctures > 0
                    and junctures_mode in ("both", "grow")
                ):
                    grown = info["grow_layer_idx"]
                    growth_cooldown[grown] = growth_cooldown_junctures
                    for i in list(growth_cooldown.keys()):
                        if i != grown:
                            growth_cooldown[i] -= 1
                            if growth_cooldown[i] <= 0:
                                del growth_cooldown[i]

                next_decision_epoch = _advance_decision_epoch(epoch)

    plot_shift_metrics(
        metrics,
        phase1_epochs=phase1_epochs,
        save_path=os.path.join(output_dir, "metrics.png"),
    )

    if best_p2_state is not None:
        eval_model = BayesianFNN(INPUT_DIM, best_p2_hidden, NUM_CLASSES).to(device)
        eval_model.load_state_dict(best_p2_state, strict=True)
        print(
            f"Loaded best Phase-2 model from epoch {best_p2_epoch} "
            f"(val_acc={best_p2_val_acc:.2f}%, hidden_sizes={best_p2_hidden})"
        )
    else:
        print("Warning: no Phase-2 checkpoint selected; using final training state.")
        eval_model = model
        best_p2_epoch = total_epochs
        best_p2_score = _val_score_for_checkpoint(
            metrics["val_loss_total"][-1],
            metrics["val_loss_nll"][-1],
            checkpoint_metric,
        )
        best_p2_val_acc = metrics["val_acc"][-1]
        best_p2_val_brier = metrics["val_brier"][-1]
        best_p2_hidden = [layer.mu_w.shape[0] for layer in eval_model.layers]

    test_beta = (1 / len(loaders["balanced_test"].dataset)) * beta
    test_loss, test_acc, _, _, test_brier, _ = validate(
        eval_model, loaders["balanced_test"], device, test_beta, lambda_penalty
    )
    test_groups = evaluate_group_accuracy(eval_model, loaders["balanced_test"])

    p2_matched_beta = (1 / len(loaders["phase2_matched_test"].dataset)) * beta
    p2_matched_loss, p2_matched_acc, _, _, p2_matched_brier, _ = validate(
        eval_model, loaders["phase2_matched_test"], device, p2_matched_beta, lambda_penalty
    )
    p2_matched_groups = evaluate_group_accuracy(eval_model, loaders["phase2_matched_test"])

    p1_matched_loader = loaders["phase1_matched_test"]
    p1_matched_beta = (1 / len(p1_matched_loader.dataset)) * beta
    if best_p1_state is None:
        raise RuntimeError("No Phase-1 checkpoint available for forgetting eval")
    p1_eval_model = BayesianFNN(INPUT_DIM, list(best_p1_hidden), NUM_CLASSES).to(device)
    p1_eval_model.load_state_dict(best_p1_state, strict=True)
    _, p1_matched_at_p1_acc, _, _, p1_matched_at_p1_brier, _ = validate(
        p1_eval_model, p1_matched_loader, device, p1_matched_beta, lambda_penalty
    )
    p1_matched_at_p1_groups = evaluate_group_accuracy(p1_eval_model, p1_matched_loader)
    del p1_eval_model

    _, p1_matched_at_p2_acc, _, _, p1_matched_at_p2_brier, _ = validate(
        eval_model, p1_matched_loader, device, p1_matched_beta, lambda_penalty
    )
    p1_matched_at_p2_groups = evaluate_group_accuracy(eval_model, p1_matched_loader)
    p1_matched_forget_acc = float(p1_matched_at_p1_acc - p1_matched_at_p2_acc)
    p1_matched_forget_group_a_acc = float(
        p1_matched_at_p1_groups["group_a"] - p1_matched_at_p2_groups["group_a"]
    )

    print(
        f"Balanced test acc={test_acc:.2f}% "
        f"(A={test_groups['group_a']:.2f}%, B={test_groups['group_b']:.2f}%), "
        f"brier={test_brier:.4f} "
        f"(A={test_groups['brier_group_a']:.4f}, B={test_groups['brier_group_b']:.4f})"
    )
    print(
        f"Phase-2 matched test acc={p2_matched_acc:.2f}% "
        f"(A={p2_matched_groups['group_a']:.2f}%, B={p2_matched_groups['group_b']:.2f}%), "
        f"brier={p2_matched_brier:.4f} "
        f"(A={p2_matched_groups['brier_group_a']:.4f}, B={p2_matched_groups['brier_group_b']:.4f})"
    )
    print(
        f"Phase-1 matched @P1 acc={p1_matched_at_p1_acc:.2f}% "
        f"(A={p1_matched_at_p1_groups['group_a']:.2f}%, B={p1_matched_at_p1_groups['group_b']:.2f}%), "
        f"brier={p1_matched_at_p1_brier:.4f}"
    )
    print(
        f"Phase-1 matched @P2 acc={p1_matched_at_p2_acc:.2f}% "
        f"(A={p1_matched_at_p2_groups['group_a']:.2f}%, B={p1_matched_at_p2_groups['group_b']:.2f}%), "
        f"brier={p1_matched_at_p2_brier:.4f}; "
        f"forget Acc={p1_matched_forget_acc:.2f}pp, "
        f"forget Group A={p1_matched_forget_group_a_acc:.2f}pp"
    )

    final_hidden_sizes = [layer.mu_w.shape[0] for layer in eval_model.layers]
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=best_p2_state if best_p2_state is not None else eval_model.state_dict(),
        epoch=best_p2_epoch,
        hidden_sizes=final_hidden_sizes,
    )

    metrics.update({
        "test_acc": test_acc,
        "test_loss_total": test_loss,
        "test_brier": test_brier,
        "test_group_a_acc": test_groups["group_a"],
        "test_group_b_acc": test_groups["group_b"],
        "test_group_a_brier": test_groups["brier_group_a"],
        "test_group_b_brier": test_groups["brier_group_b"],
        "test_per_class_acc": test_groups["per_class"],
        "phase2_matched_test_acc": p2_matched_acc,
        "phase2_matched_test_loss_total": p2_matched_loss,
        "phase2_matched_test_brier": p2_matched_brier,
        "phase2_matched_test_group_a_acc": p2_matched_groups["group_a"],
        "phase2_matched_test_group_b_acc": p2_matched_groups["group_b"],
        "phase2_matched_test_group_a_brier": p2_matched_groups["brier_group_a"],
        "phase2_matched_test_group_b_brier": p2_matched_groups["brier_group_b"],
        "phase1_matched_at_p1_acc": p1_matched_at_p1_acc,
        "phase1_matched_at_p1_brier": p1_matched_at_p1_brier,
        "phase1_matched_at_p1_group_a_acc": p1_matched_at_p1_groups["group_a"],
        "phase1_matched_at_p1_group_b_acc": p1_matched_at_p1_groups["group_b"],
        "phase1_matched_at_p1_group_a_brier": p1_matched_at_p1_groups["brier_group_a"],
        "phase1_matched_at_p1_group_b_brier": p1_matched_at_p1_groups["brier_group_b"],
        "phase1_matched_at_p2_acc": p1_matched_at_p2_acc,
        "phase1_matched_at_p2_brier": p1_matched_at_p2_brier,
        "phase1_matched_at_p2_group_a_acc": p1_matched_at_p2_groups["group_a"],
        "phase1_matched_at_p2_group_b_acc": p1_matched_at_p2_groups["group_b"],
        "phase1_matched_at_p2_group_a_brier": p1_matched_at_p2_groups["brier_group_a"],
        "phase1_matched_at_p2_group_b_brier": p1_matched_at_p2_groups["brier_group_b"],
        "phase1_matched_forget_acc": p1_matched_forget_acc,
        "phase1_matched_forget_group_a_acc": p1_matched_forget_group_a_acc,
        "param_count": count_params(eval_model),
        "trainable_param_count": eval_model.get_param_stats()["trainable_params"],
        "hidden_sizes": final_hidden_sizes,
        "best_phase1_epoch": best_p1_epoch,
        "best_phase2_epoch": best_p2_epoch,
    })

    write_shift_metrics_csv(output_dir, metrics, phase1_epochs)
    write_shift_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        hidden_sizes=metrics["hidden_sizes"],
        best_phase1_epoch=best_p1_epoch,
        best_phase2_epoch=best_p2_epoch,
        best_phase2_val_total=best_p2_score,
        best_phase2_val_acc=best_p2_val_acc,
        best_phase2_val_brier=best_p2_val_brier,
        test_acc=test_acc,
        test_brier=test_brier,
        test_group_a_acc=test_groups["group_a"],
        test_group_b_acc=test_groups["group_b"],
        test_group_a_brier=test_groups["brier_group_a"],
        test_group_b_brier=test_groups["brier_group_b"],
        phase2_matched_test_acc=p2_matched_acc,
        phase2_matched_test_brier=p2_matched_brier,
        phase2_matched_test_group_a_acc=p2_matched_groups["group_a"],
        phase2_matched_test_group_b_acc=p2_matched_groups["group_b"],
        phase2_matched_test_group_a_brier=p2_matched_groups["brier_group_a"],
        phase2_matched_test_group_b_brier=p2_matched_groups["brier_group_b"],
        phase1_matched_at_p1_acc=p1_matched_at_p1_acc,
        phase1_matched_at_p1_brier=p1_matched_at_p1_brier,
        phase1_matched_at_p1_group_a_acc=p1_matched_at_p1_groups["group_a"],
        phase1_matched_at_p1_group_b_acc=p1_matched_at_p1_groups["group_b"],
        phase1_matched_at_p1_group_a_brier=p1_matched_at_p1_groups["brier_group_a"],
        phase1_matched_at_p1_group_b_brier=p1_matched_at_p1_groups["brier_group_b"],
        phase1_matched_at_p2_acc=p1_matched_at_p2_acc,
        phase1_matched_at_p2_brier=p1_matched_at_p2_brier,
        phase1_matched_at_p2_group_a_acc=p1_matched_at_p2_groups["group_a"],
        phase1_matched_at_p2_group_b_acc=p1_matched_at_p2_groups["group_b"],
        phase1_matched_at_p2_group_a_brier=p1_matched_at_p2_groups["brier_group_a"],
        phase1_matched_at_p2_group_b_brier=p1_matched_at_p2_groups["brier_group_b"],
        phase1_matched_forget_acc=p1_matched_forget_acc,
        phase1_matched_forget_group_a_acc=p1_matched_forget_group_a_acc,
        lambda_penalty=csv_lambda_penalty,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        junctures_mode=junctures_mode,
        selected_checkpoint_metric=checkpoint_metric_label,
        run_mode=run_mode,
    )

    with open(os.path.join(output_dir, "test_per_class_acc.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                CLASS_NAMES[c]: test_groups["per_class"][c]
                for c in range(NUM_CLASSES)
            },
            f,
            indent=2,
        )

    structural_df = pd.DataFrame({
        "epoch": metrics["structural_epochs"],
        "phase": [1 if e <= phase1_epochs else 2 for e in metrics["structural_epochs"]],
        "action": metrics["structural_actions"],
        "delta_grow": metrics["structural_delta_grow"],
        "delta_prune": metrics["structural_delta_prune"],
        "L_before": metrics["structural_L_before"],
        "L_after_none": metrics["structural_L_after_none"],
        "hidden_sizes": [str(hs) for hs in metrics["structural_hidden_sizes"]],
        "junctures_mode": metrics["junctures_mode"],
        "prune_mode": metrics["structural_prune_mode"],
        "prune_target_remove": metrics["structural_prune_target_remove"],
        "prune_params_removed": metrics["structural_prune_params_removed"],
        "prune_neurons_pruned": metrics["structural_prune_neurons_pruned"],
        "global_prune_budget": metrics["structural_global_prune_budget"],
        "global_prune_normalize": metrics["structural_global_prune_normalize"],
    })
    structural_df.to_csv(os.path.join(output_dir, "structural_decisions.csv"), index=False)

    provenance = {
        "run_mode": run_mode,
        "experiment_name": experiment_name,
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "best_phase1_epoch": best_p1_epoch,
        "best_phase2_epoch": best_p2_epoch,
        "rewound_to_phase1_best_epoch": metrics["rewound_to_phase1_best_epoch"],
        "phase2_regrow_to_init": bool(phase2_regrow_to_init),
        "phase2_regrew_to_init": bool(metrics["phase2_regrew_to_init"]),
        "phase2_vcl_prior": bool(phase2_vcl_prior),
        "phase2_vcl_prior_applied": bool(metrics["phase2_vcl_prior_applied"]),
        "phase2_juncture_warmup_epochs": int(phase2_juncture_warmup_epochs),
        "phase2_first_juncture_epoch": int(phase2_start) if enable_structural else None,
        "initial_hidden_sizes": list(initial_hidden_sizes),
        "phase2_hidden_sizes_before_regrow": metrics["phase2_hidden_sizes_before_regrow"],
        "phase2_hidden_sizes_after_regrow": metrics["phase2_hidden_sizes_after_regrow"],
        "final_hidden_sizes": final_hidden_sizes,
        "lambda_penalty": float(lambda_penalty),
        "enable_structural": bool(enable_structural),
        "junctures_mode": junctures_mode,
        "phase2_only_junctures": bool(phase2_only_junctures),
        "test_acc": float(test_acc),
        "test_brier": float(test_brier),
        "test_group_a_acc": float(test_groups["group_a"]),
        "test_group_b_acc": float(test_groups["group_b"]),
        "test_group_a_brier": float(test_groups["brier_group_a"]),
        "test_group_b_brier": float(test_groups["brier_group_b"]),
        "param_count": int(metrics["param_count"]),
    }
    with open(os.path.join(output_dir, "shift_provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print(f"\n{experiment_name} prior-shift summary:")
    print(f"Phase-1 best epoch: {best_p1_epoch}")
    print(f"Phase-2 best epoch: {best_p2_epoch} (val_acc={best_p2_val_acc:.2f}%)")
    print(
        f"Balanced test: {test_acc:.2f}% "
        f"(A={test_groups['group_a']:.2f}%, B={test_groups['group_b']:.2f}%)"
    )
    print(f"Final hidden sizes: {final_hidden_sizes}")
    print(f"Structural decisions: {metrics['structural_actions']}")

    return metrics, model, total_epochs


def main(
    save_path,
    hidden_sizes,
    lambda_penalty=1e-6,
    run_mode="plasticity",
    junctures_mode="both",
    phase1_epochs=20,
    phase2_epochs=20,
    batch_size=256,
    learning_rate=0.005,
    beta=0.002,
    gamma=0.0,
    rho=0.1,
    warm_start_steps=32,
    warm_start_lr=0.002,
    decision_interval_min=1,
    decision_interval_max=1,
    decision_interval_power=1.0,
    phase2_only_junctures=False,
    phase2_juncture_warmup_epochs=0,
    prune_mode="global_param",
    global_prune_budget="neurons",
    global_prune_normalize="mad",
    checkpoint_metric="val_loss_total",
    growth_layer_score="mad",
    growth_mad_percentile=90.0,
    majority_frac=0.9,
    resume_from_plasticity_dir=None,
    phase2_regrow_to_init=False,
    phase2_vcl_prior=False,
):
    """
    run_mode:
      - "plasticity": adaptive structural plasticity under prior shift
      - "baseline": same-start static (no junctures, lambda=0)
      - "static_replay": fixed architecture from plasticity summary (same shift protocol)

    phase2_regrow_to_init:
      If True (plasticity only), after Phase-1 rewind expand every hidden layer
      back to the original init hidden_sizes before Phase-2 training.
    phase2_vcl_prior:
      If True (plasticity, baseline, or static_replay), after Phase-1 rewind
      freeze the Phase-1 posterior as the KL prior for Phase 2 (VCL-style).
      Applied before regrow.
    phase2_juncture_warmup_epochs:
      Number of Phase-2 epochs with weight-only training (no grow/prune) after
      the switch. Phase-1 junctures are unaffected. Example: phase1_epochs=30
      and warmup=5 => first juncture in Phase 2 at epoch 36.
    """
    allowed = {"plasticity", "baseline", "static_replay"}
    if run_mode not in allowed:
        raise ValueError(f"run_mode must be one of {sorted(allowed)}, got {run_mode!r}")

    os.makedirs(save_path, exist_ok=True)
    loaders, manifest = build_prior_shift_dataloaders(
        batch_size=batch_size,
        majority_frac=majority_frac,
        seed=SEED,
    )

    print("\n" + "=" * 50)
    print(f"Prior-shift setup (run_mode={run_mode})")
    print("=" * 50)
    print(f"Group A (Phase-1 majority): {[CLASS_NAMES[c] for c in GROUP_A]}")
    print(f"Group B (Phase-2 majority): {[CLASS_NAMES[c] for c in GROUP_B]}")
    print(f"majority_frac={majority_frac}")
    print(f"phase2_regrow_to_init={phase2_regrow_to_init}")
    print(f"phase2_vcl_prior={phase2_vcl_prior}")
    for key in ("phase1_train", "phase1_val", "phase2_train", "phase2_val"):
        stats = manifest[key]
        print(
            f"  {key}: n={stats['n_total']}, "
            f"A={stats['frac_group_a']:.1%}, B={stats['frac_group_b']:.1%}"
        )

    enable_structural = True
    summary_lambda = float(lambda_penalty)
    model_hidden = list(hidden_sizes)
    # Init widths for optional Phase-2 regrow (plasticity start size).
    initial_hidden_sizes = list(hidden_sizes)
    # Only meaningful for plasticity; ignore for baseline / static_replay.
    use_phase2_regrow = bool(phase2_regrow_to_init) and run_mode == "plasticity"
    # VCL prior applies to plasticity, baseline, and static_replay (fair control).
    use_phase2_vcl = bool(phase2_vcl_prior) and run_mode in (
        "plasticity",
        "baseline",
        "static_replay",
    )
    vcl_tag = "_vcl" if use_phase2_vcl else ""

    if run_mode == "baseline":
        lambda_penalty = 0.0
        enable_structural = False
        experiment_name = "shift_baseline"
        run_name = f"baseline_{hidden_sizes[0]}{vcl_tag}"
        output_dir = os.path.join(save_path, run_name)
    elif run_mode == "static_replay":
        if resume_from_plasticity_dir is None:
            raise ValueError("run_mode='static_replay' requires resume_from_plasticity_dir")
        plasticity_dir, summary_path = _resolve_plasticity_dir_for_static_replay(
            resume_from_plasticity_dir
        )
        model_hidden = _parse_hidden_sizes_from_summary(summary_path)
        source_meta = _parse_experiment_dir_name(os.path.basename(plasticity_dir))
        source_lambda = source_meta.get("lambda_penalty")
        if source_lambda is None:
            source_lambda = float(lambda_penalty)
        summary_lambda = float(source_lambda)
        lambda_penalty = 0.0
        enable_structural = False
        experiment_name = "shift_static_replay"
        # Place beside source plasticity_* (mirrors main2 naming).
        output_dir = _static_replay_output_dir(plasticity_dir)
        if use_phase2_vcl and not os.path.basename(output_dir).endswith("_vcl"):
            output_dir = output_dir + "_vcl"
        print(f"Static replay widths from {plasticity_dir}: {model_hidden}")
        print(f"Static replay output dir: {output_dir}")
        print(f"Source lambda (summary): {summary_lambda}")
    else:
        experiment_name = "shift_plasticity"
        suffix = _experiment_dir_suffix(junctures_mode)
        regrow_tag = "_regrow" if use_phase2_regrow else ""
        p2junct_tag = "_p2junct" if phase2_only_junctures else ""
        run_name = (
            f"plasticity_{hidden_sizes[0]}_{_format_lambda_dir(lambda_penalty)}"
            f"{suffix}{regrow_tag}{p2junct_tag}{vcl_tag}"
        )
        output_dir = os.path.join(save_path, run_name)

    ensure_output_dir(output_dir)

    if run_mode == "static_replay":
        write_static_replay_provenance(
            output_dir,
            {
                "source_plasticity_dir": plasticity_dir,
                "source_summary_csv": summary_path,
                "replayed_hidden_sizes": model_hidden,
                "source_lambda_penalty": summary_lambda,
                "training_lambda_penalty": 0.0,
                "protocol": "fashion_mnist_prior_shift",
                "phase2_vcl_prior": use_phase2_vcl,
            },
        )

    model = BayesianFNN(INPUT_DIM, model_hidden, NUM_CLASSES).to(device)

    return run_prior_shift_experiment(
        experiment_name,
        model,
        list(model_hidden),
        loaders,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        learning_rate=learning_rate,
        beta=beta,
        lambda_penalty=lambda_penalty,
        gamma=gamma,
        rho=rho,
        warm_start_steps=warm_start_steps,
        warm_start_lr=warm_start_lr,
        decision_interval=None,
        decision_interval_min=decision_interval_min,
        decision_interval_max=decision_interval_max,
        decision_interval_power=decision_interval_power,
        output_dir=output_dir,
        uncertainty_combine="mean",
        junctures_mode=junctures_mode,
        phase2_only_junctures=phase2_only_junctures,
        phase2_juncture_warmup_epochs=phase2_juncture_warmup_epochs,
        prune_mode=prune_mode,
        global_prune_budget=global_prune_budget,
        global_prune_normalize=global_prune_normalize,
        checkpoint_metric=checkpoint_metric,
        growth_layer_score=growth_layer_score,
        growth_mad_percentile=growth_mad_percentile,
        enable_structural=enable_structural,
        run_mode=run_mode,
        shift_manifest=manifest,
        initial_hidden_sizes=initial_hidden_sizes,
        phase2_regrow_to_init=use_phase2_regrow,
        phase2_vcl_prior=use_phase2_vcl,
        summary_lambda_penalty=summary_lambda if run_mode == "static_replay" else None,
    )

