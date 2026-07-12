"""
Curriculum training: Phase 1 FashionMNIST only, Phase 2 FashionMNIST + SVHN.

- Images aligned to 28x28 grayscale, flattened to 784 features.
- 20 output classes: Fashion 0-9, SVHN 10-19 (SVHN labels offset by +10).
- Single continuous adaptive run with a data-loader switch at the phase boundary.
- Structural grow/prune junctures can run in both phases (default) or phase 2 only
  (phase2_only_junctures=True). Optional phase2_juncture_warmup_epochs delays junctures
  for the first N epochs of phase 2 (defaults to 20 when phase2_only_junctures=True).
"""

import copy
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

from main2 import (
    SEED,
    BayesianFNN,
    count_params,
    device,
    ensure_output_dir,
    save_checkpoint,
    seed_worker,
    set_seed,
    structural_decision_juncture,
    train,
    validate,
    plot_param_count_vs_test_acc,
    _normalize_checkpoint_metric,
    _checkpoint_metric_label,
    _val_score_for_checkpoint,
    _is_better_checkpoint_score,
)

INPUT_DIM = 784
FASHION_CLASSES = 10
SVHN_CLASS_OFFSET = 10
NUM_CLASSES = FASHION_CLASSES + SVHN_CLASS_OFFSET
DATASET_ROOT = "../../Datasets"


def fashion_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])


def svhn_transform():
    return transforms.Compose([
        transforms.Resize(28),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])


def _split_dataset(dataset, train_frac=0.8, seed=SEED):
    train_size = int(train_frac * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _make_loader(dataset, batch_size, shuffle, generator):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_curriculum_dataloaders(batch_size=1024, train_frac=0.8, seed=SEED):
    """
    Build loaders for:
      - phase 1: Fashion train/val
      - phase 2: mixed Fashion+SVHN train/val
      - tests: Fashion-only, SVHN-only (offset labels), and combined test set
    """
    g = torch.Generator()
    g.manual_seed(seed)

    fashion_train_full = datasets.FashionMNIST(
        root=DATASET_ROOT,
        train=True,
        download=True,
        transform=fashion_transform(),
    )
    fashion_test = datasets.FashionMNIST(
        root=DATASET_ROOT,
        train=False,
        download=True,
        transform=fashion_transform(),
    )
    svhn_train_full = datasets.SVHN(
        root=DATASET_ROOT,
        split="train",
        download=True,
        transform=svhn_transform(),
        target_transform=lambda y: y + SVHN_CLASS_OFFSET,
    )
    svhn_test = datasets.SVHN(
        root=DATASET_ROOT,
        split="test",
        download=True,
        transform=svhn_transform(),
        target_transform=lambda y: y + SVHN_CLASS_OFFSET,
    )

    fashion_train, fashion_val = _split_dataset(fashion_train_full, train_frac, seed)
    svhn_train, svhn_val = _split_dataset(svhn_train_full, train_frac, seed + 1)

    phase1_train_loader = _make_loader(fashion_train, batch_size, True, g)
    phase1_val_loader = _make_loader(fashion_val, batch_size, False, g)

    mixed_train = ConcatDataset([fashion_train, svhn_train])
    mixed_val = ConcatDataset([fashion_val, svhn_val])
    mixed_test = ConcatDataset([fashion_test, svhn_test])

    phase2_train_loader = _make_loader(mixed_train, batch_size, True, g)
    phase2_val_loader = _make_loader(mixed_val, batch_size, False, g)

    fashion_test_loader = _make_loader(fashion_test, batch_size, False, g)
    svhn_test_loader = _make_loader(svhn_test, batch_size, False, g)
    mixed_test_loader = _make_loader(mixed_test, batch_size, False, g)

    return {
        "phase1_train": phase1_train_loader,
        "phase1_val": phase1_val_loader,
        "phase2_train": phase2_train_loader,
        "phase2_val": phase2_val_loader,
        "fashion_test": fashion_test_loader,
        "svhn_test": svhn_test_loader,
        "mixed_test": mixed_test_loader,
    }


def write_curriculum_metrics_csv(output_dir, metrics, phase1_epochs):
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
        "param_count": metrics["param_count_history"],
        "phase1_epochs": phase1_epochs,
    })
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)


def write_curriculum_summary_csv(
    output_dir,
    model_label,
    params,
    trainable_params,
    hidden_sizes,
    best_epoch,
    best_val_total,
    best_val_acc,
    best_val_brier,
    fashion_test_acc,
    fashion_test_brier,
    svhn_test_acc,
    svhn_test_brier,
    mixed_test_acc,
    mixed_test_brier,
    lambda_penalty,
    phase1_epochs,
    phase2_epochs,
    junctures_mode="both",
    selected_checkpoint_metric="phase2_mixed_val_total",
):
    summary_df = pd.DataFrame([{
        "Model": model_label,
        "Parameters": int(params),
        "Trainable Params": int(trainable_params),
        "Best Phase 2 Val Acc": float(best_val_acc),
        "Best Phase 2 Val Brier": float(best_val_brier),
        "Fashion Test Acc": float(fashion_test_acc),
        "Fashion Test Brier": float(fashion_test_brier),
        "SVHN Test Acc": float(svhn_test_acc),
        "SVHN Test Brier": float(svhn_test_brier),
        "Mixed Test Acc": float(mixed_test_acc),
        "Mixed Test Brier": float(mixed_test_brier),
        "Hidden Sizes": str(list(hidden_sizes)),
        "Lambda Penalty": float(lambda_penalty),
        "Junctures Mode": str(junctures_mode),
        "Phase 1 Epochs": int(phase1_epochs),
        "Phase 2 Epochs": int(phase2_epochs),
        "Selected checkpoint metric": str(selected_checkpoint_metric),
        "Selected epoch": int(best_epoch),
        "Selected phase2 val_total": float(best_val_total),
    }])
    summary_df.to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)


def plot_curriculum_metrics(metrics, phase1_epochs, save_path):
    """Plot all train/val metrics with a vertical line at the phase switch."""
    epochs = range(1, 1 + len(metrics["train_loss_total"]))
    phase_line = phase1_epochs + 0.5
    fig, axs = plt.subplots(4, 3, figsize=(20, 15))

    plot_specs = [
        (0, 0, "train_loss_total", "Training Loss (nll + kl)", "Loss", "blue", "train"),
        (0, 1, "train_loss_nll", "Training Loss (nll)", "Loss", "blue", "train nll"),
        (0, 2, "train_loss_kl", "Training Loss (kl)", "Loss", "blue", "train kl"),
        (1, 0, "train_acc", "Training Accuracy", "Accuracy (%)", "blue", "train acc"),
        (1, 1, "train_brier", "Training Brier", "Brier", "blue", "train brier"),
        (2, 0, "val_loss_total", "Validation Loss (nll + kl)", "Loss", "red", "val"),
        (2, 1, "val_loss_nll", "Validation Loss (nll)", "Loss", "red", "val nll"),
        (2, 2, "val_loss_kl", "Validation Loss (kl)", "Loss", "red", "val kl"),
        (3, 0, "val_acc", "Validation Accuracy", "Accuracy (%)", "red", "val acc"),
        (3, 1, "val_brier", "Validation Brier", "Brier", "red", "val brier"),
    ]

    for row, col, key, title, ylabel, color, label in plot_specs:
        ax = axs[row, col]
        ax.plot(epochs, metrics[key], color=color, label=label)
        ax.axvline(phase_line, color="black", linestyle="--", linewidth=1.2, label="phase switch")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax = axs[3, 2]
    ax.plot(epochs, metrics["param_count_history"], color="green", label="param count")
    ax.axvline(phase_line, color="black", linestyle="--", linewidth=1.2, label="phase switch")
    ax.set_title("Parameter Count")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameters")
    ax.grid(True, alpha=0.3)
    ax.legend()

    axs[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _phase2_juncture_start_epoch(phase1_epochs, phase2_juncture_warmup_epochs, junctures_at_phase_switch):
    """First epoch at which a structural juncture may run in phase 2."""
    if phase2_juncture_warmup_epochs > 0:
        return phase1_epochs + phase2_juncture_warmup_epochs + 1
    if junctures_at_phase_switch:
        return phase1_epochs + 1
    return phase1_epochs + 1


def _resolve_phase2_juncture_warmup_epochs(phase2_juncture_warmup_epochs, phase2_only_junctures):
    if phase2_juncture_warmup_epochs is None:
        return 20 if phase2_only_junctures else 0
    return int(phase2_juncture_warmup_epochs)


def run_curriculum_adaptive_experiment(
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
    decision_interval_min=5,
    decision_interval_max=5,
    decision_interval_power=1.0,
    output_dir=None,
    growth_cooldown_junctures=0,
    uncertainty_combine="mean",
    growth_layer_score="mad",
    growth_mad_percentile=100.0,
    snr_combine="geometric",
    junctures_mode="both",
    phase2_only_junctures=False,
    junctures_at_phase_switch=True,
    phase2_juncture_warmup_epochs=None,
    prune_mode="global_param",
    global_prune_budget="neurons",
    global_prune_normalize="zscore",
    checkpoint_metric="val_loss_total",
):
    """
    Two-phase curriculum adaptive experiment in a single continuous training run.

    Phase 1 uses Fashion-only loaders; phase 2 switches to mixed Fashion+SVHN loaders
    at epoch phase1_epochs + 1. Training continues from the end-of-phase-1 model state.

    The best checkpoint is selected from phase 2 only, using the lowest value of
    checkpoint_metric on the mixed Fashion+SVHN validation set.

    checkpoint_metric : str
        "val_loss_nll" or "val_loss_total" for best-checkpoint selection.

    phase2_only_junctures : bool
        If True, skip structural junctures during phase 1 so the network stays fixed
        on the source distribution before the shift.
    junctures_at_phase_switch : bool
        When phase2_juncture_warmup_epochs is 0, schedule the first phase-2 juncture at
        the first phase-2 epoch. Ignored when warmup > 0.
    phase2_juncture_warmup_epochs : int or None
        Number of phase-2 epochs (after the loader switch) with weight-only training
        before junctures resume. None defaults to 20 when phase2_only_junctures else 0.
    prune_mode : str
        "per_layer" or "global_param" (layer-normalized global SNR ranking).
    global_prune_budget : str
        "params" or "neurons" (only when prune_mode="global_param").
    global_prune_normalize : str
        "percentile", "zscore", "mad", or "raw" (only when prune_mode="global_param").
        "mad" uses within-layer robust z-score (median/MAD).
        "raw" prunes the lowest-SNR neurons globally with no within-layer normalization.
    growth_layer_score : str
        "mean" or "mad" for which layer to expand on grow.
    growth_mad_percentile : float
        Percentile of within-layer MAD z-scores when growth_layer_score="mad" (100 = max).
    """
    if output_dir is None:
        output_dir = os.path.join("./results", experiment_name)
    ensure_output_dir(output_dir)

    total_epochs = phase1_epochs + phase2_epochs
    phase2_juncture_warmup_epochs = _resolve_phase2_juncture_warmup_epochs(
        phase2_juncture_warmup_epochs, phase2_only_junctures
    )
    phase2_start = _phase2_juncture_start_epoch(
        phase1_epochs, phase2_juncture_warmup_epochs, junctures_at_phase_switch
    )
    train_loader = loaders["phase1_train"]
    val_loader = loaders["phase1_val"]

    print("\n" + "=" * 50)
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric, phase2=True)
    print(
        f"Training {experiment_name.upper()} curriculum "
        f"(phase1={phase1_epochs}, phase2={phase2_epochs}, "
        f"lambda={lambda_penalty}, junctures_mode={junctures_mode}, "
        f"phase2_only_junctures={phase2_only_junctures}, "
        f"phase2_juncture_warmup_epochs={phase2_juncture_warmup_epochs}, "
        f"prune_mode={prune_mode}, global_prune_budget={global_prune_budget}, "
        f"global_prune_normalize={global_prune_normalize}, "
        f"checkpoint_metric={checkpoint_metric_label})"
    )
    print("=" * 50)
    if phase2_only_junctures:
        print("Structural junctures disabled in phase 1")
    if phase2_juncture_warmup_epochs > 0:
        print(
            f"Phase 2 juncture warmup: {phase2_juncture_warmup_epochs} epochs "
            f"(first juncture at epoch {phase2_start})"
        )
    elif phase2_only_junctures and junctures_at_phase_switch:
        print("First juncture scheduled at phase switch")

    param_stats = model.get_param_stats()
    print(f"Model parameters: {param_stats['total_params']:,}")
    print(f"Output classes: {model.out_features} (Fashion 0-9, SVHN 10-19)")

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
    }

    growth_cooldown = {}
    best_checkpoint_score = float("inf")
    best_model_state = None
    best_epoch = 0
    best_hidden_sizes = list(hidden_sizes)
    best_val_acc_at_checkpoint = 0.0
    best_val_brier_at_checkpoint = float("inf")

    def _annealed_decision_interval(epoch: int) -> int:
        E = max(1, total_epochs)
        frac = min(1.0, max(0.0, epoch / E))
        raw = decision_interval_min + (decision_interval_max - decision_interval_min) * (frac ** decision_interval_power)
        return int(math.floor(raw))

    def _advance_decision_epoch(epoch: int) -> int:
        if decision_interval is not None:
            return min(total_epochs, epoch + int(decision_interval))
        return min(total_epochs, epoch + _annealed_decision_interval(epoch))

    next_decision_epoch = decision_interval if decision_interval is not None else decision_interval_min
    if phase2_only_junctures or phase2_juncture_warmup_epochs > 0:
        next_decision_epoch = max(next_decision_epoch, phase2_start)

    for epoch in range(1, total_epochs + 1):
        if epoch == phase1_epochs + 1:
            print("\n" + "=" * 50)
            print(f"PHASE 2 START (epoch {epoch}): switching to Fashion + SVHN loaders")
            print("=" * 50)
            save_checkpoint(
                os.path.join(output_dir, "phase1_end_checkpoint.pth"),
                state_dict=copy.deepcopy(model.state_dict()),
                epoch=phase1_epochs,
                hidden_sizes=list(hidden_sizes),
                selection_metric="phase1_end",
            )
            train_loader = loaders["phase2_train"]
            val_loader = loaders["phase2_val"]
            beta_scaled = (1 / len(train_loader.dataset)) * beta
            print(
                f"Phase 2 dataset sizes: train={len(train_loader.dataset):,}, "
                f"val={len(val_loader.dataset):,}"
            )
            if phase2_juncture_warmup_epochs > 0:
                next_decision_epoch = phase2_start
            elif phase2_only_junctures and junctures_at_phase_switch:
                next_decision_epoch = epoch

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
        metrics["val_loss_total"].append(val_loss_total)
        metrics["val_loss_nll"].append(val_loss_nll)
        metrics["val_loss_kl"].append(val_loss_kl)
        metrics["val_loss_penalty"].append(val_loss_penalty)
        metrics["val_acc"].append(val_acc)
        metrics["val_brier"].append(val_brier)
        metrics["param_count_history"].append(count_params(model))

        print(
            f"Epoch {epoch} [phase {phase_id}]: "
            f"Train Loss={train_loss_total:.4f}, Train Acc={train_acc:.2f}%, "
            f"Val Loss={val_loss_total:.4f}, Val Acc={val_acc:.2f}%, "
            f"Params={count_params(model):,}"
        )

        val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
        if phase_id == 2 and _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
            best_checkpoint_score = val_score
            best_epoch = epoch
            best_hidden_sizes = list(hidden_sizes)
            best_val_acc_at_checkpoint = val_acc
            best_val_brier_at_checkpoint = val_brier
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                hidden_sizes=best_hidden_sizes,
                selection_metric=checkpoint_metric_label,
                selection_value=best_checkpoint_score,
            )

        if epoch == next_decision_epoch and epoch != total_epochs:
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
                metrics["structural_global_prune_normalize"].append(info.get("global_prune_normalize"))

                if action != "none":
                    optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
                    )

                if action == "grow" and growth_cooldown_junctures > 0 and junctures_mode in ("both", "grow"):
                    grown = info["grow_layer_idx"]
                    growth_cooldown[grown] = growth_cooldown_junctures
                    for i in list(growth_cooldown.keys()):
                        if i != grown:
                            growth_cooldown[i] -= 1
                            if growth_cooldown[i] <= 0:
                                del growth_cooldown[i]

                next_decision_epoch = _advance_decision_epoch(epoch)

    plot_curriculum_metrics(
        metrics,
        phase1_epochs=phase1_epochs,
        save_path=os.path.join(output_dir, "metrics.png"),
    )

    if best_model_state is not None:
        eval_model = BayesianFNN(
            model.in_features,
            best_hidden_sizes,
            model.out_features,
        ).to(device)
        eval_model.load_state_dict(best_model_state, strict=True)
        print(
            f"Loaded best phase-2 mixed-val model from epoch {best_epoch} "
            f"(val_acc={best_val_acc_at_checkpoint:.2f}%, hidden_sizes={best_hidden_sizes}) "
            f"for final testing."
        )
    else:
        print(
            "Warning: no phase-2 checkpoint was selected; using final training state for testing."
        )
        eval_model = model
        best_epoch = total_epochs
        best_checkpoint_score = _val_score_for_checkpoint(
            metrics["val_loss_total"][-1],
            metrics["val_loss_nll"][-1],
            checkpoint_metric,
        )
        best_val_acc_at_checkpoint = metrics["val_acc"][-1]
        best_val_brier_at_checkpoint = metrics["val_brier"][-1]

    fashion_beta = (1 / len(loaders["fashion_test"].dataset)) * beta
    svhn_beta = (1 / len(loaders["svhn_test"].dataset)) * beta
    mixed_beta = (1 / len(loaders["mixed_test"].dataset)) * beta

    _, fashion_test_acc, _, _, fashion_test_brier, _ = validate(
        eval_model, loaders["fashion_test"], device, fashion_beta, lambda_penalty
    )
    _, svhn_test_acc, _, _, svhn_test_brier, _ = validate(
        eval_model, loaders["svhn_test"], device, svhn_beta, lambda_penalty
    )
    mixed_test_loss, mixed_test_acc, _, _, mixed_test_brier, _ = validate(
        eval_model, loaders["mixed_test"], device, mixed_beta, lambda_penalty
    )

    print(
        f"Fashion test acc={fashion_test_acc:.2f}%, SVHN test acc={svhn_test_acc:.2f}%, "
        f"Mixed test acc={mixed_test_acc:.2f}%"
    )

    final_hidden_sizes = [layer.mu_w.shape[0] for layer in eval_model.layers]
    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else total_epochs,
        hidden_sizes=final_hidden_sizes,
    )

    metrics.update({
        "test_acc": mixed_test_acc,
        "test_loss_total": mixed_test_loss,
        "test_brier": mixed_test_brier,
        "fashion_test_acc": fashion_test_acc,
        "fashion_test_brier": fashion_test_brier,
        "svhn_test_acc": svhn_test_acc,
        "svhn_test_brier": svhn_test_brier,
        "param_count": count_params(eval_model),
        "trainable_param_count": eval_model.get_param_stats()["trainable_params"],
        "hidden_sizes": final_hidden_sizes,
    })

    write_curriculum_metrics_csv(output_dir, metrics, phase1_epochs)
    write_curriculum_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        hidden_sizes=metrics["hidden_sizes"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=best_val_acc_at_checkpoint,
        best_val_brier=best_val_brier_at_checkpoint,
        selected_checkpoint_metric=checkpoint_metric_label,
        fashion_test_acc=fashion_test_acc,
        fashion_test_brier=fashion_test_brier,
        svhn_test_acc=svhn_test_acc,
        svhn_test_brier=svhn_test_brier,
        mixed_test_acc=mixed_test_acc,
        mixed_test_brier=mixed_test_brier,
        lambda_penalty=lambda_penalty,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        junctures_mode=junctures_mode,
    )

    structural_df = pd.DataFrame({
        "epoch": metrics["structural_epochs"],
        "phase": [
            1 if e <= phase1_epochs else 2 for e in metrics["structural_epochs"]
        ],
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

    print(f"\n{experiment_name} curriculum summary:")
    print(
        f"Best phase-2 mixed validation accuracy (at selected checkpoint): "
        f"{best_val_acc_at_checkpoint:.2f}%"
    )
    print(f"Fashion / SVHN / Mixed test acc: "
          f"{fashion_test_acc:.2f}% / {svhn_test_acc:.2f}% / {mixed_test_acc:.2f}%")
    print(f"Final hidden sizes: {final_hidden_sizes}")
    print(f"Structural decisions: {metrics['structural_actions']}")

    return metrics, model, total_epochs


def main(
    save_path,
    hidden_sizes,
    lambda_penalty,
    junctures_mode="both",
    phase1_epochs=100,
    phase2_epochs=100,
    batch_size=1024,
    learning_rate=0.001,
    beta=0.001,
    gamma=0.1,
    rho=0.1,
    warm_start_steps=80,
    warm_start_lr=0.001,
    decision_interval_min=5,
    decision_interval_max=5,
    decision_interval_power=1.0,
    baseline=False,
    phase2_only_junctures=False,
    junctures_at_phase_switch=False,
    phase2_juncture_warmup_epochs=20,
    prune_mode="global_param",
    global_prune_budget="neurons",
    global_prune_normalize="mad",
    checkpoint_metric="val_loss_total",
    growth_layer_score="mad",
    growth_mad_percentile=90.0,
):
    os.makedirs(save_path, exist_ok=True)

    total_epochs = phase1_epochs + phase2_epochs
    resolved_warmup = _resolve_phase2_juncture_warmup_epochs(
        phase2_juncture_warmup_epochs, phase2_only_junctures
    )
    if baseline:
        lambda_penalty = 0.0
        decision_interval_min = total_epochs + 1
        decision_interval_max = total_epochs + 1
        experiment_name = "curriculum_baseline"
        junctures_mode = "both"  # valid value; junctures won't run if interval is disabled
        phase2_juncture_warmup_epochs = 0
        phase2_only_junctures = False
    else:
        experiment_name = "curriculum_plasticity"

    loaders = build_curriculum_dataloaders(batch_size=batch_size)

    print("\n" + "=" * 50)
    print("Curriculum setup" + (" (baseline)" if baseline else ""))
    print("=" * 50)
    print(f"Phase 1 ({phase1_epochs} epochs): FashionMNIST only")
    print(f"Phase 2 ({phase2_epochs} epochs): FashionMNIST + SVHN")
    print(f"Input dim: {INPUT_DIM}, classes: {NUM_CLASSES}")
    if baseline:
        print("Baseline: lambda_penalty=0, structural junctures disabled")
    elif phase2_only_junctures:
        print("Plasticity: structural junctures enabled from phase 2 only")
        if resolved_warmup > 0:
            phase2_start = _phase2_juncture_start_epoch(
                phase1_epochs, resolved_warmup, junctures_at_phase_switch
            )
            print(
                f"Phase 2 juncture warmup: {resolved_warmup} epochs "
                f"(first juncture at epoch {phase2_start})"
            )
    if not baseline:
        print(
            f"Pruning: prune_mode={prune_mode}, global_prune_budget={global_prune_budget}, "
            f"global_prune_normalize={global_prune_normalize}"
        )

    initial_model = BayesianFNN(INPUT_DIM, hidden_sizes, NUM_CLASSES).to(device)
    initial_state_dict = copy.deepcopy(initial_model.state_dict())

    suffix = "" if junctures_mode == "both" else f"_{junctures_mode}_only"
    if baseline:
        run_name = f"baseline_{hidden_sizes[0]}"
    else:
        run_name = f"plasticity_{hidden_sizes[0]}_{lambda_penalty:g}{suffix}"
    output_dir = os.path.join(save_path, run_name)

    model = BayesianFNN(INPUT_DIM, hidden_sizes, NUM_CLASSES).to(device)
    model.load_state_dict(initial_state_dict)

    run_curriculum_adaptive_experiment(
        experiment_name,
        model,
        list(hidden_sizes),
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
        junctures_at_phase_switch=junctures_at_phase_switch,
        phase2_juncture_warmup_epochs=phase2_juncture_warmup_epochs,
        prune_mode=prune_mode,
        global_prune_budget=global_prune_budget,
        global_prune_normalize=global_prune_normalize,
        checkpoint_metric=checkpoint_metric,
        growth_layer_score=growth_layer_score,
        growth_mad_percentile=growth_mad_percentile,
    )


if __name__ == "__main__":
    hidden_sizes = [200,200]
    phase1_epochs = 50
    phase2_epochs = 300

    # for r in [0.5]:
    #     for lambda_penalty in [2.5e-07,2.5e-06]:
    #         for junctures_mode in ["both"]:
    #             set_seed(SEED)
    #             for i in range(1, 6):
    #                 main(
    #                     f"results_curriculum_5/run_{i}",
    #                     hidden_sizes,
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                     phase1_epochs=phase1_epochs,
    #                     phase2_epochs=phase2_epochs,
    #                     baseline=False,
    #                 )

    # for r in [0.75,0.85]:
    #     for lambda_penalty in [1e-08]:
    #         for junctures_mode in ["both"]:
    #             set_seed(SEED)
    #             for i in range(1, 6):
    #                 main(
    #                     f"results_curriculum_5/run_{i}",
    #                     [int(hidden_size*r) for hidden_size in hidden_sizes],
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                     phase1_epochs=phase1_epochs,
    #                     phase2_epochs=phase2_epochs,
    #                     baseline=True,
    #                 )

    # for r in [0.5]:
    #     for lambda_penalty in [0,5e-07,5e-06,5e-05]:
    #         for junctures_mode in ["both"]:
    #             set_seed(SEED)
    #             for i in range(1, 6):
    #                 main(
    #                     f"results_curriculum3/run_{i}",
    #                     hidden_sizes,
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                     phase1_epochs=phase1_epochs,
    #                     phase2_epochs=phase2_epochs,
    #                     baseline=False,
    #                 )
    # print("All curriculum experiments completed.")

    plot_param_count_vs_test_acc(
        save_path="results_curriculum_5",
        experiments=[
            "baseline_30", "baseline_50", "baseline_70","baseline_110","baseline_150",
            "plasticity_200_1e-05", "plasticity_200_1e-06",
            "plasticity_200_1e-07", "plasticity_200_5e-06","plasticity_200_2.5e-06","plasticity_200_2.5e-07"
        ],
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        save_path_out="param_count_vs_test_acc_mixed_test_acc_cur5.png",
        y_col="Mixed Test Acc",
        title="(Fashion+SVHN) Test Accuracy vs Parameter Count"
    )
