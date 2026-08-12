"""
Bayesian DropNet (Tan & Motani) on Fashion-MNIST.

Dense Bayesian FNN with iterative structured neuron pruning:
  - Start from a configurable dense seed (default [500, 500]).
  - Each cycle: reset surviving neurons to θ₀, train with early stopping,
    then drop the lowest mean-|post-activation| neurons (global or layer-wise).
  - Stop when val_acc <= κ * original_val_acc (after at least one prune).
  - Report the last trained cycle model (no extra final retrain).

Backbone matches BayesianFNN (LayerNorm + SiLU). No λ param-count penalty.
"""

from __future__ import annotations

import copy
import json
import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from BayesianFNN import BayesianFNN
from main_plasticity import (
    SEED,
    build_dataloaders,
    device,
    ensure_output_dir,
    loss_function,
    set_seed,
    truncate_and_load_encoder_layer,
    write_experiment_summary_csv,
)

INPUT_DIM = 784
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Mean-field helpers (BayesianFNN forward is always stochastic)
# ---------------------------------------------------------------------------


def _linear_mean_field(layer, x):
    bias = layer.mu_b if layer.bias_flag else None
    return F.linear(x, layer.mu_w, bias)


def forward_mean_field(model: BayesianFNN, x):
    for layer, ln in zip(model.layers, model.ln_layers):
        x = _linear_mean_field(layer, x)
        x = ln(x)
        x = F.silu(x)
    return _linear_mean_field(model.out, x)


def count_params(model: BayesianFNN) -> int:
    return int(model.get_param_stats()["total_params"])


def hidden_widths(model: BayesianFNN):
    return [int(layer.out_features) for layer in model.layers]


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def train_epoch(model, loader, optimizer, device_, beta_scaled, epoch=None):
    model.train()
    total_loss = total_nll = total_kl = 0.0
    correct = total = 0
    desc = f"Epoch {epoch}" if epoch is not None else "Train"
    progress_bar = tqdm(loader, desc=desc)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device_), labels.to(device_)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, nll, kl = loss_function(outputs, labels, model.kl_loss(), beta_scaled)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_nll += nll.item()
        total_kl += kl.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        acc = 100.0 * correct / max(total, 1)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")
    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "nll": total_nll / n,
        "kl": total_kl / n,
        "acc": 100.0 * correct / max(total, 1),
    }


@torch.no_grad()
def eval_model(model, loader, device_, beta_scaled, desc="Validating"):
    model.eval()
    total_loss = total_nll = 0.0
    running_brier = 0.0
    correct = total = 0
    for inputs, labels in tqdm(loader, desc=desc):
        inputs, labels = inputs.to(device_), labels.to(device_)
        outputs = forward_mean_field(model, inputs)
        loss, nll, _ = loss_function(outputs, labels, model.kl_loss(), beta_scaled)
        total_loss += loss.item()
        total_nll += nll.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        probs = F.softmax(outputs, dim=1)
        one_hot = F.one_hot(labels, num_classes=outputs.size(1)).float()
        running_brier += torch.sum((probs - one_hot) ** 2, dim=1).sum().item()
    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "nll": total_nll / n,
        "acc": 100.0 * correct / max(total, 1),
        "brier": running_brier / max(total, 1),
    }


def make_optimizer(model, lr):
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def train_cycle(
    model,
    train_loader,
    val_loader,
    device_,
    beta_scaled,
    learning_rate,
    max_epochs,
    patience,
    global_epoch_start,
    metric_rows,
    cycle_idx,
    phase,
):
    """
    Train with early stopping on validation loss.
    Restores best-in-cycle weights before return.
    """
    optimizer = make_optimizer(model, learning_rate)
    best = None
    bad_epochs = 0
    global_epoch = global_epoch_start

    for local_epoch in range(1, max_epochs + 1):
        global_epoch += 1
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device_,
            beta_scaled,
            epoch=global_epoch,
        )
        val_stats = eval_model(model, val_loader, device_, beta_scaled)
        widths = hidden_widths(model)
        params = count_params(model)
        print(
            f"[{phase} c{cycle_idx}] epoch {local_epoch}: "
            f"train_acc={train_stats['acc']:.2f}% "
            f"val_acc={val_stats['acc']:.2f}% val_loss={val_stats['loss']:.4f} "
            f"val_brier={val_stats['brier']:.4f} params={params} widths={widths}"
        )
        metric_rows.append(
            {
                "epoch": global_epoch,
                "cycle": cycle_idx,
                "phase": phase,
                "train_acc": train_stats["acc"],
                "val_acc": val_stats["acc"],
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "val_nll": val_stats["nll"],
                "val_brier": val_stats["brier"],
                "param_count": params,
                "hidden_sizes": str(widths),
            }
        )

        if best is None or val_stats["loss"] < best["val_loss"]:
            best = {
                "state_dict": copy.deepcopy(model.state_dict()),
                "hidden_sizes": list(widths),
                "epoch": global_epoch,
                "val_acc": float(val_stats["acc"]),
                "val_loss": float(val_stats["loss"]),
                "val_nll": float(val_stats["nll"]),
                "val_brier": float(val_stats["brier"]),
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"  early stop at local epoch {local_epoch} "
                    f"(best val_loss={best['val_loss']:.4f})"
                )
                break

    model.load_state_dict(best["state_dict"])
    return {
        "model": model,
        "global_epoch": global_epoch,
        "val_acc": best["val_acc"],
        "val_loss": best["val_loss"],
        "val_nll": best["val_nll"],
        "val_brier": best["val_brier"],
        "best_state": best,
    }


# ---------------------------------------------------------------------------
# Activation scoring + prune selection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_activation_scores(model: BayesianFNN, loader, device_, max_batches=None):
    """
    Mean absolute post-SiLU activation per hidden neuron (mean-field).
    Returns list of 1D tensors, one per hidden layer.
    """
    model.eval()
    n_layers = len(model.layers)
    sums = [None] * n_layers
    n_seen = 0
    for b_idx, (inputs, _) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        inputs = inputs.to(device_)
        x = inputs
        batch_n = inputs.size(0)
        for i, (layer, ln) in enumerate(zip(model.layers, model.ln_layers)):
            x = _linear_mean_field(layer, x)
            x = ln(x)
            x = F.silu(x)
            abs_mean = x.abs().sum(dim=0)  # (width,)
            if sums[i] is None:
                sums[i] = abs_mean
            else:
                sums[i] = sums[i] + abs_mean
        n_seen += batch_n
    if n_seen == 0:
        raise RuntimeError("No batches available for activation scoring")
    return [s / float(n_seen) for s in sums]


def select_keep_indices(scores, prune_frac, mode):
    """
    DropNet minimum / minimum_layer selection.

    scores: list of 1D score tensors (higher = more important).
    Returns keep_local: list of LongTensors of indices to keep in each layer
            and dropped_per_layer counts.
    """
    if mode not in ("global", "layer"):
        raise ValueError(f"prune_mode must be 'global' or 'layer', got {mode!r}")
    if not (0.0 < prune_frac <= 1.0):
        raise ValueError(f"prune_frac must be in (0, 1], got {prune_frac}")

    device_ = scores[0].device
    n_layers = len(scores)
    widths = [int(s.numel()) for s in scores]

    if mode == "layer":
        keep_local = []
        dropped = []
        for s, n in zip(scores, widths):
            if n <= 1:
                keep_local.append(torch.arange(n, device=device_, dtype=torch.long))
                dropped.append(0)
                continue
            n_drop = min(max(1, int(math.ceil(prune_frac * n))), n - 1)
            n_keep = n - n_drop
            _, top_idx = torch.topk(s, k=n_keep, largest=True, sorted=False)
            keep_local.append(torch.sort(top_idx).values)
            dropped.append(n_drop)
        return keep_local, dropped

    # global: drop bottom ceil(p * N_total), but leave >=1 neuron per layer
    all_scores = []
    all_meta = []  # (layer, local_idx)
    for li, s in enumerate(scores):
        for j in range(s.numel()):
            all_scores.append(float(s[j].item()))
            all_meta.append((li, j))
    n_total = len(all_scores)
    if n_total <= n_layers:
        # already minimal
        keep_local = [
            torch.arange(w, device=device_, dtype=torch.long) for w in widths
        ]
        return keep_local, [0] * n_layers

    n_drop = min(max(1, int(math.ceil(prune_frac * n_total))), n_total - n_layers)
    order = sorted(range(n_total), key=lambda t: all_scores[t])  # ascending
    drop_set = set()
    for idx in order:
        if len(drop_set) >= n_drop:
            break
        li, _ = all_meta[idx]
        # would this leave layer empty?
        kept_in_layer = widths[li] - sum(1 for d in drop_set if all_meta[d][0] == li)
        if kept_in_layer <= 1:
            continue
        drop_set.add(idx)

    keep_local = []
    dropped = []
    for li, n in enumerate(widths):
        keep_js = [
            j
            for t, (l, j) in enumerate(all_meta)
            if l == li and t not in drop_set
        ]
        if not keep_js:
            keep_js = [int(torch.argmax(scores[li]).item())]
        keep_local.append(
            torch.tensor(sorted(keep_js), device=device_, dtype=torch.long)
        )
        dropped.append(n - len(keep_js))
    return keep_local, dropped


def rebuild_from_theta0(
    theta0,
    orig_keep,
    in_features,
    out_features,
    device_,
):
    """
    Build a BayesianFNN whose params are θ₀ restricted to surviving original indices.

    orig_keep: list of LongTensors, indices into the original dense network per layer.
    """
    keep_dict = {i: orig_keep[i].detach().cpu() for i in range(len(orig_keep))}
    hidden_sizes = [int(len(keep_dict[i])) for i in range(len(keep_dict))]
    model = BayesianFNN(in_features, hidden_sizes, out_features).to(device_)
    # truncate_and_load expects tensors / lists indexable on CPU state_dict
    truncate_and_load_encoder_layer(theta0, keep_dict, model)
    return model, hidden_sizes


def update_orig_keep(orig_keep, keep_local):
    """Map current-layer keep indices back to original neuron ids."""
    new_keep = []
    for prev, local in zip(orig_keep, keep_local):
        local_cpu = local.detach().cpu()
        new_keep.append(prev.detach().cpu()[local_cpu].clone())
    return new_keep


def can_prune(widths):
    return any(w > 1 for w in widths)


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------


def _write_metrics_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _maybe_update_best(best, cycle_best, model):
    """Track best validation accuracy across cycles (higher is better)."""
    if best is None or cycle_best["val_acc"] > best["val_acc"]:
        return {
            "state_dict": copy.deepcopy(model.state_dict()),
            "hidden_sizes": list(hidden_widths(model)),
            "epoch": cycle_best["epoch"],
            "val_acc": float(cycle_best["val_acc"]),
            "val_loss": float(cycle_best["val_loss"]),
            "val_nll": float(cycle_best["val_nll"]),
            "val_brier": float(cycle_best["val_brier"]),
        }
    return best


def run_dropnet_experiment(
    model,
    theta0,
    orig_keep,
    train_loader,
    val_loader,
    test_loader,
    output_dir,
    learning_rate=0.005,
    beta=0.002,
    kappa=0.99,
    prune_frac=0.2,
    prune_mode="global",
    max_epochs_per_cycle=100,
    early_stop_patience=5,
    max_prune_cycles=40,
    score_max_batches=None,
):
    """
    kappa:
      Stop pruning once val_acc <= kappa * original_val_acc (after >=1 prune).
    prune_mode:
      'global' (DropNet minimum) or 'layer' (minimum_layer).
    """
    ensure_output_dir(output_dir)
    kappa = float(kappa)
    if not (0.0 < kappa <= 1.0):
        raise ValueError(f"kappa must be in (0, 1], got {kappa}")

    beta_scaled = (1.0 / len(train_loader.dataset)) * beta
    metric_rows = []
    event_rows = []
    global_epoch = 0
    best_state = None
    original_val_acc = None
    n_prunes = 0
    cycle_idx = 0
    last_result = None

    print("\n" + "=" * 50)
    print("DropNet iterative prune")
    print("=" * 50)
    print(f"kappa={kappa}, prune_frac={prune_frac}, prune_mode={prune_mode}")

    # -------------------- Prune cycles --------------------
    while True:
        cycle_idx += 1
        phase = "cycle"
        result = train_cycle(
            model,
            train_loader,
            val_loader,
            device,
            beta_scaled,
            learning_rate,
            max_epochs_per_cycle,
            early_stop_patience,
            global_epoch,
            metric_rows,
            cycle_idx,
            phase,
        )
        last_result = result
        global_epoch = result["global_epoch"]
        a_prime = float(result["val_acc"])
        best_state = _maybe_update_best(best_state, result["best_state"], model)

        if original_val_acc is None:
            original_val_acc = a_prime
            acc_floor = kappa * original_val_acc
            print(
                f"Original val_acc a={original_val_acc:.4f}% "
                f"(floor κ·a={acc_floor:.4f}%)"
            )
        else:
            acc_floor = kappa * original_val_acc

        # Stop if below floor after at least one prune (do not prune further).
        if n_prunes >= 1 and a_prime <= acc_floor:
            print(
                f"Stop: val_acc={a_prime:.4f}% <= floor={acc_floor:.4f}% "
                f"after {n_prunes} prune(s)."
            )
            event_rows.append(
                {
                    "epoch": global_epoch,
                    "cycle": cycle_idx,
                    "event": "stop_kappa",
                    "val_acc": a_prime,
                    "acc_floor": acc_floor,
                    "original_val_acc": original_val_acc,
                    "hidden_sizes": str(hidden_widths(model)),
                    "param_count": count_params(model),
                }
            )
            break

        if n_prunes >= max_prune_cycles:
            print(f"Stop: reached max_prune_cycles={max_prune_cycles}.")
            break

        widths = hidden_widths(model)
        if not can_prune(widths):
            print(f"Stop: cannot prune further (widths={widths}).")
            break

        scores = collect_activation_scores(
            model, train_loader, device, max_batches=score_max_batches
        )
        keep_local, dropped = select_keep_indices(scores, prune_frac, prune_mode)
        if sum(dropped) == 0:
            print("Stop: prune selected zero neurons.")
            break

        orig_keep = update_orig_keep(orig_keep, keep_local)
        model, new_widths = rebuild_from_theta0(
            theta0,
            orig_keep,
            in_features=INPUT_DIM,
            out_features=NUM_CLASSES,
            device_=device,
        )
        n_prunes += 1
        event_rows.append(
            {
                "epoch": global_epoch,
                "cycle": cycle_idx,
                "event": "prune",
                "val_acc_before_prune": a_prime,
                "acc_floor": acc_floor,
                "dropped_per_layer": str(dropped),
                "hidden_sizes_after": str(new_widths),
                "param_count_after": count_params(model),
                "prune_mode": prune_mode,
                "prune_frac": prune_frac,
            }
        )
        print(
            f"  prune #{n_prunes}: dropped={dropped} → widths={new_widths} "
            f"params={count_params(model)}"
        )

    # Report the last trained cycle model (no extra θ₀ retrain).
    val_final = {
        "acc": float(last_result["val_acc"]),
        "loss": float(last_result["val_loss"]),
        "nll": float(last_result["val_nll"]),
        "brier": float(last_result["val_brier"]),
    }
    test_stats = eval_model(
        model, test_loader, device, beta_scaled, desc="Testing"
    )

    _write_metrics_csv(os.path.join(output_dir, "metrics.csv"), metric_rows)
    pd.DataFrame(event_rows).to_csv(
        os.path.join(output_dir, "structural_events.csv"), index=False
    )

    widths = hidden_widths(model)
    params = count_params(model)
    summary = write_experiment_summary_csv(
        output_dir,
        model_label="bayesian_dropnet",
        params=params,
        trainable_params=params,
        hidden_sizes=widths,
        best_epoch=global_epoch,
        best_val_total=val_final["loss"],
        best_val_acc=val_final["acc"],
        best_val_brier=val_final["brier"],
        test_acc=test_stats["acc"],
        test_brier=test_stats["brier"],
        lambda_penalty=0.0,
        selected_checkpoint_metric="val_acc",
        junctures_mode=f"dropnet_{prune_mode}",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_sizes": widths,
            "orig_keep": [k.cpu() for k in orig_keep],
        },
        os.path.join(output_dir, "final_model.pth"),
    )
    if best_state is not None:
        torch.save(best_state, os.path.join(output_dir, "best_checkpoint.pth"))

    with open(os.path.join(output_dir, "final_architecture.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "hidden_sizes": widths,
                "param_count": params,
                "kappa": float(kappa),
                "prune_frac": float(prune_frac),
                "prune_mode": prune_mode,
                "original_val_acc": float(original_val_acc),
                "acc_floor": float(kappa * original_val_acc),
                "n_prunes": int(n_prunes),
                "final_val_acc": float(val_final["acc"]),
                "final_val_loss": float(val_final["loss"]),
                "final_val_nll": float(val_final["nll"]),
                "final_val_brier": float(val_final["brier"]),
                "test_acc": float(test_stats["acc"]),
                "test_brier": float(test_stats["brier"]),
            },
            f,
            indent=2,
        )

    print("\nBayesian DropNet Summary:")
    print(f"  Final widths: {widths}")
    print(f"  Params: {params}")
    print(f"  Original val acc: {original_val_acc:.2f}%")
    print(f"  κ·a floor: {kappa * original_val_acc:.2f}%")
    print(f"  Prunes: {n_prunes}")
    print(f"  Val acc: {val_final['acc']:.2f}%")
    print(f"  Test acc: {test_stats['acc']:.2f}%")
    print(f"  Test brier: {test_stats['brier']:.4f}")

    return summary, model, metric_rows


def main(
    save_path,
    hidden_sizes=(500, 500),
    learning_rate=0.005,
    beta=0.002,
    batch_size=256,
    kappa=0.99,
    prune_frac=0.2,
    prune_mode="layer",
    max_epochs_per_cycle=100,
    early_stop_patience=5,
    max_prune_cycles=40,
):
    """
    Run Bayesian DropNet on Fashion-MNIST.

    hidden_sizes: dense starting widths (default DropNet-style large MLP).
    prune_mode: 'global' or 'layer'.
    kappa: stop when val_acc <= kappa * original_val_acc.
    """
    h = [max(1, int(w)) for w in hidden_sizes]
    if len(h) < 1:
        raise ValueError("hidden_sizes must be non-empty")
    if prune_mode not in ("global", "layer"):
        raise ValueError(f"prune_mode must be 'global' or 'layer', got {prune_mode!r}")

    os.makedirs(save_path, exist_ok=True)
    train_loader, val_loader, test_loader = build_dataloaders(
        "fashion_mnist", batch_size=batch_size, seed=SEED
    )

    model = BayesianFNN(INPUT_DIM, h, NUM_CLASSES).to(device)
    theta0 = copy.deepcopy(model.state_dict())
    orig_keep = [
        torch.arange(w, dtype=torch.long) for w in h
    ]

    run_name = (
        f"dropnet_{'_'.join(str(x) for x in h)}"
        f"_p{prune_frac:g}_mode{prune_mode}_kappa{kappa:g}"
    )
    output_dir = os.path.join(save_path, run_name)
    ensure_output_dir(output_dir)

    print("\n" + "=" * 50)
    print("Bayesian DropNet setup")
    print("=" * 50)
    print(f"Start widths: {h}")
    print(f"Params (dense): {count_params(model)}")
    print(f"prune_frac p: {prune_frac}")
    print(f"prune_mode: {prune_mode}")
    print(f"kappa: {kappa}")
    print(f"Output: {output_dir}")

    with open(os.path.join(output_dir, "dropnet_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "hidden_sizes": h,
                "prune_frac": float(prune_frac),
                "prune_mode": prune_mode,
                "kappa": float(kappa),
                "learning_rate": float(learning_rate),
                "beta": float(beta),
                "batch_size": int(batch_size),
                "max_epochs_per_cycle": int(max_epochs_per_cycle),
                "early_stop_patience": int(early_stop_patience),
                "max_prune_cycles": int(max_prune_cycles),
            },
            f,
            indent=2,
        )

    summary, model, _ = run_dropnet_experiment(
        model,
        theta0,
        orig_keep,
        train_loader,
        val_loader,
        test_loader,
        output_dir=output_dir,
        learning_rate=learning_rate,
        beta=beta,
        kappa=kappa,
        prune_frac=prune_frac,
        prune_mode=prune_mode,
        max_epochs_per_cycle=max_epochs_per_cycle,
        early_stop_patience=early_stop_patience,
        max_prune_cycles=max_prune_cycles,
    )
    return summary, model


if __name__ == "__main__":
    for prune_mode in ("global",):
        for kappa in (1,0.998,0.996,0.994,0.992,0.990):
            for i in range(1, 6):
                set_seed(SEED + i)
                main(
                    f"results_dropnet_FashionMnist_FNN_lenet/run_{i}",
                    hidden_sizes=(300, 100),
                    prune_frac=0.2,
                    prune_mode=prune_mode,
                    kappa=kappa,
                )
