from __future__ import annotations
import torch.nn.functional as F

import copy
import json
import math
import os

import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from lib.data import SUPPORTED_DATASETS, build_dataloaders, get_dataset_metadata
from lib.flops import theoretical_sparse_fnn_flops
from lib.seed import SEED, device, set_seed
from lib.train import ensure_output_dir, loss_function
from models.sparse_bayesian_fnn import (
    SparseBayesianFNN,
    cleanup_dead_neurons,
    connection_growth_step,
    init_seed_masks,
    neuron_growth_step,
    snr_prune_step,
)

def write_nest_experiment_summary_csv(
    output_dir,
    *,
    sparse_params,
    dense_params,
    flops,
    hidden_sizes,
    best_epoch,
    best_val_total,
    best_val_acc,
    best_val_brier,
    test_acc,
    test_brier,
):
    """Write Nest summary with sparse/dense variational param counts (no legacy Parameters)."""
    summary_df = pd.DataFrame(
        [
            {
                "Model": "bayesian_nest",
                "Sparse Parameters": int(sparse_params),
                "Dense Parameters": int(dense_params),
                "Best Val Acc": float(best_val_acc),
                "Best Val Brier": float(best_val_brier),
                "Test Acc": float(test_acc),
                "Test Brier": float(test_brier),
                "Hidden Sizes": str(list(hidden_sizes)),
                "FLOPs": int(flops),
                "Lambda Penalty": 0.0,
                "Junctures Mode": "nest",
                "Selected checkpoint metric": "val_acc",
                "Selected epoch": int(best_epoch),
                "Selected val_total": float(best_val_total),
            }
        ]
    )
    summary_df.to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)
    return summary_df.iloc[0].to_dict()

def train_epoch(model, loader, optimizer, device, beta_scaled, epoch=None):
    model.train()
    total_loss = total_nll = total_kl = 0.0
    correct = total = 0
    desc = f"Epoch {epoch}" if epoch is not None else "Train"
    progress_bar = tqdm(loader, desc=desc)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, nll, kl = loss_function(outputs, labels, model.kl_loss(), beta_scaled)
        loss.backward()
        # Zero grads on inactive weights so they stay dormant.
        with torch.no_grad():
            for layer in model.all_masked_layers():
                if layer.mu_w.grad is not None:
                    layer.mu_w.grad.mul_(layer.mask_w)
                if layer.rho_w.grad is not None:
                    layer.rho_w.grad.mul_(layer.mask_w)
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
def eval_model(model, loader, device, beta_scaled, mean_field=True, desc="Validating"):
    model.eval()
    total_loss = total_nll = 0.0
    running_brier = 0.0
    correct = total = 0
    for inputs, labels in tqdm(loader, desc=desc):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs, mean_field=mean_field)
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

def _write_metrics_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _maybe_update_best(best, val_stats, model, global_epoch):
    if best is None or val_stats["acc"] > best["val_acc"]:
        return {
            "state_dict": copy.deepcopy(model.state_dict()),
            "hidden_sizes": list(model.hidden_sizes),
            "epoch": global_epoch,
            "val_acc": float(val_stats["acc"]),
            "val_loss": float(val_stats["loss"]),
            "val_nll": float(val_stats["nll"]),
            "val_brier": float(val_stats["brier"]),
        }
    return best


def run_nest_experiment(
    model,
    train_loader,
    val_loader,
    test_loader,
    output_dir,
    learning_rate=0.005,
    beta=0.002,
    reference_acc=89.5,
    prune_acc_floor=None,
    max_growth_epochs=40,
    grow_interval=2,
    conn_grow_frac=0.01,
    neuron_grow_each_interval=True,
    beta_growth=0.4,
    birth_strength=0.4,
    prune_frac=0.01,
    max_prune_rounds=50,
    prune_retrain_epochs=3,
    growth_signal_batches=8,
):
    ensure_output_dir(output_dir)
    if prune_acc_floor is None:
        prune_acc_floor = float(reference_acc)

    beta_scaled = (1.0 / len(train_loader.dataset)) * beta
    optimizer = make_optimizer(model, learning_rate)

    metric_rows = []
    event_rows = []
    global_epoch = 0
    best_state = None

    # -------------------- Growth phase --------------------
    print("\n" + "=" * 50)
    print("NeST Growth Phase")
    print("=" * 50)
    print(f"reference_acc={reference_acc}%")
    print(f"prune_acc_floor={prune_acc_floor}%")
    phase = "growth"
    reached_ref = False

    for epoch in range(1, max_growth_epochs + 1):
        global_epoch += 1
        train_stats = train_epoch(
            model, train_loader, optimizer, device, beta_scaled, epoch=global_epoch
        )
        val_stats = eval_model(model, val_loader, device, beta_scaled)
        widths = model.hidden_widths()
        active = model.active_param_count()
        print(
            f"[growth] epoch {epoch}: train_acc={train_stats['acc']:.2f}% "
            f"val_acc={val_stats['acc']:.2f}% val_nll={val_stats['nll']:.4f} "
            f"val_brier={val_stats['brier']:.4f} "
            f"active={active} widths={widths}"
        )
        metric_rows.append(
            {
                "epoch": global_epoch,
                "phase": phase,
                "train_acc": train_stats["acc"],
                "val_acc": val_stats["acc"],
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "val_nll": val_stats["nll"],
                "val_brier": val_stats["brier"],
                "active_params": active,
                "hidden_sizes": str(widths),
                "sparsity": model.sparsity(),
            }
        )
        best_state = _maybe_update_best(best_state, val_stats, model, global_epoch)

        if val_stats["acc"] >= reference_acc:
            reached_ref = True
            print(
                f"Reference accuracy {reference_acc}% reached at growth "
                f"epoch {epoch} (val_acc={val_stats['acc']:.2f}%)."
            )
            break

        if epoch % grow_interval == 0:
            conn_info = connection_growth_step(
                model,
                train_loader,
                device,
                beta_scaled,
                conn_grow_frac=conn_grow_frac,
                max_batches=growth_signal_batches,
            )
            event_rows.append(
                {
                    "epoch": global_epoch,
                    "phase": phase,
                    "event": "grow_conn",
                    **{k: str(v) for k, v in conn_info.items()},
                }
            )
            print(f"  connection growth: +{conn_info['grown_connections']}")

            if neuron_grow_each_interval:
                # Alternate which hidden layer grows.
                h_idx = 0 if (epoch // grow_interval) % 2 == 1 else 1
                neur_info = neuron_growth_step(
                    model,
                    train_loader,
                    device,
                    beta_scaled,
                    hidden_layer_idx=h_idx,
                    beta_growth=beta_growth,
                    birth_strength=birth_strength,
                    max_batches=growth_signal_batches,
                )
                event_rows.append(
                    {
                        "epoch": global_epoch,
                        "phase": phase,
                        "event": "grow_neuron",
                        **{k: str(v) for k, v in neur_info.items()},
                    }
                )
                print(
                    f"  neuron growth layer {h_idx}: width→{neur_info['new_width']} "
                    f"(pairs={neur_info['selected_pairs']})"
                )
                optimizer = make_optimizer(model, learning_rate)

    if not reached_ref:
        best_acc = best_state["val_acc"] if best_state is not None else float("nan")
        print(
            f"Growth finished without reaching reference_acc={reference_acc}% "
            f"(best val_acc={best_acc:.2f}%). Proceeding to prune anyway."
        )

    # -------------------- Prune phase --------------------
    print("\n" + "=" * 50)
    print("NeST Prune Phase (SNR)")
    print("=" * 50)
    phase = "prune"

    for round_i in range(1, max_prune_rounds + 1):
        # Snapshot before prune to allow rollback if floor violated.
        pre_masks = [layer.mask_w.clone() for layer in model.all_masked_layers()]
        pre_state = copy.deepcopy(model.state_dict())
        pre_hidden = list(model.hidden_sizes)

        prune_info = snr_prune_step(model, prune_frac=prune_frac)
        if prune_info["pruned_connections"] == 0:
            print("No more connections to prune.")
            break

        optimizer = make_optimizer(model, learning_rate)
        for _ in range(prune_retrain_epochs):
            global_epoch += 1
            train_stats = train_epoch(
                model, train_loader, optimizer, device, beta_scaled, epoch=global_epoch
            )
            val_stats = eval_model(model, val_loader, device, beta_scaled)
            widths = model.hidden_widths()
            metric_rows.append(
                {
                    "epoch": global_epoch,
                    "phase": phase,
                    "train_acc": train_stats["acc"],
                    "val_acc": val_stats["acc"],
                    "train_loss": train_stats["loss"],
                    "val_loss": val_stats["loss"],
                    "val_nll": val_stats["nll"],
                    "val_brier": val_stats["brier"],
                    "active_params": model.active_param_count(),
                    "hidden_sizes": str(widths),
                    "sparsity": model.sparsity(),
                }
            )

        val_stats = eval_model(model, val_loader, device, beta_scaled)
        print(
            f"[prune] round {round_i}: pruned={prune_info['pruned_connections']} "
            f"val_acc={val_stats['acc']:.2f}% val_nll={val_stats['nll']:.4f} "
            f"val_brier={val_stats['brier']:.4f} "
            f"active={model.active_param_count()} widths={model.hidden_widths()}"
        )
        event_rows.append(
            {
                "epoch": global_epoch,
                "phase": phase,
                "event": "prune",
                **{k: str(v) for k, v in prune_info.items()},
                "val_acc_after": val_stats["acc"],
                "val_nll_after": val_stats["nll"],
                "val_brier_after": val_stats["brier"],
            }
        )

        if val_stats["acc"] < prune_acc_floor:
            print(
                f"Val acc {val_stats['acc']:.2f}% < floor {prune_acc_floor}%; "
                "rolling back last prune."
            )
            model.load_state_dict(pre_state)
            model.hidden_sizes = list(pre_hidden)
            for layer, m in zip(model.all_masked_layers(), pre_masks):
                layer.mask_w.copy_(m)
            break

        dead = cleanup_dead_neurons(model)
        if dead["neurons_removed"] > 0:
            print(f"  cleanup: removed {dead['neurons_removed']} dead neurons → {dead['hidden_sizes']}")
            event_rows.append(
                {
                    "epoch": global_epoch,
                    "phase": phase,
                    "event": "cleanup",
                    **{k: str(v) for k, v in dead.items()},
                }
            )
            optimizer = make_optimizer(model, learning_rate)

        best_state = _maybe_update_best(best_state, val_stats, model, global_epoch)

    test_stats = eval_model(
        model, test_loader, device, beta_scaled, desc="Testing"
    )
    val_final = eval_model(
        model, val_loader, device, beta_scaled, desc="Final val"
    )

    if best_state is None:
        best_state = {
            "epoch": global_epoch,
            "val_acc": float(val_final["acc"]),
            "val_loss": float(val_final["loss"]),
            "val_nll": float(val_final["nll"]),
            "val_brier": float(val_final["brier"]),
            "hidden_sizes": list(model.hidden_sizes),
            "state_dict": copy.deepcopy(model.state_dict()),
        }

    sparse_params = model.sparse_param_count()
    dense_params = model.dense_param_count()
    active_weights = sum(
        layer.active_weight_count() for layer in model.all_masked_layers()
    )
    flops = theoretical_sparse_fnn_flops(active_weights)
    summary = write_nest_experiment_summary_csv(
        output_dir,
        sparse_params=sparse_params,
        dense_params=dense_params,
        flops=flops,
        hidden_sizes=model.hidden_widths(),
        best_epoch=int(global_epoch),
        best_val_total=float(val_final["loss"]),
        best_val_acc=float(val_final["acc"]),
        best_val_brier=float(val_final["brier"]),
        test_acc=float(test_stats["acc"]),
        test_brier=float(test_stats["brier"]),
    )

    _write_metrics_csv(os.path.join(output_dir, "metrics.csv"), metric_rows)
    _write_metrics_csv(os.path.join(output_dir, "structural_events.csv"), event_rows)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_sizes": model.hidden_sizes,
            "masks": [layer.mask_w.cpu() for layer in model.all_masked_layers()],
        },
        os.path.join(output_dir, "final_model.pth"),
    )
    if best_state is not None:
        torch.save(best_state, os.path.join(output_dir, "best_checkpoint.pth"))

    with open(os.path.join(output_dir, "final_architecture.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "hidden_sizes": model.hidden_widths(),
                "active_params": model.active_param_count(),
                "sparse_params": sparse_params,
                "dense_params": dense_params,
                "total_variational_params": dense_params,
                "sparsity": model.sparsity(),
                "per_layer_active_weights": [
                    layer.active_weight_count() for layer in model.all_masked_layers()
                ],
                "reference_acc": float(reference_acc),
                "prune_acc_floor": float(prune_acc_floor),
                "reached_reference": bool(reached_ref),
                "final_val_acc": float(val_final["acc"]),
                "final_val_loss": float(val_final["loss"]),
                "final_val_nll": float(val_final["nll"]),
                "final_val_brier": float(val_final["brier"]),
            },
            f,
            indent=2,
        )

    print("\nBayesian NeST Summary:")
    print(f"  Final widths: {model.hidden_widths()}")
    print(f"  Active connections: {model.active_param_count()}")
    print(f"  Sparse params: {sparse_params}")
    print(f"  Dense params: {dense_params}")
    print(f"  FLOPs: {flops}")
    print(f"  Sparsity: {model.sparsity():.3f}")
    print(f"  Val acc: {val_final['acc']:.2f}%")
    print(f"  Test acc: {test_stats['acc']:.2f}%")
    print(f"  Test brier: {test_stats['brier']:.4f}")

    return summary, model, metric_rows


def main(
    save_path,
    hidden_sizes=(300, 100),
    seed_activate_frac=0.1,
    seed_scale=1.0,
    learning_rate=None,
    beta=0.002,
    batch_size=None,
    reference_acc=90.0,
    prune_acc_floor=None,
    max_growth_epochs=40,
    grow_interval=1,
    conn_grow_frac=0.01,
    beta_growth=0.4,
    birth_strength=0.4,
    prune_frac=0.01,
    max_prune_rounds=50,
    prune_retrain_epochs=3,
    dataset="fashion_mnist",
):
    """
    dataset options:
      - "fashion_mnist": Fashion-MNIST (default)
      - "kmnist": Kuzushiji-MNIST
      - "cifar10": CIFAR-10 greyscale (1024-dim input)

    When using cifar10, set reference_acc and prune_acc_floor to values
    appropriate for that dataset (Fashion-MNIST thresholds like 89% will not
    work sensibly).
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {sorted(SUPPORTED_DATASETS)}, got {dataset!r}"
        )
    dataset_meta = get_dataset_metadata(dataset)
    input_dim = dataset_meta["input_dim"]
    num_classes = dataset_meta["num_classes"]
    if learning_rate is None:
        learning_rate = 0.005 if dataset == "fashion_mnist" else 0.001
    if batch_size is None:
        batch_size = 256 if dataset == "fashion_mnist" else 128

    h = [max(1, int(round(w * seed_scale))) for w in hidden_sizes]
    if len(h) != 2:
        raise ValueError("hidden_sizes must have length 2")

    os.makedirs(save_path, exist_ok=True)
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, batch_size=batch_size, seed=SEED
    )
    print(
        f"Dataset: {dataset} "
        f"(train={len(train_loader.dataset):,}, "
        f"val={len(val_loader.dataset):,}, "
        f"test={len(test_loader.dataset):,})"
    )

    model = SparseBayesianFNN(input_dim, h, num_classes).to(device)
    g = torch.Generator()
    g.manual_seed(SEED)
    init_seed_masks(model, activate_frac=seed_activate_frac, generator=g)

    if prune_acc_floor is None:
        prune_acc_floor = float(reference_acc)
    else:
        prune_acc_floor = float(prune_acc_floor)

    run_name = (
        f"nest_{h[0]}_{h[1]}_p{seed_activate_frac:g}"
        f"_refacc{reference_acc:g}_flooracc{prune_acc_floor:g}"
    )
    output_dir = os.path.join(save_path, run_name)
    ensure_output_dir(output_dir)

    print("\n" + "=" * 50)
    print("Bayesian NeST setup")
    print("=" * 50)
    print(f"Dataset: {dataset}")
    print(f"Seed widths: {h}")
    print(f"Seed activate frac: {seed_activate_frac}")
    print(f"Active params (seed): {model.active_param_count()}")
    print(f"Sparsity (seed): {model.sparsity():.3f}")
    print(f"Learning rate: {learning_rate}")
    print(f"Reference acc: {reference_acc}%")
    print(f"Prune acc floor: {prune_acc_floor}%")
    print(f"Output: {output_dir}")

    with open(os.path.join(output_dir, "nest_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "input_dim": int(input_dim),
                "num_classes": int(num_classes),
                "seed_hidden_sizes": h,
                "seed_activate_frac": float(seed_activate_frac),
                "reference_acc": float(reference_acc),
                "prune_acc_floor": float(prune_acc_floor),
                "learning_rate": float(learning_rate),
                "beta": float(beta),
                "batch_size": int(batch_size),
            },
            f,
            indent=2,
        )

    summary, model, _ = run_nest_experiment(
        model,
        train_loader,
        val_loader,
        test_loader,
        output_dir=output_dir,
        learning_rate=learning_rate,
        beta=beta,
        reference_acc=reference_acc,
        prune_acc_floor=prune_acc_floor,
        max_growth_epochs=max_growth_epochs,
        grow_interval=grow_interval,
        conn_grow_frac=conn_grow_frac,
        beta_growth=beta_growth,
        birth_strength=birth_strength,
        prune_frac=prune_frac,
        max_prune_rounds=max_prune_rounds,
        prune_retrain_epochs=prune_retrain_epochs,
    )
    return summary, model


if __name__ == "__main__":

    # for prune_acc_floor in [89.0, 88.5, 88.0, 87.5, 87.0]:
    #     for i in range(1, 6):
    #         set_seed(SEED + i)
    #         main(
    #             f"results_fashionmnist/run_{i}",
    #             hidden_sizes=(300, 100),
    #             dataset="fashion_mnist",
    #             seed_scale=1,
    #             seed_activate_frac=0.1,
    #             reference_acc=89.0,
    #             prune_acc_floor=prune_acc_floor,
    #             max_growth_epochs=60,
    #             grow_interval=2,
    #             conn_grow_frac=0.01,
    #             beta_growth=0.4,
    #             birth_strength=0.4,
    #             prune_frac=0.01,
    #             max_prune_rounds=200,
    #             prune_retrain_epochs=2,
    #         )

    for prune_acc_floor in [44.0, 43.5, 43.0, 42.5, 42.0]:
        for i in range(1, 6):
            set_seed(SEED + i)
            main(
                f"results_cifar10_plasticity_0.1_ws64_gn32/run_{i}",
                hidden_sizes=(300, 100),
                dataset="cifar10",
                seed_scale=1,
                seed_activate_frac=0.1,
                reference_acc=44.0,
                prune_acc_floor=prune_acc_floor,
                max_growth_epochs=60,
                grow_interval=2,
                conn_grow_frac=0.01,
                beta_growth=0.4,
                birth_strength=0.4,
                prune_frac=0.01,
                max_prune_rounds=500,
                prune_retrain_epochs=2,
            )


