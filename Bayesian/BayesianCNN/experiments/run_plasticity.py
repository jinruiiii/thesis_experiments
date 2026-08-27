"""
Bayesian CNN plasticity experiments (baseline / plasticity / three_phase / static_replay).
"""
import ast
import copy
import json
import math
import os

import pandas as pd
import torch
import torch.optim as optim

from lib.data import DATASET_CONFIGS, build_dataloaders
from lib.plot import _parse_experiment_dir_name, plot_metrics
from lib.plasticity import (
    _conv_channels_from_model,
    build_grown_model_at_conv_layer,
    build_three_phase_pruned_model,
    expand_and_load_conv_stack,
    filterapoptosis,
    filtergenesis,
    structural_decision_juncture,
)
from lib.seed import SEED, device, set_seed
from lib.train import (
    _checkpoint_metric_label,
    _initialise_standard_metrics,
    _is_better_checkpoint_score,
    _normalize_checkpoint_metric,
    _val_score_for_checkpoint,
    count_params,
    ensure_output_dir,
    save_checkpoint,
    train,
    validate,
    write_experiment_summary_csv,
    write_metrics_csv,
)
from models.bayesian_cnn import BayesianCNN


def _make_eval_model(model, conv_channels):
    return BayesianCNN(
        model.in_channels,
        conv_channels,
        model.num_classes,
        kernel_size=model.kernel_size,
        padding=model.padding,
        stride=model.stride,
        fc_hidden=getattr(model, "fc_hidden", model.fc.out_features),
    ).to(device)


def _experiment_dir_suffix(junctures_mode):
    return "" if junctures_mode == "both" else f"_{junctures_mode}_only"


def _format_lambda_dir(lambda_penalty):
    return f"{float(lambda_penalty):g}"


def _conv_channels_tag(conv_channels):
    return "_".join(f"{c}f" for c in conv_channels)


def _last_structural_epoch(num_epochs, decision_cooldown_epochs=0):
    return max(0, num_epochs - max(0, int(decision_cooldown_epochs)))


def _annealed_decision_interval_length(
    epoch,
    num_epochs,
    decision_interval_min,
    decision_interval_max,
    decision_interval_power,
    decision_warmup_epochs=0,
    decision_cooldown_epochs=0,
):
    structural_epochs = max(
        1,
        num_epochs - decision_warmup_epochs - max(0, int(decision_cooldown_epochs)),
    )
    frac = min(1.0, max(0.0, (epoch - decision_warmup_epochs) / structural_epochs))
    raw = (
        decision_interval_min
        + (decision_interval_max - decision_interval_min) * (frac ** decision_interval_power)
    )
    return max(1, int(math.floor(raw)))


def _initial_decision_epoch(
    num_epochs,
    decision_interval,
    decision_interval_min,
    decision_warmup_epochs=0,
):
    warmup = max(0, int(decision_warmup_epochs))
    if warmup >= num_epochs:
        return num_epochs
    if decision_interval is not None:
        return warmup + int(decision_interval)
    return warmup + decision_interval_min


def _resolve_plasticity_dir_for_static_replay(resume_from_plasticity_dir):
    plasticity_dir = os.path.normpath(str(resume_from_plasticity_dir))
    if not os.path.isdir(plasticity_dir):
        raise FileNotFoundError(f"Plasticity directory not found: {plasticity_dir}")
    summary_path = os.path.join(plasticity_dir, "experiment_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"experiment_summary.csv not found in plasticity directory: {plasticity_dir}"
        )
    return plasticity_dir, summary_path


def _parse_conv_channels_from_summary(summary_path):
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"Empty experiment summary: {summary_path}")
    if "Conv Channels" not in df.columns:
        raise ValueError(f"'Conv Channels' column missing in {summary_path}")
    raw = df.iloc[0]["Conv Channels"]
    if pd.isna(raw):
        raise ValueError(f"'Conv Channels' is empty in {summary_path}")
    sizes = ast.literal_eval(str(raw))
    if not isinstance(sizes, (list, tuple)) or not sizes:
        raise ValueError(f"Invalid Conv Channels value {raw!r} in {summary_path}")
    sizes = [int(s) for s in sizes]
    if any(s <= 0 for s in sizes):
        raise ValueError(f"Conv Channels must be positive ints, got {sizes} from {summary_path}")
    return sizes


def _static_replay_output_dir(plasticity_output_dir):
    plasticity_dir = os.path.normpath(plasticity_output_dir)
    plasticity_dir_name = os.path.basename(plasticity_dir)
    if not plasticity_dir_name.startswith("plasticity_"):
        raise ValueError(
            f"Expected plasticity_* directory name for static replay, got {plasticity_dir_name!r}"
        )
    return os.path.join(
        os.path.dirname(plasticity_dir),
        "static_replay_" + plasticity_dir_name[len("plasticity_"):],
    )


def write_static_replay_provenance(output_dir, provenance):
    path = os.path.join(output_dir, "static_replay_provenance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    return path

def run_experiment(
    experiment_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    learning_rate=0.005,
    start_epoch=1,
    metrics=None,
    beta=0.1,
    lambda_penalty=0,
    output_dir=None,
    checkpoint_metric="val_loss_total",
    junctures_mode="both",
    summary_lambda_penalty=None,
):
    if output_dir is None:
        output_dir = os.path.join("./results", experiment_name)
    ensure_output_dir(output_dir)

    print(f"\n{'-'*20} Running {experiment_name} experiment {'-'*20}")
    param_stats = model.get_param_stats()
    print(f"Model parameters: {param_stats['total_params']:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    if not metrics:
        metrics = _initialise_standard_metrics()

    loss_label = "Penalised ELBO" if lambda_penalty > 0 else "ELBO"
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    best_checkpoint_score = float("inf")
    best_model_state = None
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    csv_lambda_penalty = (
        float(summary_lambda_penalty)
        if summary_lambda_penalty is not None
        else float(lambda_penalty)
    )
    last_epoch = best_epoch = start_epoch - 1

    for epoch in range(start_epoch, start_epoch + num_epochs):
        last_epoch = epoch
        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty = train(
            model, train_loader, optimizer, epoch, device, beta_scaled, lambda_penalty
        )
        metrics["train_loss_total"].append(train_loss_total)
        metrics["train_loss_nll"].append(train_loss_nll)
        metrics["train_loss_kl"].append(train_loss_kl)
        metrics["train_acc"].append(train_acc)
        metrics["train_brier"].append(train_brier)
        metrics["train_loss_penalty"].append(train_loss_penalty)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics["val_loss_total"].append(val_loss_total)
        metrics["val_loss_nll"].append(val_loss_nll)
        metrics["val_loss_kl"].append(val_loss_kl)
        metrics["val_acc"].append(val_acc)
        metrics["val_brier"].append(val_brier)
        metrics["val_loss_penalty"].append(val_loss_penalty)
        metrics["param_count_history"].append(count_params(model))

        penalty_str = f", Penalty={train_loss_penalty:.4f}" if lambda_penalty > 0 else ""
        print(
            f"Epoch {epoch}: Train Loss({loss_label})={train_loss_total:.4f}, "
            f"Train Loss(NLL)={train_loss_nll:.4f}, Train Loss(KL)={train_loss_kl:.4f}"
            f"{penalty_str}, Train Acc={train_acc:.2f}%, "
            f"Val Loss({loss_label})={val_loss_total:.4f}, Val Acc={val_acc:.2f}%"
        )

        val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
        if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
            best_checkpoint_score = val_score
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                model=model,
                selection_metric=checkpoint_metric_label,
                selection_value=best_checkpoint_score,
            )

    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, "metrics.png"),
    )

    eval_model = model
    if best_model_state is not None:
        eval_model = copy.deepcopy(model)
        eval_model.load_state_dict(best_model_state)
        print(f"Loaded best model from Epoch {best_epoch} for final testing.")

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f"Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}")

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else last_epoch,
        model=eval_model,
    )

    conv_channels = _conv_channels_from_model(eval_model)
    metrics.update({
        "test_acc": test_acc,
        "test_loss_total": test_loss_total,
        "test_loss_nll": test_loss_nll,
        "test_loss_kl": test_loss_kl,
        "test_brier": test_brier,
        "param_count": count_params(eval_model),
        "trainable_param_count": eval_model.get_param_stats()["trainable_params"],
        "conv_channels": conv_channels,
    })

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        conv_channels=metrics["conv_channels"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=csv_lambda_penalty,
        selected_checkpoint_metric=checkpoint_metric_label,
        junctures_mode=junctures_mode,
    )

    epochs_run = last_epoch - start_epoch + 1
    return metrics, model, epochs_run


def run_adaptive_experiment(
    experiment_name,
    model,
    conv_channels,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    learning_rate,
    beta,
    lambda_penalty,
    gamma,
    rho,
    warm_start_steps,
    warm_start_lr,
    decision_interval=None,
    decision_interval_min=2,
    decision_interval_max=20,
    decision_interval_power=2.0,
    decision_warmup_epochs=0,
    decision_cooldown_epochs=0,
    output_dir=None,
    growth_cooldown_junctures=0,
    uncertainty_combine="mean",
    growth_layer_score="mean",
    growth_mad_percentile=100.0,
    snr_combine="geometric",
    junctures_mode="both",
    prune_mode="per_layer",
    global_prune_budget="params",
    global_prune_normalize="percentile",
    checkpoint_metric="val_loss_total",
    grow_new_only_steps=None,
):
    if output_dir is None:
        output_dir = os.path.join("./results", experiment_name)
    ensure_output_dir(output_dir)

    print("\n" + "=" * 50)
    print(
        f"Training {experiment_name.upper()} CNN (Penalised ELBO, lambda={lambda_penalty}, "
        f"junctures_mode={junctures_mode})"
    )
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    print(f"Checkpoint selection metric: {checkpoint_metric_label}")
    resolved_grow_new_only_steps = (
        warm_start_steps if grow_new_only_steps is None else int(grow_new_only_steps)
    )
    print(
        f"Grow warm-start: new_only_steps={resolved_grow_new_only_steps}/{warm_start_steps} "
        f"(remaining steps update all params)"
    )
    print("=" * 50)

    param_stats = model.get_param_stats()
    print(f"Model parameters: {param_stats['total_params']:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    metrics = _initialise_standard_metrics()
    metrics.update({
        "structural_epochs": [],
        "structural_actions": [],
        "structural_delta_grow": [],
        "structural_delta_prune": [],
        "structural_delta_grow_prune": [],
        "structural_L_before": [],
        "structural_L_after_none": [],
        "structural_conv_channels": [],
        "structural_prune_mode": [],
        "structural_prune_target_remove": [],
        "structural_prune_params_removed": [],
        "structural_prune_filters_pruned": [],
        "structural_global_prune_budget": [],
        "structural_global_prune_normalize": [],
        "structural_grow_new_only_steps": [],
        "junctures_mode": junctures_mode,
    })

    growth_cooldown = {}
    best_checkpoint_score = float("inf")
    best_model_state = None
    best_epoch = 0
    best_conv_channels = list(conv_channels)
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    decision_warmup_epochs = max(0, int(decision_warmup_epochs))
    decision_cooldown_epochs = max(0, int(decision_cooldown_epochs))
    last_structural_epoch = _last_structural_epoch(num_epochs, decision_cooldown_epochs)
    next_decision_epoch = min(
        last_structural_epoch,
        _initial_decision_epoch(
            num_epochs,
            decision_interval,
            decision_interval_min,
            decision_warmup_epochs,
        ),
    )

    for epoch in range(1, num_epochs + 1):
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
            f"Epoch {epoch}: Train Loss(Penalised ELBO)={train_loss_total:.4f}, "
            f"Train Acc={train_acc:.2f}%, Params={count_params(model):,}, "
            f"Val Loss(Penalised ELBO)={val_loss_total:.4f}, Val Acc={val_acc:.2f}%"
        )

        val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
        if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
            best_checkpoint_score = val_score
            best_epoch = epoch
            best_conv_channels = list(conv_channels)
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                model=model,
                conv_channels=best_conv_channels,
                selection_metric=checkpoint_metric_label,
                selection_value=best_checkpoint_score,
            )

        if (
            epoch == next_decision_epoch
            and epoch <= last_structural_epoch
            and epoch > decision_warmup_epochs
        ):
            print("-" * 20 + f" Decision juncture (epoch {epoch}) " + "-" * 20)
            exclude_layers = [i for i, rem in growth_cooldown.items() if rem > 0]
            action, model, conv_channels, info = structural_decision_juncture(
                model,
                conv_channels,
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
                grow_new_only_steps=grow_new_only_steps,
            )
            metrics["structural_epochs"].append(epoch)
            metrics["structural_actions"].append(action)
            metrics["structural_delta_grow"].append(info.get("delta_grow"))
            metrics["structural_delta_prune"].append(info.get("delta_prune"))
            metrics["structural_delta_grow_prune"].append(info.get("delta_grow_prune"))
            metrics["structural_L_before"].append(info.get("L_before"))
            metrics["structural_L_after_none"].append(info.get("L_after_none"))
            metrics["structural_conv_channels"].append(list(conv_channels))
            metrics["structural_prune_mode"].append(info.get("prune_mode"))
            metrics["structural_prune_target_remove"].append(info.get("prune_target_remove"))
            metrics["structural_prune_params_removed"].append(info.get("prune_params_removed"))
            metrics["structural_prune_filters_pruned"].append(info.get("prune_neurons_pruned"))
            metrics["structural_global_prune_budget"].append(info.get("global_prune_budget"))
            metrics["structural_global_prune_normalize"].append(
                info.get("global_prune_normalize")
            )
            metrics["structural_grow_new_only_steps"].append(
                info.get("grow_new_only_steps")
            )

            if action != "none":
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
                )

            if (
                action in ("grow", "grow_prune")
                and growth_cooldown_junctures > 0
                and junctures_mode in ("both", "grow", "both_gp")
            ):
                grown = info["grow_layer_idx"]
                growth_cooldown[grown] = growth_cooldown_junctures
                for i in list(growth_cooldown.keys()):
                    if i != grown:
                        growth_cooldown[i] -= 1
                        if growth_cooldown[i] <= 0:
                            del growth_cooldown[i]

            if decision_interval is not None:
                next_decision_epoch = min(
                    last_structural_epoch, epoch + int(decision_interval)
                )
            else:
                gap = _annealed_decision_interval_length(
                    epoch,
                    num_epochs,
                    decision_interval_min,
                    decision_interval_max,
                    decision_interval_power,
                    decision_warmup_epochs,
                    decision_cooldown_epochs,
                )
                next_decision_epoch = min(last_structural_epoch, epoch + gap)

    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, "metrics.png"),
    )

    if best_model_state is not None:
        eval_model = _make_eval_model(model, best_conv_channels)
        eval_model.load_state_dict(best_model_state, strict=True)
        print(
            f"Loaded best model from epoch {best_epoch} "
            f"(conv_channels={best_conv_channels}) for final testing."
        )
    else:
        eval_model = model

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f"Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}")

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else num_epochs,
        model=eval_model,
    )

    final_conv_channels = _conv_channels_from_model(eval_model)
    metrics.update({
        "test_acc": test_acc,
        "test_loss_total": test_loss_total,
        "test_loss_nll": test_loss_nll,
        "test_loss_kl": test_loss_kl,
        "test_brier": test_brier,
        "param_count": count_params(eval_model),
        "trainable_param_count": eval_model.get_param_stats()["trainable_params"],
        "conv_channels": final_conv_channels,
    })

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        conv_channels=metrics["conv_channels"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=lambda_penalty,
        selected_checkpoint_metric=checkpoint_metric_label,
        junctures_mode=junctures_mode,
    )

    structural_df = pd.DataFrame({
        "epoch": metrics["structural_epochs"],
        "action": metrics["structural_actions"],
        "delta_grow": metrics["structural_delta_grow"],
        "delta_prune": metrics["structural_delta_prune"],
        "delta_grow_prune": metrics["structural_delta_grow_prune"],
        "L_before": metrics["structural_L_before"],
        "L_after_none": metrics["structural_L_after_none"],
        "conv_channels": [str(cc) for cc in metrics["structural_conv_channels"]],
        "junctures_mode": metrics["junctures_mode"],
        "prune_mode": metrics["structural_prune_mode"],
        "prune_target_remove": metrics["structural_prune_target_remove"],
        "prune_params_removed": metrics["structural_prune_params_removed"],
        "filters_pruned": metrics["structural_prune_filters_pruned"],
        "global_prune_budget": metrics["structural_global_prune_budget"],
        "global_prune_normalize": metrics["structural_global_prune_normalize"],
        "grow_new_only_steps": metrics["structural_grow_new_only_steps"],
    })
    structural_df.to_csv(os.path.join(output_dir, "structural_decisions.csv"), index=False)

    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(metrics['val_acc']):.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final conv channels: {final_conv_channels}")
    print(f"Structural decisions: {metrics['structural_actions']}")

    return metrics, model, num_epochs


def run_three_phase_baseline(
    experiment_name,
    model,
    conv_channels,
    train_loader,
    val_loader,
    test_loader,
    phase1_epochs,
    phase2_epochs,
    phase3_epochs,
    growth_layer_idx,
    growth_gamma,
    learning_rate=0.001,
    beta=0.1,
    lambda_penalty=0,
    output_dir=None,
    checkpoint_metric="val_loss_total",
    junctures_mode="three_phase",
):
    if output_dir is None:
        output_dir = os.path.join("./results", experiment_name)
    ensure_output_dir(output_dir)

    phase1_epochs = int(phase1_epochs)
    phase2_epochs = int(phase2_epochs)
    phase3_epochs = int(phase3_epochs)
    total_epochs = phase1_epochs + phase2_epochs + phase3_epochs

    conv_channels = _conv_channels_from_model(model)
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    loss_label = "Penalised ELBO" if lambda_penalty > 0 else "ELBO"

    print(f"\n{'-'*20} Running {experiment_name} CNN experiment {'-'*20}")
    print(f"Initial conv channels: {conv_channels}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    metrics = _initialise_standard_metrics()
    metrics["phase_history"] = []
    metrics["conv_channel_history"] = []

    best_checkpoint_score = float("inf")
    best_model_state = None
    best_conv_channels = list(conv_channels)
    best_epoch = 0
    last_epoch = 0
    old_width = None
    filters_to_add = 0
    phase = "phase1_static"

    for epoch in range(1, total_epochs + 1):
        if epoch == phase1_epochs + 1 and phase2_epochs > 0:
            print("-" * 20 + f" Three-phase growth boundary (epoch {epoch}) " + "-" * 20)
            conv_channels = _conv_channels_from_model(model)
            model, conv_channels, old_width, filters_to_add = build_grown_model_at_conv_layer(
                model, conv_channels, growth_layer_idx, growth_gamma
            )
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            print(
                f"Added {filters_to_add} filters to conv layer {growth_layer_idx} "
                f"({old_width} -> {conv_channels[growth_layer_idx]})."
            )
        if epoch == phase1_epochs + phase2_epochs + 1 and phase3_epochs > 0:
            print("-" * 20 + f" Three-phase prune boundary (epoch {epoch}) " + "-" * 20)
            if old_width is None:
                raise RuntimeError("cannot enter phase 3 before phase 2 growth has occurred")
            model, conv_channels = build_three_phase_pruned_model(
                model, conv_channels, growth_layer_idx, old_width
            )
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            print(
                f"Removed the {filters_to_add} appended filters from conv layer "
                f"{growth_layer_idx}; conv channels restored to {conv_channels}."
            )

        if epoch <= phase1_epochs:
            phase = "phase1_static"
        elif epoch <= phase1_epochs + phase2_epochs:
            phase = "phase2_grown"
        else:
            phase = "phase3_pruned"

        last_epoch = epoch
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
        metrics["phase_history"].append(phase)
        metrics["conv_channel_history"].append(list(conv_channels))

        print(
            f"Epoch {epoch} ({phase}): Train Loss({loss_label})={train_loss_total:.4f}, "
            f"Train Acc={train_acc:.2f}%, Params={count_params(model):,}, "
            f"Val Acc={val_acc:.2f}%"
        )

        if phase == "phase3_pruned":
            val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
            if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
                best_checkpoint_score = val_score
                best_epoch = epoch
                best_conv_channels = list(conv_channels)
                best_model_state = copy.deepcopy(model.state_dict())
                save_checkpoint(
                    os.path.join(output_dir, "best_checkpoint.pth"),
                    state_dict=best_model_state,
                    epoch=best_epoch,
                    conv_channels=best_conv_channels,
                    selection_metric=checkpoint_metric_label,
                    selection_value=best_checkpoint_score,
                )

    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, "metrics.png"),
    )

    if best_model_state is not None:
        eval_model = _make_eval_model(model, best_conv_channels)
        eval_model.load_state_dict(best_model_state, strict=True)
    else:
        eval_model = model

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f"Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}")

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else last_epoch,
        model=eval_model,
    )

    final_conv_channels = _conv_channels_from_model(eval_model)
    metrics.update({
        "test_acc": test_acc,
        "test_loss_total": test_loss_total,
        "test_loss_nll": test_loss_nll,
        "test_loss_kl": test_loss_kl,
        "test_brier": test_brier,
        "param_count": count_params(eval_model),
        "trainable_param_count": eval_model.get_param_stats()["trainable_params"],
        "conv_channels": final_conv_channels,
    })

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        conv_channels=metrics["conv_channels"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=lambda_penalty,
        selected_checkpoint_metric=checkpoint_metric_label,
        junctures_mode=junctures_mode,
    )

    return metrics, model, total_epochs


def main(
    save_path,
    conv_channels,
    lambda_penalty,
    dataset="fashion_mnist",
    junctures_mode="both",
    checkpoint_metric="val_loss_total",
    growth_layer_score="mad",
    growth_mad_percentile=90.0,
    decision_cooldown_epochs=0,
    phase1_epochs=10,
    phase2_epochs=10,
    phase3_epochs=10,
    three_phase_growth_layer_idx=1,
    three_phase_growth_gamma=None,
    resume_from_plasticity_dir=None,
    run_mode="plasticity",
    fc_hidden=128,
):
    """
    dataset options:
      - "fashion_mnist"
      - "cifar10"

    run_mode options:
      - "baseline"
      - "plasticity"
      - "three_phase"
      - "static_replay" (requires resume_from_plasticity_dir)
    """
    allowed_run_modes = {
        "baseline",
        "plasticity",
        "three_phase",
        "static_replay",
    }
    if run_mode not in allowed_run_modes:
        raise ValueError(
            f"run_mode must be one of {sorted(allowed_run_modes)}, got {run_mode!r}"
        )
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"dataset must be one of {sorted(DATASET_CONFIGS)}, got {dataset!r}"
        )
    in_channels = DATASET_CONFIGS[dataset]["in_channels"]
    num_classes = DATASET_CONFIGS[dataset]["num_classes"]

    num_epochs = 90
    batch_size = 64
    learning_rate = 0.01
    beta = 0.01
    decision_interval_min = 1
    decision_interval_max = 1
    decision_interval_power = 1
    decision_warmup_epochs = 0
    decision_cooldown_epochs = 0

    gamma = 0.0
    rho = 0.1
    if three_phase_growth_gamma is None:
        three_phase_growth_gamma = gamma
    prune_mode = "global_param"
    global_prune_budget = "filters"
    global_prune_normalize = "mad"
    # warm_start_steps = 16
    warm_start_steps = 32
    warm_start_lr = learning_rate * 0.4
    # grow_new_only_steps = 4
    grow_new_only_steps = 8

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

    conv_channels = list(conv_channels)
    conv_tag = _conv_channels_tag(conv_channels)

    if run_mode == "baseline":
        baseline_model = BayesianCNN(
            in_channels, conv_channels, num_classes, fc_hidden=fc_hidden
        ).to(device)
        baseline_output_dir = os.path.join(save_path, f"baseline_{conv_tag}")
        run_experiment(
            "baseline",
            baseline_model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            learning_rate,
            beta=beta,
            output_dir=baseline_output_dir,
            checkpoint_metric=checkpoint_metric,
        )

    if run_mode == "static_replay":
        if resume_from_plasticity_dir is None:
            raise ValueError(
                "run_mode='static_replay' requires resume_from_plasticity_dir to be set"
            )
        plasticity_dir, summary_path = _resolve_plasticity_dir_for_static_replay(
            resume_from_plasticity_dir
        )
        replay_conv_channels = _parse_conv_channels_from_summary(summary_path)
        static_output_dir = _static_replay_output_dir(plasticity_dir)
        ensure_output_dir(static_output_dir)

        source_meta = _parse_experiment_dir_name(os.path.basename(plasticity_dir))
        source_lambda = source_meta.get("lambda_penalty")
        if source_lambda is None:
            source_lambda = float(lambda_penalty)

        print("\n\n" + "=" * 50)
        print("Training Static Replay of Plasticity Architecture")
        print("=" * 50)
        print(f"Source plasticity dir: {plasticity_dir}")
        print(f"Replayed conv channels: {replay_conv_channels}")
        print(f"Output dir: {static_output_dir}")

        write_static_replay_provenance(
            static_output_dir,
            {
                "source_plasticity_dir": plasticity_dir,
                "source_summary_csv": summary_path,
                "replayed_conv_channels": replay_conv_channels,
                "source_lambda_penalty": source_lambda,
                "training_lambda_penalty": 0.0,
            },
        )

        static_model = BayesianCNN(
            in_channels, replay_conv_channels, num_classes, fc_hidden=fc_hidden
        ).to(device)
        run_experiment(
            "static_replay",
            static_model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            learning_rate,
            beta=beta,
            lambda_penalty=0,
            output_dir=static_output_dir,
            checkpoint_metric=checkpoint_metric,
            junctures_mode="static_replay",
            summary_lambda_penalty=source_lambda,
        )

    if run_mode == "three_phase":
        three_phase_model = BayesianCNN(
            in_channels, conv_channels, num_classes, fc_hidden=fc_hidden
        ).to(device)
        three_phase_output_dir = os.path.join(
            save_path, f"three_phase_baseline_{conv_tag}"
        )
        run_three_phase_baseline(
            "three_phase_baseline",
            three_phase_model,
            conv_channels,
            train_loader,
            val_loader,
            test_loader,
            phase1_epochs=phase1_epochs,
            phase2_epochs=phase2_epochs,
            phase3_epochs=phase3_epochs,
            growth_layer_idx=three_phase_growth_layer_idx,
            growth_gamma=three_phase_growth_gamma,
            learning_rate=learning_rate,
            beta=beta,
            lambda_penalty=0,
            output_dir=three_phase_output_dir,
            checkpoint_metric=checkpoint_metric,
        )

    if run_mode == "plasticity":
        suffix = _experiment_dir_suffix(junctures_mode)
        lam_tag = _format_lambda_dir(lambda_penalty)
        plasticity_output_dir = os.path.join(
            save_path, f"plasticity_{conv_tag}_{lam_tag}{suffix}"
        )
        base_model = BayesianCNN(
            in_channels, conv_channels, num_classes, fc_hidden=fc_hidden
        ).to(device)
        run_adaptive_experiment(
            "plasticity",
            base_model,
            conv_channels,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            learning_rate,
            beta,
            lambda_penalty=lambda_penalty,
            gamma=gamma,
            rho=rho,
            warm_start_steps=warm_start_steps,
            warm_start_lr=warm_start_lr,
            decision_interval=None,
            decision_interval_min=decision_interval_min,
            decision_interval_max=decision_interval_max,
            decision_interval_power=decision_interval_power,
            decision_warmup_epochs=decision_warmup_epochs,
            decision_cooldown_epochs=decision_cooldown_epochs,
            output_dir=plasticity_output_dir,
            uncertainty_combine="mean",
            growth_layer_score=growth_layer_score,
            growth_mad_percentile=growth_mad_percentile,
            junctures_mode=junctures_mode,
            prune_mode=prune_mode,
            global_prune_budget=global_prune_budget,
            global_prune_normalize=global_prune_normalize,
            checkpoint_metric=checkpoint_metric,
            grow_new_only_steps=grow_new_only_steps,
        )


def test_conv_surgery():
    print("Testing conv surgery...")
    model = BayesianCNN(3, [32, 64], 10, fc_hidden=128).to(device)
    grown_model, grown_channels, layer_idx, old_width = filtergenesis(
        model, [32, 64], gamma=0.1, growth_layer_score="mean"
    )
    expand_and_load_conv_stack(model.state_dict(), grown_model)
    assert grown_model.conv_layers[layer_idx].out_channels == grown_channels[layer_idx]
    if layer_idx + 1 < len(grown_model.conv_layers):
        assert (
            grown_model.conv_layers[layer_idx + 1].in_channels
            == grown_channels[layer_idx]
        )
    else:
        assert grown_model.fc.in_features == grown_channels[layer_idx]

    keep_dict, _ = filterapoptosis(
        grown_model,
        0.1,
        prune_mode="per_layer",
        min_filters_per_layer=2,
    )
    if keep_dict is not None:
        from lib.plasticity import build_pruned_model

        pruned_model, pruned_channels = build_pruned_model(grown_model, keep_dict)
        assert pruned_model.fc.in_features == pruned_channels[-1]
        x = torch.randn(2, 3, 32, 32, device=device)
        out = pruned_model(x)
        assert out.shape == (2, 10)
        print(f"Pruned channels: {pruned_channels}")
    print("Conv surgery test passed.")


if __name__ == "__main__":


    # for channels in [[100, 100],[75, 75],[50, 50], [25, 25]]:
    #     for lambda_penalty in [0]:
    #         for i in range(1, 6):
    #             set_seed(SEED + i)
    #             main(
    #                 save_path=f"./results_fashionmnist_cnn/run_{i}",
    #                 conv_channels=channels,
    #                 lambda_penalty=lambda_penalty,
    #                 run_mode="baseline",
    #                 dataset="cifar10",
    #                 fc_hidden=128,
    #             )

    for channels in [[300,300]]:
        for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
            for i in range(1, 6):
                set_seed(SEED + i)
                main(
                    save_path=f"./results_fashionmnist_cnn_ws32/run_{i}",
                    conv_channels=channels,
                    lambda_penalty=lambda_penalty,
                    run_mode="plasticity",
                    dataset="cifar10",
                    fc_hidden=128,
                )

    for channels in [[300,300]]:
        for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
            for i in range(1, 6):
                set_seed(SEED + i)
                main(
                    save_path=f"./results_fashionmnist_cnn_ws32/run_{i}",
                    conv_channels=channels,
                    lambda_penalty=lambda_penalty,
                    run_mode="plasticity",
                    dataset="cifar10",
                    fc_hidden=128,
                    junctures_mode="prune"
                )



