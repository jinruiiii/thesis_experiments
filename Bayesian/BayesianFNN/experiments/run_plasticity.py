import ast
import copy
import json
import math
import os

import pandas as pd
import torch
import torch.optim as optim

from lib.data import build_dataloaders
from lib.flops import dense_fnn_flops
from lib.plot import _parse_experiment_dir_name, plot_metrics
from lib.plasticity import (
    _hidden_sizes_from_model,
    build_grown_model_at_layer,
    build_three_phase_pruned_model,
    structural_decision_juncture,
)
from lib.seed import SEED, device
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
    write_hybrid_provenance,
    write_metrics_csv,
)
from models.bayesian_fnn import BayesianFNN

def run_three_phase_baseline(
    experiment_name,
    model,
    hidden_sizes,
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
        output_dir = os.path.join('./results', experiment_name)
    ensure_output_dir(output_dir)

    phase1_epochs = int(phase1_epochs)
    phase2_epochs = int(phase2_epochs)
    phase3_epochs = int(phase3_epochs)
    if phase1_epochs < 0:
        raise ValueError("phase1_epochs must be non-negative")
    if phase2_epochs <= 0 or phase3_epochs <= 0:
        raise ValueError("phase2_epochs and phase3_epochs must be positive")
    total_epochs = phase1_epochs + phase2_epochs + phase3_epochs
    if total_epochs <= 0:
        raise ValueError("at least one phase must have a positive epoch count")

    hidden_sizes = _hidden_sizes_from_model(model)
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    loss_label = 'Penalised ELBO' if lambda_penalty > 0 else 'ELBO'

    print(f"\n{'-'*20} Running {experiment_name} experiment {'-'*20}")
    if phase1_epochs == 0:
        print(
            "Grow-prune refinement: "
            f"phase2={phase2_epochs}, phase3={phase3_epochs}, "
            f"growth_layer_idx={growth_layer_idx}, growth_gamma={growth_gamma}"
        )
    else:
        print(
            "Three-phase baseline: "
            f"phase1={phase1_epochs}, phase2={phase2_epochs}, phase3={phase3_epochs}, "
            f"growth_layer_idx={growth_layer_idx}, growth_gamma={growth_gamma}"
        )
    print(f"Initial hidden sizes: {hidden_sizes}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    metrics = _initialise_standard_metrics()
    metrics['phase_history'] = []
    metrics['hidden_size_history'] = []

    best_checkpoint_score = float("inf")
    best_model_state = None
    best_hidden_sizes = list(hidden_sizes)
    best_epoch = 0
    last_epoch = 0
    old_width = None
    neurons_to_add = 0
    phase = "phase1_static"

    for epoch in range(1, total_epochs + 1):
        if epoch == phase1_epochs + 1 and phase2_epochs > 0:
            print("-" * 20 + f" Three-phase growth boundary (epoch {epoch}) " + "-" * 20)
            hidden_sizes = _hidden_sizes_from_model(model)
            model, hidden_sizes, old_width, neurons_to_add = build_grown_model_at_layer(
                model,
                hidden_sizes,
                growth_layer_idx,
                growth_gamma,
            )
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            print(
                f"Added {neurons_to_add} neurons to hidden layer {growth_layer_idx} "
                f"using base width {old_width} "
                f"({old_width} -> {hidden_sizes[growth_layer_idx]})."
            )
        if epoch == phase1_epochs + phase2_epochs + 1 and phase3_epochs > 0:
            print("-" * 20 + f" Three-phase prune boundary (epoch {epoch}) " + "-" * 20)
            if old_width is None:
                raise RuntimeError("cannot enter phase 3 before phase 2 growth has occurred")
            model, hidden_sizes = build_three_phase_pruned_model(
                model,
                hidden_sizes,
                growth_layer_idx,
                old_width,
            )
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            print(
                f"Removed the {neurons_to_add} appended neurons from hidden layer "
                f"{growth_layer_idx}; hidden sizes restored to {hidden_sizes}."
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
        metrics['train_loss_total'].append(train_loss_total)
        metrics['train_loss_nll'].append(train_loss_nll)
        metrics['train_loss_kl'].append(train_loss_kl)
        metrics['train_loss_penalty'].append(train_loss_penalty)
        metrics['train_acc'].append(train_acc)
        metrics['train_brier'].append(train_brier)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics['val_loss_total'].append(val_loss_total)
        metrics['val_loss_nll'].append(val_loss_nll)
        metrics['val_loss_kl'].append(val_loss_kl)
        metrics['val_loss_penalty'].append(val_loss_penalty)
        metrics['val_acc'].append(val_acc)
        metrics['val_brier'].append(val_brier)
        metrics['param_count_history'].append(count_params(model))
        metrics['phase_history'].append(phase)
        metrics['hidden_size_history'].append(list(hidden_sizes))

        penalty_str = f', Penalty={train_loss_penalty:.4f}' if lambda_penalty > 0 else ''
        print(
            f'Epoch {epoch} ({phase}): Train Loss({loss_label})={train_loss_total:.4f}, '
            f'Train Loss(NLL)={train_loss_nll:.4f}, Train Loss(KL)={train_loss_kl:.4f}'
            f'{penalty_str}, Params={count_params(model):,}, Train Acc={train_acc:.2f}%, '
            f'Val Loss({loss_label})={val_loss_total:.4f}, Val Acc={val_acc:.2f}%'
        )
        if phase == "phase3_pruned":
            val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
            if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
                best_checkpoint_score = val_score
                best_epoch = epoch
                best_hidden_sizes = list(hidden_sizes)
                best_model_state = copy.deepcopy(model.state_dict())
                save_checkpoint(
                    os.path.join(output_dir, "best_checkpoint.pth"),
                    state_dict=best_model_state,
                    epoch=best_epoch,
                    hidden_sizes=best_hidden_sizes,
                    selection_metric=checkpoint_metric_label,
                    selection_value=best_checkpoint_score,
                )

    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, 'metrics.png'),
    )

    if best_model_state is not None:
        eval_model = BayesianFNN(
            model.in_features,
            best_hidden_sizes,
            model.out_features,
        ).to(device)
        eval_model.load_state_dict(best_model_state, strict=True)
        print(
            f"Loaded best model from epoch {best_epoch} "
            f"(hidden_sizes={best_hidden_sizes}) for final testing."
        )
    else:
        eval_model = model

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f'Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}')

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else last_epoch,
        hidden_sizes=[layer.mu_w.shape[0] for layer in eval_model.layers],
    )

    final_hidden_sizes = [layer.mu_w.shape[0] for layer in eval_model.layers]
    metrics.update({
        'test_acc': test_acc,
        'test_loss_total': test_loss_total,
        'test_loss_nll': test_loss_nll,
        'test_loss_kl': test_loss_kl,
        'test_brier': test_brier,
        'param_count': count_params(eval_model),
        'trainable_param_count': eval_model.get_param_stats()['trainable_params'],
        'hidden_sizes': final_hidden_sizes,
    })

    # Restrict best-validation reporting to phase 3 epochs, matching checkpoint selection.
    phase3_val_acc = [
        acc for acc, ph in zip(metrics['val_acc'], metrics['phase_history'])
        if ph == "phase3_pruned"
    ]
    phase3_val_brier = [
        brier for brier, ph in zip(metrics['val_brier'], metrics['phase_history'])
        if ph == "phase3_pruned"
    ]

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        hidden_sizes=metrics["hidden_sizes"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(phase3_val_acc),
        best_val_brier=min(phase3_val_brier),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=lambda_penalty,
        flops=dense_fnn_flops(784, metrics["hidden_sizes"], 10),
        selected_checkpoint_metric=checkpoint_metric_label,
        junctures_mode=junctures_mode,
    )

    phase_df = pd.DataFrame({
        'epoch': range(1, total_epochs + 1),
        'phase': metrics['phase_history'],
        'hidden_sizes': [str(hs) for hs in metrics['hidden_size_history']],
        'param_count': metrics['param_count_history'],
    })
    phase_df.to_csv(os.path.join(output_dir, 'phase_history.csv'), index=False)

    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy (phase 3): {max(phase3_val_acc):.2f}%")
    print(f"Best checkpoint score (phase 3): {best_checkpoint_score:.4f} at epoch {best_epoch}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Selected hidden sizes: {final_hidden_sizes}")
    print(f"Phase history written to {os.path.join(output_dir, 'phase_history.csv')}")

    return metrics, model, total_epochs


def _experiment_dir_suffix(junctures_mode):
    return "" if junctures_mode == "both" else f"_{junctures_mode}_only"


def _format_lambda_dir(lambda_penalty):
    return f"{float(lambda_penalty):g}"


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


def _parse_hidden_sizes_from_summary(summary_path):
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"Empty experiment summary: {summary_path}")
    if "Hidden Sizes" not in df.columns:
        raise ValueError(f"'Hidden Sizes' column missing in {summary_path}")
    raw = df.iloc[0]["Hidden Sizes"]
    if pd.isna(raw):
        raise ValueError(f"'Hidden Sizes' is empty in {summary_path}")
    sizes = ast.literal_eval(str(raw))
    if not isinstance(sizes, (list, tuple)) or not sizes:
        raise ValueError(f"Invalid Hidden Sizes value {raw!r} in {summary_path}")
    sizes = [int(s) for s in sizes]
    if any(s <= 0 for s in sizes):
        raise ValueError(f"Hidden Sizes must be positive ints, got {sizes} from {summary_path}")
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


def run_experiment(experiment_name, model, train_loader, val_loader, test_loader, num_epochs,
                   learning_rate=0.001, start_epoch=1, metrics=None, beta=0.1,
                   lambda_penalty=0, output_dir=None, checkpoint_metric="val_loss_total",
                   junctures_mode="both", summary_lambda_penalty=None):
    if output_dir is None:
        output_dir = os.path.join('./results', experiment_name)
    ensure_output_dir(output_dir)

    print(f"\n{'-'*20} Running {experiment_name} experiment {'-'*20}")
    
    # Display model parameters
    param_stats = model.get_param_stats() if hasattr(model, 'get_param_stats') else {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    print(f"Model parameters: {param_stats['total_params']:,}")
    print(f"Trainable parameters: {param_stats.get('trainable_params', param_stats['total_params']):,}")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Track metrics
    if not metrics:
        metrics = {}
        metrics['train_loss_total'] = []
        metrics['train_loss_nll'] = []
        metrics['train_loss_kl'] = []
        metrics['train_acc'] = []
        metrics['train_brier'] = []
        metrics['val_loss_total'] = []
        metrics['val_loss_nll'] = []
        metrics['val_loss_kl'] = []
        metrics['val_acc'] = []
        metrics['val_brier'] = []
        metrics['train_loss_penalty'] = []
        metrics['val_loss_penalty'] = []
        metrics['param_count_history'] = []

    loss_label = 'Penalised ELBO' if lambda_penalty > 0 else 'ELBO'
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
    
    # Training loop
    last_epoch = best_epoch = start_epoch - 1
    for epoch in range(start_epoch, start_epoch + num_epochs):
        last_epoch = epoch
        
        # Train
        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty = train(
            model, train_loader, optimizer, epoch, device, beta_scaled, lambda_penalty
        )
        metrics['train_loss_total'].append(train_loss_total)
        metrics['train_loss_nll'].append(train_loss_nll)
        metrics['train_loss_kl'].append(train_loss_kl)
        metrics['train_acc'].append(train_acc)
        metrics['train_brier'].append(train_brier)
        metrics['train_loss_penalty'].append(train_loss_penalty)

        # Validate
        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics['val_loss_total'].append(val_loss_total)
        metrics['val_loss_nll'].append(val_loss_nll)
        metrics['val_loss_kl'].append(val_loss_kl)
        metrics['val_acc'].append(val_acc)
        metrics['val_brier'].append(val_brier)
        metrics['val_loss_penalty'].append(val_loss_penalty)
        metrics['param_count_history'].append(count_params(model))

        penalty_str = f', Penalty={train_loss_penalty:.4f}' if lambda_penalty > 0 else ''
        print(
            f'Epoch {epoch}: Train Loss({loss_label})={train_loss_total:.4f}, Train Loss(NLL)={train_loss_nll:.4f}, '
            f'Train Loss(KL)={train_loss_kl:.4f}{penalty_str}, Train Acc={train_acc:.2f}%, Train Brier={train_brier:.3f}, '
            f'Val Loss({loss_label})={val_loss_total:.4f}, Val Loss(NLL)={val_loss_nll:.4f}, '
            f'Val Loss(KL)={val_loss_kl:.4f}, Val Acc={val_acc:.2f}%, Val Brier={val_brier:.3f}'
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
                hidden_sizes=[layer.mu_w.shape[0] for layer in model.layers],
                selection_metric=checkpoint_metric_label,
                selection_value=best_checkpoint_score,
            )
        
    # Plot and save metrics
    plot_metrics(
        {experiment_name: {
            'train_loss_total': metrics['train_loss_total'],
            'train_loss_nll': metrics['train_loss_nll'],
            'train_loss_kl': metrics['train_loss_kl'],
            'train_acc': metrics['train_acc'],
            'train_brier': metrics['train_brier'],
            'val_loss_total': metrics['val_loss_total'],
            'val_loss_nll': metrics['val_loss_nll'],
            'val_loss_kl': metrics['val_loss_kl'],
            'val_acc': metrics['val_acc'],
            'val_brier': metrics['val_brier']
        }}, 
        save_path=os.path.join(output_dir, 'metrics.png')
    )

    # Load best model for test
    best_model_for_eval = None
    if best_model_state is not None:
        best_model_for_eval = copy.deepcopy(model)
        best_model_for_eval.load_state_dict(best_model_state)
        print(f"Loaded best model from Epoch {best_epoch} based on validation loss for final testing.")

    eval_model = best_model_for_eval if best_model_for_eval is not None else model
    
    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f'Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}')
    
    # Save best checkpoint (same weights used for test eval)
    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else last_epoch,
        hidden_sizes=[layer.mu_w.shape[0] for layer in eval_model.layers],
    )

    # get final architecture
    hidden_sizes = []
    for i, layer in enumerate(eval_model.layers):
        hidden_sizes.append(layer.mu_w.shape[0])
    
    # Update metrics
    metrics.update({
        'test_acc': test_acc,
        'test_loss_total': test_loss_total,
        'test_loss_nll': test_loss_nll,
        'test_loss_kl': test_loss_kl,
        'test_brier': test_brier,
        'param_count': count_params(eval_model),
        'trainable_param_count': eval_model.get_param_stats()['trainable_params'],
        'hidden_sizes': hidden_sizes
    })
    
    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        hidden_sizes=metrics["hidden_sizes"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=csv_lambda_penalty,
        flops=dense_fnn_flops(784, metrics["hidden_sizes"], 10),
        selected_checkpoint_metric=checkpoint_metric_label,
        junctures_mode=junctures_mode,
    )
    
    # Print summary
    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(metrics['val_acc'][start_epoch-1:]):.2f}%")
    print(f"Best validation loss: {min(metrics['val_loss_total'][start_epoch-1:]):.4f}")
    print(f"Best validation loss (NLL): {min(metrics['val_loss_nll'][start_epoch-1:]):.4f}")
    print(f"Best validation loss (KL): {min(metrics['val_loss_kl'][start_epoch-1:]):.4f}")
    print(f"Best validation brier: {min(metrics['val_brier'][start_epoch-1:]):.3f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final test loss: {test_loss_total:.4f}")
    print(f"Final test loss (NLL): {test_loss_nll:.4f}")
    print(f"Final test loss (KL): {test_loss_kl:.4f}")
    print(f"Final test brier: {test_brier:.3f}")
    
    epochs_run = last_epoch - start_epoch + 1
    return metrics, model, epochs_run


def _last_structural_epoch(num_epochs, decision_cooldown_epochs=0):
    """Last epoch at which a structural juncture may occur."""
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


def run_adaptive_experiment(
    experiment_name,
    model,
    hidden_sizes,
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
    """
    Dynamic structural adaptation via penalised ELBO.
    Structural decisions happen at annealed intervals (or fixed if decision_interval is set).
    No structural junctures occur for the first decision_warmup_epochs epochs or the
    last decision_cooldown_epochs epochs (weight-only convergence at the end).
    junctures_mode: "both" (grow or prune), "grow" (grow only), "prune" (prune only),
    or "both_gp" (grow, prune, or grow+prune combined).
    global_prune_budget: "params" or "neurons" (only used when prune_mode="global_param").
    global_prune_normalize: "percentile", "zscore", "mad", or "raw" (only used when prune_mode="global_param").
    growth_layer_score: "mean" or "mad" for which layer to expand on grow.
    growth_mad_percentile: percentile of within-layer MAD z-scores when growth_layer_score="mad".
    checkpoint_metric: "val_loss_nll" or "val_loss_total" for best-checkpoint selection.
    grow_new_only_steps: first N grow warm-start steps use new-only mask, rest update all
    (default None = all warm_start_steps are new-only).
    """
    if output_dir is None:
        output_dir = os.path.join('./results', experiment_name)
    ensure_output_dir(output_dir)

    print("\n" + "=" * 50)
    print(f"Training {experiment_name.upper()} (Penalised ELBO, lambda={lambda_penalty}, junctures_mode={junctures_mode})")
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
    if decision_warmup_epochs > 0:
        print(f"Structural juncture warmup: {decision_warmup_epochs} epoch(s) of weight-only training")
    if decision_cooldown_epochs > 0:
        print(
            f"Structural juncture cooldown: {decision_cooldown_epochs} epoch(s) of "
            "weight-only convergence at the end"
        )
    print("=" * 50)

    param_stats = model.get_param_stats()
    print(f"Model parameters: {param_stats['total_params']:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    metrics = {
        'train_loss_total': [],
        'train_loss_nll': [],
        'train_loss_kl': [],
        'train_loss_penalty': [],
        'train_acc': [],
        'train_brier': [],
        'val_loss_total': [],
        'val_loss_nll': [],
        'val_loss_kl': [],
        'val_loss_penalty': [],
        'val_acc': [],
        'val_brier': [],
        'structural_epochs': [],
        'structural_actions': [],
        'structural_delta_grow': [],
        'structural_delta_prune': [],
        'structural_delta_grow_prune': [],
        'structural_L_before': [],
        'structural_L_after_none': [],
        'structural_hidden_sizes': [],
        'structural_prune_mode': [],
        'structural_prune_target_remove': [],
        'structural_prune_params_removed': [],
        'structural_prune_neurons_pruned': [],
        'structural_global_prune_budget': [],
        'structural_global_prune_normalize': [],
        'structural_grow_new_only_steps': [],
        'param_count_history': [],
        'junctures_mode': junctures_mode,
    }

    growth_cooldown = {}

    best_checkpoint_score = float('inf')
    best_model_state = None
    best_epoch = 0
    best_hidden_sizes = list(hidden_sizes)
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    decision_warmup_epochs = max(0, int(decision_warmup_epochs))
    decision_cooldown_epochs = max(0, int(decision_cooldown_epochs))
    last_structural_epoch = _last_structural_epoch(num_epochs, decision_cooldown_epochs)
    if decision_warmup_epochs + decision_cooldown_epochs >= num_epochs:
        print(
            "Warning: decision_warmup_epochs + decision_cooldown_epochs >= num_epochs; "
            "no structural junctures will occur."
        )

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
        metrics['train_loss_total'].append(train_loss_total)
        metrics['train_loss_nll'].append(train_loss_nll)
        metrics['train_loss_kl'].append(train_loss_kl)
        metrics['train_loss_penalty'].append(train_loss_penalty)
        metrics['train_acc'].append(train_acc)
        metrics['train_brier'].append(train_brier)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics['val_loss_total'].append(val_loss_total)
        metrics['val_loss_nll'].append(val_loss_nll)
        metrics['val_loss_kl'].append(val_loss_kl)
        metrics['val_loss_penalty'].append(val_loss_penalty)
        metrics['val_acc'].append(val_acc)
        metrics['val_brier'].append(val_brier)
        metrics['param_count_history'].append(count_params(model))

        print(
            f'Epoch {epoch}: Train Loss(Penalised ELBO)={train_loss_total:.4f}, Train Loss(NLL)={train_loss_nll:.4f}, '
            f'Train Loss(KL)={train_loss_kl:.4f}, Penalty={train_loss_penalty:.4f}, Params={count_params(model):,}, '
            f'Train Acc={train_acc:.2f}%, Val Loss(Penalised ELBO)={val_loss_total:.4f}, Val Acc={val_acc:.2f}%'
        )

        val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
        if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
            best_checkpoint_score = val_score
            best_epoch = epoch
            best_hidden_sizes = list(hidden_sizes)
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                hidden_sizes=best_hidden_sizes,
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
                grow_new_only_steps=grow_new_only_steps,
            )
            metrics['structural_epochs'].append(epoch)
            metrics['structural_actions'].append(action)
            metrics['structural_delta_grow'].append(info.get('delta_grow'))
            metrics['structural_delta_prune'].append(info.get('delta_prune'))
            metrics['structural_delta_grow_prune'].append(info.get('delta_grow_prune'))
            metrics['structural_L_before'].append(info.get('L_before'))
            metrics['structural_L_after_none'].append(info.get('L_after_none'))
            metrics['structural_hidden_sizes'].append(list(hidden_sizes))
            metrics['structural_prune_mode'].append(info.get('prune_mode'))
            metrics['structural_prune_target_remove'].append(info.get('prune_target_remove'))
            metrics['structural_prune_params_removed'].append(info.get('prune_params_removed'))
            metrics['structural_prune_neurons_pruned'].append(info.get('prune_neurons_pruned'))
            metrics['structural_global_prune_budget'].append(info.get('global_prune_budget'))
            metrics['structural_global_prune_normalize'].append(
                info.get('global_prune_normalize')
            )
            metrics['structural_grow_new_only_steps'].append(
                info.get('grow_new_only_steps')
            )

            if action != 'none':
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

            # Schedule next decision juncture.
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

    # Final test evaluation with best checkpoint
    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, 'metrics.png'),
    )

    if best_model_state is not None:
        eval_model = BayesianFNN(
            model.in_features,
            best_hidden_sizes,
            model.out_features,
        ).to(device)
        eval_model.load_state_dict(best_model_state, strict=True)
        print(
            f"Loaded best model from epoch {best_epoch} "
            f"(hidden_sizes={best_hidden_sizes}) for final testing."
        )
    else:
        eval_model = model

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f'Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}')

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else num_epochs,
        hidden_sizes=[layer.mu_w.shape[0] for layer in eval_model.layers],
    )

    final_hidden_sizes = [layer.mu_w.shape[0] for layer in eval_model.layers]
    metrics.update({
        'test_acc': test_acc,
        'test_loss_total': test_loss_total,
        'test_loss_nll': test_loss_nll,
        'test_loss_kl': test_loss_kl,
        'test_brier': test_brier,
        'param_count': count_params(eval_model),
        'trainable_param_count': eval_model.get_param_stats()['trainable_params'],
        'hidden_sizes': final_hidden_sizes,
    })

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        hidden_sizes=metrics["hidden_sizes"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=lambda_penalty,
        flops=dense_fnn_flops(784, metrics["hidden_sizes"], 10),
        selected_checkpoint_metric=checkpoint_metric_label,
        junctures_mode=junctures_mode,
    )


    structural_df = pd.DataFrame({
        'epoch': metrics['structural_epochs'],
        'action': metrics['structural_actions'],
        'delta_grow': metrics['structural_delta_grow'],
        'delta_prune': metrics['structural_delta_prune'],
        'delta_grow_prune': metrics['structural_delta_grow_prune'],
        'L_before': metrics['structural_L_before'],
        'L_after_none': metrics['structural_L_after_none'],
        'hidden_sizes': [str(hs) for hs in metrics['structural_hidden_sizes']],
        'junctures_mode': metrics['junctures_mode'],
        'prune_mode': metrics['structural_prune_mode'],
        'prune_target_remove': metrics['structural_prune_target_remove'],
        'prune_params_removed': metrics['structural_prune_params_removed'],
        'prune_neurons_pruned': metrics['structural_prune_neurons_pruned'],
        'global_prune_budget': metrics['structural_global_prune_budget'],
        'global_prune_normalize': metrics['structural_global_prune_normalize'],
        'grow_new_only_steps': metrics['structural_grow_new_only_steps'],
    })
    structural_df.to_csv(os.path.join(output_dir, 'structural_decisions.csv'), index=False)

    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(metrics['val_acc']):.2f}%")
    print(f"Best validation total: {min(metrics['val_loss_total']):.4f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final hidden sizes: {final_hidden_sizes}")
    print(f"Structural decisions: {metrics['structural_actions']}")

    return metrics, model, num_epochs

def main(
    save_path,
    hidden_sizes,
    lambda_penalty,
    junctures_mode="both",
    checkpoint_metric="val_loss_total",
    growth_layer_score="mad",
    growth_mad_percentile=90.0,
    decision_cooldown_epochs=50,
    phase1_epochs=10,
    phase2_epochs=10,
    phase3_epochs=10,
    three_phase_growth_layer_idx=1,
    three_phase_growth_gamma=1,
    resume_from_plasticity_dir=None,
    dataset="fashion_mnist",
    run_mode="plasticity",
):
    """

    run_mode options:
      - "baseline": run only fixed-width baseline
      - "plasticity": run only plasticity (no baseline, no hybrid refinement)
      - "three_phase": run only three-phase baseline
      - "static_replay": train a static FNN using Hidden Sizes from an existing
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
    # Hyperparameters
    num_epochs = 60
    batch_size = 256
    learning_rate = 0.005 if dataset == "fashion_mnist" else 0.001
    beta = 0.002
    decision_interval_min = 1
    decision_interval_max = 1
    decision_interval_power = 1
    decision_warmup_epochs = 0
    decision_cooldown_epochs = 0

    gamma = 0.0
    rho = 0.1
    if three_phase_growth_gamma is None:
        three_phase_growth_gamma = gamma
    # gamma = 0.10
    # rho = 0.10
    prune_mode = "global_param"
    global_prune_budget = "neurons"  # "params" or "neurons"
    global_prune_normalize = "mad"  # "percentile", "zscore", "mad", or "raw"

    warm_start_steps = 64
    warm_start_lr = learning_rate * 0.4 if dataset == "fashion_mnist" else learning_rate
    grow_new_only_steps = 16 if dataset == "fashion_mnist" else 32
    
    # Create results directory
    os.makedirs(f'{save_path}', exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, batch_size=batch_size, seed=SEED
    )
    print(
        f"Dataset: {dataset} "
        f"(train={len(train_loader.dataset):,}, "
        f"val={len(val_loader.dataset):,}, "
        f"test={len(test_loader.dataset):,})"
    )

    if run_mode in ("baseline"):
        print("\n\n" + "="*50)
        print("Training Baseline Model")
        print("="*50)
        baseline_model = BayesianFNN(784, hidden_sizes, 10).to(device)
        initial_state_dict = copy.deepcopy(baseline_model.state_dict())
        baseline_output_dir = os.path.join(save_path, f'baseline_{hidden_sizes[0]}')
        baseline_metrics, _, _= run_experiment(
            'baseline', 
            baseline_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs, 
            learning_rate,
            start_epoch=1,
            beta=beta,
            output_dir=baseline_output_dir,
        )


    if run_mode in ("static_replay",):
        if resume_from_plasticity_dir is None:
            raise ValueError(
                "run_mode='static_replay' requires resume_from_plasticity_dir to be set"
            )
        plasticity_dir, summary_path = _resolve_plasticity_dir_for_static_replay(
            resume_from_plasticity_dir
        )
        replay_hidden_sizes = _parse_hidden_sizes_from_summary(summary_path)
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
        print(f"Replayed hidden sizes: {replay_hidden_sizes}")
        print(f"Output dir: {static_output_dir}")

        write_static_replay_provenance(
            static_output_dir,
            {
                "source_plasticity_dir": plasticity_dir,
                "source_summary_csv": summary_path,
                "replayed_hidden_sizes": replay_hidden_sizes,
                "source_lambda_penalty": source_lambda,
                "training_lambda_penalty": 0.0,
            },
        )

        static_model = BayesianFNN(784, replay_hidden_sizes, 10).to(device)
        run_experiment(
            "static_replay",
            static_model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            learning_rate,
            start_epoch=1,
            beta=beta,
            lambda_penalty=0,
            output_dir=static_output_dir,
            checkpoint_metric=checkpoint_metric,
            junctures_mode="static_replay",
            summary_lambda_penalty=source_lambda,
        )


    if run_mode in ("three_phase"):
        three_phase_model = BayesianFNN(784, hidden_sizes, 10).to(device)
        three_phase_output_dir = os.path.join(
            save_path,
            f'three_phase_baseline_{hidden_sizes[0]}',
        )
        run_three_phase_baseline(
            "three_phase_baseline",
            three_phase_model,
            hidden_sizes,
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

    if run_mode in ("plasticity"):
        suffix = _experiment_dir_suffix(junctures_mode)
        lam_tag = _format_lambda_dir(lambda_penalty)
        plasticity_output_dir = os.path.join(
            save_path, f"plasticity_{hidden_sizes[0]}_{lam_tag}{suffix}"
        )
        base_model = BayesianFNN(784, hidden_sizes, 10).to(device)
        run_adaptive_experiment(
            "plasticity",
            base_model,
            hidden_sizes,
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

if __name__ == "__main__":
    hidden_sizes = [500, 500]
    for hidden_size in [[20,20],[50,50],[100,100],[200,200],[500,500]]:
        for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
            for junctures_mode in ["both"]:
                for i in range(1, 6):
                    set_seed(SEED+i)
                    print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
                    main(
                        f"results_fashionmnist/run_{i}",
                        hidden_sizes,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="fashion_mnist",
                        run_mode="plasticity",
                    )


    hidden_sizes = [500, 500]
    for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
        for junctures_mode in ["prune"]:
            for i in range(1, 6):
                set_seed(SEED+i)
                print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
                main(
                    f"results_fashionmnist/run_{i}",
                    hidden_sizes,
                    lambda_penalty,
                    junctures_mode=junctures_mode,
                    dataset="fashion_mnist",
                    run_mode="plasticity",
                )


    for hidden_size in [[20,20],[50,50],[100,100],[150,150],[200,200]]:
        for lambda_penalty in [0]:
            for junctures_mode in ["both"]:
                for i in range(1,6):
                    set_seed(SEED+i)
                    print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
                    main(
                        f"results_fashionmnist/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="fashion_mnist",
                        run_mode="baseline",
                    )
    print("All experiments completed.")



    for hidden_size in [[200,200],[150,150],[100,100],[50,50],[20,20]]:
        for lambda_penalty in [0]:
            for junctures_mode in ["both"]:
                for i in range(1,6):
                    set_seed(SEED+i)
                    print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
                    main(
                        f"results_fashionmnist/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="fashion_mnist",
                        run_mode="three_phase",
                        three_phase_growth_layer_idx=1,
                        three_phase_growth_gamma=1,
                    )
    print("All experiments completed.")

    for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
        for i in range(1, 6):
            set_seed(SEED+i)
            plasticity_dir = (
                f"results_fashionmnist/run_{i}/plasticity_500_{_format_lambda_dir(lambda_penalty)}"
            )
            print(
                "Running static replay for run",
                i,
                f"(source={plasticity_dir})",
            )
            main(
                f"results_fashionmnist/run_{i}",
                [500, 500],  # unused for architecture; taken from plasticity summary
                lambda_penalty=0,
                dataset="fashion_mnist",
                run_mode="static_replay",
                resume_from_plasticity_dir=plasticity_dir,
            )
    print("All static replay experiments completed.")

