import ast
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

def plot_metrics(metrics_dict, save_path='./results/metrics_comparison.png'):
    """Plot comparison of metrics across all models"""
    # Define colors for each model
    colors = {
        'baseline': 'blue',
        'plasticity': 'red',
    }
    
    # Create figure with subplots
    fig, axs = plt.subplots(4, 3, figsize=(20, 15))
    
    # Training loss total
    ax = axs[0, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['train_loss_total']))
        ax.plot(epochs, metrics['train_loss_total'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Training Loss (nll + kl)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Training loss nll
    ax = axs[0, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['train_loss_nll']))
        ax.plot(epochs, metrics['train_loss_nll'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Training Loss (nll)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Training loss kl
    ax = axs[0, 2]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['train_loss_kl']))
        ax.plot(epochs, metrics['train_loss_kl'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Training Loss (kl)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Training Accuracy
    ax = axs[1, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['train_acc']))
        ax.plot(epochs, metrics['train_acc'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()

    # Training Brier
    ax = axs[1, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['train_brier']))
        ax.plot(epochs, metrics['train_brier'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Training Brier')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Brier')
    ax.legend()
    
    # Validation loss total
    ax = axs[2, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['val_loss_total']))
        ax.plot(epochs, metrics['val_loss_total'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Validation Loss (nll + kl)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Validation loss nll
    ax = axs[2, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['val_loss_nll']))
        ax.plot(epochs, metrics['val_loss_nll'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Validation Loss (nll)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Validation loss kl
    ax = axs[2, 2]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['val_loss_kl']))
        ax.plot(epochs, metrics['val_loss_kl'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Validation Loss (kl)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # Validation accuracy
    ax = axs[3, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['val_acc']))
        ax.plot(epochs, metrics['val_acc'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()

    # Validation brier
    ax = axs[3, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics['val_brier']))
        ax.plot(epochs, metrics['val_brier'], color=colors.get(model_name, 'gray'), label=model_name)
    ax.set_title('Validation Brier')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Brier')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_statistics(
    save_path="results",
    experiments=["plasticity"],
    num_runs=None,
):
    """
    Aggregate per-run experiment summaries into mean/std across runs.

    This function reads the per-run `experiment_summary.csv` files written by
    `write_experiment_summary_csv(...)` and aggregates across runs.

    Expected layout (as produced by `__main__` at the bottom of this file):
      - {save_path}/run_{i}/{experiment}/experiment_summary.csv

    Example:
      - results/run_1/plasticity/experiment_summary.csv

    If `num_runs` is provided, only `run_1..run_{num_runs}` are considered.
    Otherwise all `run_*` directories found under `save_path` are used.

    Note: `params_def` is kept for backwards compatibility but is ignored when
    reading from `experiment_summary.csv` (which already stores a single
    parameter count per run).
    """
    save_path = os.path.normpath(str(save_path))

    # Discover run indices from folders named run_<int>
    run_indices = []
    if os.path.isdir(save_path):
        for name in os.listdir(save_path):
            if not name.startswith("run_"):
                continue
            suffix = name[len("run_") :]
            if suffix.isdigit():
                run_indices.append(int(suffix))
    run_indices = sorted(set(run_indices))
    if num_runs is not None:
        run_indices = [i for i in run_indices if 1 <= i <= int(num_runs)]

    if not run_indices:
        raise FileNotFoundError(
            f"No run folders found under '{save_path}'. Expected folders like '{save_path}/run_1/'."
        )

    results = {}
    for experiment in experiments:
        rows = []
        for run_i in run_indices:
            summary_path = os.path.join(
                save_path, f"run_{run_i}", str(experiment), "experiment_summary.csv"
            )
            if not os.path.exists(summary_path):
                raise FileNotFoundError(
                    f"Missing experiment summary for run {run_i}: '{summary_path}'"
                )
            df = pd.read_csv(summary_path)
            if df.empty:
                raise ValueError(f"Empty experiment summary: {summary_path}")

            row0 = df.iloc[0]
            params = int(float(row0["Parameters"]))
            best_val_acc = float(row0["Best Val Acc"])
            best_val_brier = float(row0["Best Val Brier"])
            test_acc = float(row0["Test Acc"])
            test_brier = float(row0["Test Brier"])

            rows.append(
                {
                    "run": run_i,
                    "Experiment": str(experiment),
                    "Parameters": params,
                    "Best Val Acc": best_val_acc,
                    "Best Val Brier": best_val_brier,
                    "Test Acc": test_acc,
                    "Test Brier": test_brier
                }
            )

        per_run = pd.DataFrame(rows).sort_values("run").reset_index(drop=True)
        summary = pd.DataFrame(
            [
                {
                    "Metric": "Parameters",
                    "Mean": float(per_run["Parameters"].mean()),
                    "Std": float(per_run["Parameters"].std(ddof=1)),
                },
                {
                    "Metric": "Best Val Acc",
                    "Mean": float(per_run["Best Val Acc"].mean()),
                    "Std": float(per_run["Best Val Acc"].std(ddof=1)),
                },
                {
                    "Metric": "Best Val Brier",
                    "Mean": float(per_run["Best Val Brier"].mean()),
                    "Std": float(per_run["Best Val Brier"].std(ddof=1)),
                },
                                {
                    "Metric": "Test Acc",
                    "Mean": float(per_run["Test Acc"].mean()),
                    "Std": float(per_run["Test Acc"].std(ddof=1)),
                },
                                {
                    "Metric": "Test Brier",
                    "Mean": float(per_run["Test Brier"].mean()),
                    "Std": float(per_run["Test Brier"].std(ddof=1)),
                },
            ]
        )

        results[str(experiment)] = {"per_run": per_run, "summary": summary}

    return results


def plot_param_count(
    save_path="results",
    experiments=None,
    show_individual=False,
    alpha_band=0.25,
    show_checkpoint=True,
    save_path_out=None,
    show=True,
    figsize=(12, 6),
    dpi=150,
):
    """
    Plot parameter count vs epoch with mean line and std band per initial width,
    aggregated across run_* folders.

    Pass one lambda per call (e.g. all plasticity_*_1e-06); mixing lambdas with
    the same init_width pools them under that width.

    If show_individual is True, plot one line per run (no mean/std band) with a
    checkpoint dot per run at its selected epoch. Otherwise plot mean line, std
    band, and aggregated checkpoint markers per init_width.
    """
    base = Path(save_path)
    rows = []
    checkpoint_rows = []

    for experiment in experiments:
        meta = _parse_experiment_dir_name(experiment)
        for run_dir in sorted(base.glob("run_*")):
            exp_dir = run_dir / experiment
            metrics_path = exp_dir / "metrics.csv"
            summary_path = exp_dir / "experiment_summary.csv"
            if not metrics_path.exists() or not summary_path.exists():
                continue

            run_id = int(run_dir.name.split("_", 1)[1])
            summary = pd.read_csv(summary_path)
            best_epoch = int(summary.iloc[0]["Selected epoch"])
            metrics_df = pd.read_csv(metrics_path)

            for _, mrow in metrics_df.iterrows():
                rows.append(
                    {
                        "run": run_id,
                        "init_width": meta["init_width"],
                        "experiment": experiment,
                        "experiment_label": meta["label"],
                        "junctures_mode": meta.get("junctures_mode"),
                        "epoch": int(mrow["epoch"]),
                        "param_count": float(mrow["param_count"]),
                    }
                )

            best_rows = metrics_df.loc[metrics_df["epoch"] == best_epoch]
            if not best_rows.empty:
                checkpoint_rows.append(
                    {
                        "run": run_id,
                        "experiment": experiment,
                        "experiment_label": meta["label"],
                        "best_epoch": best_epoch,
                        "best_param_count": float(best_rows.iloc[0]["param_count"]),
                    }
                )

    if not rows:
        raise FileNotFoundError(
            f"No metrics.csv files found under {save_path} for experiments={experiments}"
        )

    df = pd.DataFrame(rows)
    stats = (
        df.groupby(["experiment", "epoch"], as_index=False)
        .agg(
            mean=("param_count", "mean"),
            std=("param_count", "std"),
            n_runs=("run", "nunique"),
            experiment_label=("experiment_label", "first"),
        )
    )
    stats["std"] = stats["std"].fillna(0.0)

    experiments_ordered = list(dict.fromkeys(df["experiment"]))
    colors = plt.cm.tab10.colors
    color_by_experiment = {
        exp: colors[i % len(colors)] for i, exp in enumerate(experiments_ordered)
    }
    label_by_experiment = df.groupby("experiment")["experiment_label"].first().to_dict()

    fig, ax = plt.subplots(figsize=figsize)

    if show_individual:
        for (run_id, experiment), sub in df.groupby(["run", "experiment"]):
            ax.plot(
                sub["epoch"],
                sub["param_count"],
                color=color_by_experiment[experiment],
                linestyle="-",
                alpha=0.85,
                linewidth=1.5,
            )
    else:
        for experiment in experiments_ordered:
            sub = stats.loc[stats["experiment"] == experiment].sort_values("epoch")
            if sub.empty:
                continue
            color = color_by_experiment[experiment]
            ax.plot(
                sub["epoch"],
                sub["mean"],
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )
            ax.fill_between(
                sub["epoch"],
                sub["mean"] - sub["std"],
                sub["mean"] + sub["std"],
                color=color,
                alpha=alpha_band,
            )

    if show_checkpoint and checkpoint_rows:
        ckpt_df = pd.DataFrame(checkpoint_rows)
        if show_individual:
            for _, row in ckpt_df.iterrows():
                experiment = row["experiment"]
                color = color_by_experiment[experiment]
                ax.scatter(
                    row["best_epoch"],
                    row["best_param_count"],
                    color=color,
                    s=50,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.5,
                )
        else:
            ckpt_stats = (
                ckpt_df.groupby("experiment", as_index=False)
                .agg(
                    best_epoch_mean=("best_epoch", "mean"),
                    best_epoch_std=("best_epoch", "std"),
                    best_param_mean=("best_param_count", "mean"),
                    best_param_std=("best_param_count", "std"),
                )
            )
            ckpt_stats["best_epoch_std"] = ckpt_stats["best_epoch_std"].fillna(0.0)
            ckpt_stats["best_param_std"] = ckpt_stats["best_param_std"].fillna(0.0)

            for _, row in ckpt_stats.iterrows():
                experiment = row["experiment"]
                color = color_by_experiment[experiment]
                ax.errorbar(
                    row["best_epoch_mean"],
                    row["best_param_mean"],
                    xerr=row["best_epoch_std"] if row["best_epoch_std"] > 0 else None,
                    yerr=row["best_param_std"] if row["best_param_std"] > 0 else None,
                    fmt="o",
                    color=color,
                    markersize=7,
                    capsize=3,
                    linestyle="none",
                    zorder=5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )

    color_handles = [
        Line2D(
            [0], [0],
            color=color_by_experiment[exp],
            lw=2,
            label=label_by_experiment.get(exp, exp),
        )
        for exp in experiments_ordered
        if exp in color_by_experiment
    ]
    ax.legend(handles=color_handles, title="Experiment", loc="upper right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter count")
    title_suffix = "per run" if show_individual else "mean ± std"
    checkpoint_note = "; dot = selected checkpoint" if show_checkpoint else ""
    ax.set_title(f"Parameter count vs epoch ({title_suffix}{checkpoint_note})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path_out is None:
        tag = "_".join(experiments[0].split("_")[1:3]) if len(experiments) == 1 else "multi"
        save_path_out = f"param_count_{tag}.png"
    fig.savefig(save_path_out, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, stats

def _junctures_mode_label(mode):
    if mode is None or (isinstance(mode, float) and pd.isna(mode)):
        return None
    if mode == "both":
        return "grow+prune"
    if mode == "prune":
        return "prune only"
    if mode == "grow":
        return "grow only"
    return str(mode)


_OPTIONAL_EXPERIMENT_DIR_TAGS = ("_vcl", "_p2junct", "_regrow")


def _strip_optional_experiment_dir_tags(name):
    """
    Strip trailing main4 tags (_vcl, _p2junct, _regrow) in any order.
    Returns (stripped_name, list of tag names without leading underscores).
    """
    name = str(name)
    tags = []
    changed = True
    while changed:
        changed = False
        for tag in _OPTIONAL_EXPERIMENT_DIR_TAGS:
            if name.endswith(tag):
                name = name[: -len(tag)]
                tags.append(tag.lstrip("_"))
                changed = True
                break
    return name, tags


def _parse_experiment_dir_name(dirname):
    """
    Parse folder names like:
      baseline_64
      baseline_50_vcl
      baseline_10f_10f          (CNN)
      plasticity_128_1e-06
      plasticity_500_1e-06_vcl
      plasticity_500_1e-06_p2junct_vcl
      plasticity_20f_20f_5e-06  (CNN)
      plasticity_300_1e-06_prune_only
      plasticity_300_1e-06_grow_only
      static_replay_400_1e-06
      static_replay_400_1e-06_prune_only
      nest_300_100_p0.1_refacc89_flooracc89
      dropnet_300_100_p0.2_modeglobal_kappa0.99
    Returns dict with keys: kind, init_width, lambda_penalty, junctures_mode, label, dir_tags
    (plus nest_p / nest_ref_acc / nest_floor_acc for Nest folders,
     dropnet_p / dropnet_mode / dropnet_kappa for DropNet folders).
    """
    raw_name = str(dirname)
    name, dir_tags = _strip_optional_experiment_dir_tags(raw_name)

    def _with_tags(meta):
        meta["dir_tags"] = list(dir_tags)
        return meta

    m = re.match(r"^baseline_(\d+)$", name)
    if m:
        return _with_tags({
            "kind": "baseline",
            "init_width": int(m.group(1)),
            "lambda_penalty": None,
            "junctures_mode": None,
            "label": f"baseline (init {m.group(1)})",
        })
    m = re.match(r"^baseline_(\d+)f(?:_\d+f)*$", name)
    if m:
        width = int(m.group(1))
        return _with_tags({
            "kind": "baseline",
            "init_width": width,
            "lambda_penalty": None,
            "junctures_mode": None,
            "label": f"baseline (init {m.group(0).split('_', 1)[1]})",
        })
    m = re.match(r"^three_phase_baseline_(\d+)$", name)
    if m:
        return _with_tags({
            "kind": "three_phase",
            "init_width": int(m.group(1)),
            "lambda_penalty": None,
            "junctures_mode": "three_phase",
            "label": f"three-phase baseline (init {m.group(1)})",
        })
    m = re.match(r"^three_phase_baseline_(\d+)f(?:_\d+f)*$", name)
    if m:
        width = int(m.group(1))
        channels = name[len("three_phase_baseline_"):]
        return _with_tags({
            "kind": "three_phase",
            "init_width": width,
            "lambda_penalty": None,
            "junctures_mode": "three_phase",
            "label": f"three-phase baseline (init {channels})",
        })
    m = re.match(r"^static_replay_(\d+)_([0-9.e+-]+)_(prune|grow)_only$", name)
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        mode = m.group(3)
        mode_label = _junctures_mode_label(mode)
        return _with_tags({
            "kind": "static_replay",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": "static_replay",
            "label": f"static replay {width} λ={lam:g} ({mode_label})",
        })
    m = re.match(r"^static_replay_(\d+)_([0-9.e+-]+)$", name)
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        return _with_tags({
            "kind": "static_replay",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": "static_replay",
            "label": f"static replay {width} λ={lam:g}",
        })
    m = re.match(r"^plasticity_three_phase_(\d+)_([0-9.e+-]+)_(prune|grow)_only$", name)
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        mode = m.group(3)
        mode_label = _junctures_mode_label(mode)
        return _with_tags({
            "kind": "plasticity_three_phase",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": mode,
            "label": f"plasticity+three-phase {width} λ={lam:g} ({mode_label})",
        })
    m = re.match(r"^plasticity_three_phase_(\d+)_([0-9.e+-]+)$", name)
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        return _with_tags({
            "kind": "plasticity_three_phase",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": "both",
            "label": f"plasticity+three-phase {width} λ={lam:g}",
        })
    m = re.match(
        r"^plasticity_three_phase_(\d+)f(?:_\d+f)*_([0-9.e+-]+)_(prune|grow)_only$",
        name,
    )
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        mode = m.group(3)
        mode_label = _junctures_mode_label(mode)
        channels = name[len("plasticity_three_phase_"):].rsplit("_", 3)[0]
        return _with_tags({
            "kind": "plasticity_three_phase",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": mode,
            "label": f"plasticity+three-phase {channels} λ={lam:g} ({mode_label})",
        })
    m = re.match(r"^plasticity_three_phase_(\d+)f(?:_\d+f)*_([0-9.e+-]+)$", name)
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        channels = name[len("plasticity_three_phase_"):].rsplit("_", 1)[0]
        return _with_tags({
            "kind": "plasticity_three_phase",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": "both",
            "label": f"plasticity+three-phase {channels} λ={lam:g}",
        })
    m = re.match(r"^plasticity_(\d+)_([0-9.e+-]+)_(prune|grow)_only$", name)
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        mode = m.group(3)
        mode_label = _junctures_mode_label(mode)
        return _with_tags({
            "kind": "plasticity",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": mode,
            "label": f"plasticity {width} λ={lam:g} ({mode_label})",
        })
    m = re.match(r"^plasticity_(\d+)_([0-9.e+-]+)$", name)
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        return _with_tags({
            "kind": "plasticity",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": "both",
            "label": f"plasticity {width} λ={lam:g}",
        })
    m = re.match(r"^plasticity_(\d+)f(?:_\d+f)*_([0-9.e+-]+)_(prune|grow)_only$", name)
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        mode = m.group(3)
        mode_label = _junctures_mode_label(mode)
        channels = name[len("plasticity_"):].rsplit("_", 3)[0]
        return _with_tags({
            "kind": "plasticity",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": mode,
            "label": f"plasticity {channels} λ={lam:g} ({mode_label})",
        })
    m = re.match(r"^plasticity_(\d+)f(?:_\d+f)*_([0-9.e+-]+)$", name)
    if m:
        width = int(m.group(1))
        lam = float(m.group(2))
        channels = name[len("plasticity_"):].rsplit("_", 1)[0]
        return _with_tags({
            "kind": "plasticity",
            "init_width": width,
            "lambda_penalty": lam,
            "junctures_mode": "both",
            "label": f"plasticity {channels} λ={lam:g}",
        })
    m = re.match(
        r"^nest_(\d+)_(\d+)_p([0-9.e+-]+)_refacc([0-9.e+-]+)_flooracc([0-9.e+-]+)$",
        name,
    )
    if m:
        w1 = int(m.group(1))
        w2 = int(m.group(2))
        nest_p = float(m.group(3))
        nest_ref_acc = float(m.group(4))
        nest_floor_acc = float(m.group(5))
        return _with_tags({
            "kind": "nest",
            "init_width": w1,
            "lambda_penalty": None,
            "junctures_mode": "nest",
            "nest_p": nest_p,
            "nest_ref_acc": nest_ref_acc,
            "nest_floor_acc": nest_floor_acc,
            "label": f"nest {w1}/{w2} p={nest_p:g} floor={nest_floor_acc:g}",
        })
    m = re.match(
        r"^dropnet_(\d+(?:_\d+)*)_p([0-9.e+-]+)_mode(global|layer)_kappa([0-9.e+-]+)$",
        name,
    )
    if m:
        widths = [int(w) for w in m.group(1).split("_")]
        dropnet_p = float(m.group(2))
        dropnet_mode = m.group(3)
        dropnet_kappa = float(m.group(4))
        width_label = "/".join(str(w) for w in widths)
        return _with_tags({
            "kind": "dropnet",
            "init_width": widths[0],
            "lambda_penalty": None,
            "junctures_mode": "dropnet",
            "dropnet_p": dropnet_p,
            "dropnet_mode": dropnet_mode,
            "dropnet_kappa": dropnet_kappa,
            "label": (
                f"dropnet {width_label} p={dropnet_p:g} "
                f"{dropnet_mode} κ={dropnet_kappa:g}"
            ),
        })
    return _with_tags({
        "kind": "other",
        "init_width": None,
        "lambda_penalty": None,
        "junctures_mode": None,
        "label": raw_name,
    })
def collect_experiment_summaries(
    save_path="results",
    experiments=None,
    experiment_glob="*",
    num_runs=None,
    x_col="Parameters",
    y_col="Test Acc",
):
    """
    Load per-run rows from experiment_summary.csv.
  Parameters
    ----------
    save_path : str
        Root folder containing run_1, run_2, ...
    experiments : list[str] or None
        Explicit experiment folder names, e.g.
        ["baseline_64", "plasticity_64_1e-06"].
        If None, auto-discover under each run folder using experiment_glob.
    experiment_glob : str
        Glob for auto-discovery, e.g. "plasticity_*", "baseline_*", "*".
    num_runs : int or None
        If set, only use run_1 .. run_{num_runs}.
    x_col, y_col : str
        Columns from experiment_summary.csv.
    Returns
    -------
    pd.DataFrame with one row per (run, experiment_folder).
    """
    base = Path(save_path)
    if not base.is_dir():
        raise FileNotFoundError(f"save_path not found: {save_path}")
    run_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    if num_runs is not None:
        run_dirs = [p for p in run_dirs if int(p.name.split("_", 1)[1]) <= int(num_runs)]
    if not run_dirs:
        raise FileNotFoundError(f"No run_* folders under {save_path}")
    rows = []
    for run_dir in run_dirs:
        run_id = int(run_dir.name.split("_", 1)[1])
        if experiments is not None:
            exp_dirs = [run_dir / e for e in experiments]
        else:
            exp_dirs = sorted(run_dir.glob(experiment_glob))
        for exp_dir in exp_dirs:
            if not exp_dir.is_dir():
                continue
            summary_path = exp_dir / "experiment_summary.csv"
            if not summary_path.exists():
                continue
            df = pd.read_csv(summary_path)
            if df.empty:
                continue
            row0 = df.iloc[0]
            meta = _parse_experiment_dir_name(exp_dir.name)
            csv_junctures = row0.get("Junctures Mode")
            junctures_mode = meta.get("junctures_mode")
            # Keep DropNet kind/mode from the folder name; CSV stores dropnet_global/layer.
            if meta.get("kind") != "dropnet":
                if csv_junctures is not None and not pd.isna(csv_junctures):
                    junctures_mode = str(csv_junctures)
            rows.append(
                {
                    "run": run_id,
                    "experiment": exp_dir.name,
                    "kind": meta["kind"],
                    "init_width": meta["init_width"],
                    "lambda_penalty": meta["lambda_penalty"],
                    "junctures_mode": junctures_mode,
                    "nest_p": meta.get("nest_p"),
                    "nest_ref_acc": meta.get("nest_ref_acc"),
                    "nest_floor_acc": meta.get("nest_floor_acc"),
                    "dropnet_p": meta.get("dropnet_p"),
                    "dropnet_mode": meta.get("dropnet_mode"),
                    "dropnet_kappa": meta.get("dropnet_kappa"),
                    "model_label": str(row0.get("Model", meta["kind"])),
                    "x": float(row0[x_col]),
                    "y": float(row0[y_col]),
                    "summary_path": str(summary_path),
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"No experiment_summary.csv files found under {save_path} "
            f"(experiments={experiments}, glob={experiment_glob!r})."
        )
    return pd.DataFrame(rows)


def _lambda_penalty_color_map(df, palette=None):
    """Distinct color per plasticity / static_replay lambda_penalty value."""
    lambdas = sorted(
        df.loc[df["kind"].isin(["plasticity", "static_replay"]), "lambda_penalty"]
        .dropna()
        .unique()
    )
    if palette is None:
        palette = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)
    return {lam: palette[i % len(palette)] for i, lam in enumerate(lambdas)}


def _nest_floor_color_map(df, palette=None):
    """Distinct color per Nest prune-acc floor value."""
    floors = sorted(
        df.loc[df["kind"] == "nest", "nest_floor_acc"].dropna().unique()
    )
    if palette is None:
        # Prefer a warm / distinct range from plasticity λ tab10 blues/reds.
        palette = list(plt.cm.Dark2.colors) + list(plt.cm.Set1.colors)
    return {floor: palette[i % len(palette)] for i, floor in enumerate(floors)}


def _dropnet_kappa_color_map(df, palette=None):
    """Distinct color per DropNet kappa value."""
    kappas = sorted(
        df.loc[df["kind"] == "dropnet", "dropnet_kappa"].dropna().unique()
    )
    if palette is None:
        palette = list(plt.cm.Set3.colors) + list(plt.cm.Dark2.colors)
    return {kappa: palette[i % len(palette)] for i, kappa in enumerate(kappas)}


def _junctures_mode_marker(mode):
    if mode == "grow":
        return "^"
    return "o"


def _plasticity_point_color(row, style_map, lambda_colors):
    lam = row.get("lambda_penalty")
    if lam is not None and not pd.isna(lam) and lam in lambda_colors:
        return lambda_colors[lam]
    kind = row.get("kind")
    if kind == "static_replay":
        return style_map.get("static_replay", style_map["plasticity"])["color"]
    return style_map["plasticity"]["color"]


def _point_style(
    row,
    style_map,
    lambda_colors,
    style_by_junctures=False,
    nest_floor_colors=None,
    dropnet_kappa_colors=None,
    dropnet_style_by_mode=False,
):
    kind = row["kind"]
    if kind == "baseline":
        color = style_map["baseline"]["color"]
        return {
            "color": color,
            "marker": style_map["baseline"]["marker"],
            "facecolor": color,
            "edgecolor": "black",
            "linewidth": 0.4,
        }
    if kind == "three_phase":
        style = style_map.get("three_phase", {"color": "#1f77b4", "marker": "D"})
        color = style["color"]
        return {
            "color": color,
            "marker": style["marker"],
            "facecolor": color,
            "edgecolor": "black",
            "linewidth": 0.4,
        }
    if kind == "plasticity_three_phase":
        style = style_map.get("plasticity_three_phase", {"color": "#9467bd", "marker": "P"})
        color = style["color"]
        return {
            "color": color,
            "marker": style["marker"],
            "facecolor": color,
            "edgecolor": "black",
            "linewidth": 0.4,
        }
    if kind == "static_replay":
        style = style_map.get("static_replay", {"color": "#2ca02c", "marker": "X"})
        color = _plasticity_point_color(row, style_map, lambda_colors)
        return {
            "color": color,
            "marker": style["marker"],
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.0,
        }
    if kind == "nest":
        style = style_map.get("nest", {"color": "#ff7f0e", "marker": "v"})
        floor = row.get("nest_floor_acc")
        if (
            nest_floor_colors is not None
            and floor is not None
            and not pd.isna(floor)
            and floor in nest_floor_colors
        ):
            color = nest_floor_colors[floor]
        else:
            color = style["color"]
        return {
            "color": color,
            "marker": style["marker"],
            "facecolor": color,
            "edgecolor": "black",
            "linewidth": 0.4,
        }
    if kind == "dropnet":
        style = style_map.get("dropnet", {"color": "#17becf", "marker": "^"})
        kappa = row.get("dropnet_kappa")
        if (
            dropnet_kappa_colors is not None
            and kappa is not None
            and not pd.isna(kappa)
            and kappa in dropnet_kappa_colors
        ):
            color = dropnet_kappa_colors[kappa]
        else:
            color = style["color"]
        mode = row.get("dropnet_mode")
        hollow = dropnet_style_by_mode and mode == "layer"
        return {
            "color": color,
            "marker": style["marker"],
            "facecolor": "white" if hollow else color,
            "edgecolor": color if hollow else "black",
            "linewidth": 1.0 if hollow else 0.4,
        }
    if kind == "plasticity":
        mode = row.get("junctures_mode")
        if mode is None or (isinstance(mode, float) and pd.isna(mode)):
            mode = "both"
        color = _plasticity_point_color(row, style_map, lambda_colors)
        marker = (
            _junctures_mode_marker(mode)
            if style_by_junctures
            else style_map["plasticity"]["marker"]
        )
        hollow = style_by_junctures and mode == "prune"
        return {
            "color": color,
            "marker": marker,
            "facecolor": "white" if hollow else color,
            "edgecolor": color,
            "linewidth": 1.0 if hollow else 0.5,
        }
    color = style_map["other"]["color"]
    return {
        "color": color,
        "marker": style_map["other"]["marker"],
        "facecolor": color,
        "edgecolor": "black",
        "linewidth": 0.4,
    }


def _scatter_point(ax, x, y, style, marker_size, alpha=0.9):
    ax.scatter(
        x,
        y,
        marker=style["marker"],
        s=marker_size,
        facecolors=style["facecolor"],
        edgecolors=style["edgecolor"],
        linewidths=style["linewidth"],
        alpha=alpha,
        zorder=4,
    )


def _infer_pareto_y_goal(y_col, ylabel=None):
    text = f"{y_col} {ylabel or ''}".lower()
    if any(k in text for k in ("brier", "loss", "error", "nll", "mse", "rmse")):
        return "minimize"
    return "maximize"


def _pareto_frontier_xy(points, y_goal="maximize"):
    """Non-dominated (x, y) pairs: lower x is better; y direction set by y_goal."""
    frontier = []
    if y_goal == "maximize":
        best_y = float("-inf")
        sort_key = lambda p: (p[0], -p[1])
        is_better = lambda y, best: y > best
    else:
        best_y = float("inf")
        sort_key = lambda p: (p[0], p[1])
        is_better = lambda y, best: y < best
    for x, y in sorted(points, key=sort_key):
        if is_better(y, best_y):
            frontier.append((x, y))
            best_y = y
    return frontier


def _draw_pareto_frontier(ax, df, color, label, y_goal="maximize", linestyle="--", linewidth=1.5):
    """Draw Pareto frontier line for a point set; returns legend handle or None."""
    if len(df) < 2:
        return None
    points = list(zip(df["x"], df["y"]))
    frontier = _pareto_frontier_xy(points, y_goal=y_goal)
    if len(frontier) < 2:
        return None
    deduped = list(dict.fromkeys(frontier))
    if len(deduped) < 2:
        return None
    xs, ys = zip(*deduped)
    (line,) = ax.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=3,
        label=label,
    )
    return line


def _draw_kind_pareto_frontier(ax, df, kind, color, label, y_goal="maximize", linestyle="--", linewidth=1.5):
    """Draw Pareto frontier line for one experiment kind; returns legend handle or None."""
    subset = df.loc[df["kind"] == kind]
    return _draw_pareto_frontier(
        ax, subset, color=color, label=label, y_goal=y_goal, linestyle=linestyle, linewidth=linewidth
    )


# Kinds whose λ-sweeps are tied to a fixed initial width (split Pareto by init_width).
_PARETO_SPLIT_BY_INIT_WIDTH_KINDS = frozenset(
    {"plasticity", "static_replay", "plasticity_three_phase"}
)
# Same kinds can also be split by grow/prune juncture mode.
_PARETO_SPLIT_BY_JUNCTURES_KINDS = frozenset(
    {"plasticity", "static_replay", "plasticity_three_phase"}
)
_DEFAULT_JUNCTURES_PARETO_LINESTYLES = {
    "both": "-",
    "prune": "--",
    "grow": ":",
}


def _init_width_pareto_colors(widths):
    """Distinct colors for init-width Pareto lines."""
    palette = [
        "#d62728",  # red
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
    ]
    widths_sorted = sorted(
        {
            int(w)
            for w in widths
            if w is not None and not (isinstance(w, float) and pd.isna(w))
        }
    )
    return {w: palette[i % len(palette)] for i, w in enumerate(widths_sorted)}


def _normalize_junctures_mode_for_pareto(mode):
    if mode is None or (isinstance(mode, float) and pd.isna(mode)):
        return "both"
    mode = str(mode)
    return mode if mode else "both"


def _resolve_pareto_linestyle(pareto_linestyle, kind, init_width=None, junctures_mode=None):
    if not isinstance(pareto_linestyle, dict):
        if junctures_mode is not None and pareto_linestyle == "--":
            # Default string "--" still allows mode-specific styles when splitting.
            return _DEFAULT_JUNCTURES_PARETO_LINESTYLES.get(
                _normalize_junctures_mode_for_pareto(junctures_mode), pareto_linestyle
            )
        return pareto_linestyle
    mode = (
        _normalize_junctures_mode_for_pareto(junctures_mode)
        if junctures_mode is not None
        else None
    )
    if mode is not None and (kind, mode) in pareto_linestyle:
        return pareto_linestyle[(kind, mode)]
    if mode is not None and mode in pareto_linestyle:
        return pareto_linestyle[mode]
    if init_width is not None and (kind, init_width) in pareto_linestyle:
        return pareto_linestyle[(kind, init_width)]
    if init_width is not None and init_width in pareto_linestyle:
        return pareto_linestyle[init_width]
    if kind in pareto_linestyle:
        return pareto_linestyle[kind]
    if mode is not None:
        return _DEFAULT_JUNCTURES_PARETO_LINESTYLES.get(mode, "--")
    return "--"


def _draw_kind_pareto_frontiers(
    ax,
    df,
    kind,
    color,
    y_goal="maximize",
    linestyle="--",
    linewidth=1.5,
    split_by_init_width=False,
    split_by_junctures_mode=False,
    init_width_colors=None,
):
    """
    Draw one or more Pareto frontiers for a kind.
    If split_by_init_width and kind is plasticity-like, draw a frontier per init_width.
    If split_by_junctures_mode and kind is plasticity-like, draw a frontier per
    junctures_mode (both / prune / grow).
    Returns a list of legend handles.
    """
    handles = []
    subset = df.loc[df["kind"] == kind]
    if subset.empty:
        return handles

    if split_by_junctures_mode and kind in _PARETO_SPLIT_BY_JUNCTURES_KINDS:
        if "junctures_mode" not in subset.columns:
            handle = _draw_pareto_frontier(
                ax,
                subset,
                color=color,
                label=f"{kind} Pareto",
                y_goal=y_goal,
                linestyle=_resolve_pareto_linestyle(linestyle, kind),
                linewidth=linewidth,
            )
            if handle is not None:
                handles.append(handle)
            return handles

        mode_series = subset["junctures_mode"].map(_normalize_junctures_mode_for_pareto)
        work = subset.assign(junctures_mode=mode_series)
        for mode, sub in work.groupby("junctures_mode", dropna=False):
            mode_label = _normalize_junctures_mode_for_pareto(mode)
            label = f"{kind} {mode_label} Pareto"
            ls = _resolve_pareto_linestyle(
                linestyle, kind, junctures_mode=mode_label
            )
            handle = _draw_pareto_frontier(
                ax,
                sub,
                color=color,
                label=label,
                y_goal=y_goal,
                linestyle=ls,
                linewidth=linewidth,
            )
            if handle is not None:
                handles.append(handle)
        return handles

    if split_by_init_width and kind in _PARETO_SPLIT_BY_INIT_WIDTH_KINDS:
        if "init_width" not in subset.columns:
            handle = _draw_pareto_frontier(
                ax,
                subset,
                color=color,
                label=f"{kind} Pareto",
                y_goal=y_goal,
                linestyle=_resolve_pareto_linestyle(linestyle, kind),
                linewidth=linewidth,
            )
            if handle is not None:
                handles.append(handle)
            return handles

        width_colors = init_width_colors or _init_width_pareto_colors(
            subset["init_width"].dropna().unique()
        )
        for width, sub in subset.groupby("init_width", dropna=False):
            if width is None or (isinstance(width, float) and pd.isna(width)):
                label = f"{kind} Pareto"
                line_color = color
            else:
                width_int = int(width)
                label = f"{kind} init={width_int} Pareto"
                line_color = width_colors.get(width_int, color)
            ls = _resolve_pareto_linestyle(linestyle, kind, init_width=width)
            handle = _draw_pareto_frontier(
                ax,
                sub,
                color=line_color,
                label=label,
                y_goal=y_goal,
                linestyle=ls,
                linewidth=linewidth,
            )
            if handle is not None:
                handles.append(handle)
        return handles

    handle = _draw_pareto_frontier(
        ax,
        subset,
        color=color,
        label=f"{kind} Pareto",
        y_goal=y_goal,
        linestyle=_resolve_pareto_linestyle(linestyle, kind),
        linewidth=linewidth,
    )
    if handle is not None:
        handles.append(handle)
    return handles


def _legend_handles_for_test_acc_plot(
    df,
    style_map,
    lambda_colors,
    style_by_junctures=False,
    nest_floor_colors=None,
    dropnet_kappa_colors=None,
    dropnet_style_by_mode=False,
):
    handles = []
    if nest_floor_colors is None:
        nest_floor_colors = {}
    if dropnet_kappa_colors is None:
        dropnet_kappa_colors = {}
    if (df["kind"] == "baseline").any():
        handles.append(
            Line2D(
                [0], [0],
                marker=style_map["baseline"]["marker"],
                color="w",
                markerfacecolor=style_map["baseline"]["color"],
                markeredgecolor="black",
                markersize=8,
                label="baseline",
            )
        )
    if (df["kind"] == "three_phase").any():
        tp_style = style_map.get("three_phase", {"color": "#1f77b4", "marker": "D"})
        handles.append(
            Line2D(
                [0], [0],
                marker=tp_style["marker"],
                color="w",
                markerfacecolor=tp_style["color"],
                markeredgecolor="black",
                markersize=8,
                label="three-phase baseline",
            )
        )
    if (df["kind"] == "plasticity_three_phase").any():
        hybrid_style = style_map.get("plasticity_three_phase", {"color": "#9467bd", "marker": "P"})
        handles.append(
            Line2D(
                [0], [0],
                marker=hybrid_style["marker"],
                color="w",
                markerfacecolor=hybrid_style["color"],
                markeredgecolor="black",
                markersize=8,
                label="plasticity + three-phase",
            )
        )
    plasticity_df = df.loc[df["kind"] == "plasticity"]
    if style_by_junctures and not plasticity_df.empty:
        pairs = plasticity_df[["lambda_penalty", "junctures_mode"]].drop_duplicates()
        for _, pair in pairs.sort_values(["lambda_penalty", "junctures_mode"]).iterrows():
            lam = pair["lambda_penalty"]
            mode = pair["junctures_mode"]
            if mode is None or (isinstance(mode, float) and pd.isna(mode)):
                mode = "both"
            color = lambda_colors.get(lam, style_map["plasticity"]["color"])
            marker = _junctures_mode_marker(mode)
            hollow = mode == "prune"
            mode_label = _junctures_mode_label(mode)
            handles.append(
                Line2D(
                    [0], [0],
                    marker=marker,
                    color="w",
                    markerfacecolor="white" if hollow else color,
                    markeredgecolor=color,
                    markeredgewidth=1.0 if hollow else 0.5,
                    markersize=8,
                    label=f"λ={lam:g} ({mode_label})",
                )
            )
    elif not plasticity_df.empty:
        for lam in sorted(plasticity_df["lambda_penalty"].dropna().unique()):
            color = lambda_colors.get(lam, style_map["plasticity"]["color"])
            handles.append(
                Line2D(
                    [0], [0],
                    marker=style_map["plasticity"]["marker"],
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.5,
                    markersize=8,
                    label=f"plasticity λ={lam:g}",
                )
            )
    static_replay_df = df.loc[df["kind"] == "static_replay"]
    if not static_replay_df.empty:
        sr_style = style_map.get("static_replay", {"color": "#2ca02c", "marker": "X"})
        for lam in sorted(static_replay_df["lambda_penalty"].dropna().unique()):
            color = lambda_colors.get(lam, sr_style["color"])
            handles.append(
                Line2D(
                    [0], [0],
                    marker=sr_style["marker"],
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    markersize=8,
                    label=f"static replay λ={lam:g}",
                )
            )
    nest_df = df.loc[df["kind"] == "nest"]
    if not nest_df.empty:
        nest_style = style_map.get("nest", {"color": "#ff7f0e", "marker": "v"})
        floors = sorted(nest_df["nest_floor_acc"].dropna().unique())
        if floors:
            for floor in floors:
                color = nest_floor_colors.get(floor, nest_style["color"])
                handles.append(
                    Line2D(
                        [0], [0],
                        marker=nest_style["marker"],
                        color="w",
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markeredgewidth=0.4,
                        markersize=8,
                        label=f"nest floor={floor:g}",
                    )
                )
        else:
            handles.append(
                Line2D(
                    [0], [0],
                    marker=nest_style["marker"],
                    color="w",
                    markerfacecolor=nest_style["color"],
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    markersize=8,
                    label="nest",
                )
            )
    dropnet_df = df.loc[df["kind"] == "dropnet"]
    if not dropnet_df.empty:
        dropnet_style = style_map.get("dropnet", {"color": "#17becf", "marker": "^"})
        kappas = sorted(dropnet_df["dropnet_kappa"].dropna().unique())
        modes = sorted(dropnet_df["dropnet_mode"].dropna().unique())
        if kappas:
            mode_iter = modes if dropnet_style_by_mode and modes else [None]
            for kappa in kappas:
                for mode in mode_iter:
                    color = dropnet_kappa_colors.get(kappa, dropnet_style["color"])
                    hollow = dropnet_style_by_mode and mode == "layer"
                    if mode is None:
                        label = f"dropnet κ={kappa:g}"
                    else:
                        label = f"dropnet {mode} κ={kappa:g}"
                    handles.append(
                        Line2D(
                            [0], [0],
                            marker=dropnet_style["marker"],
                            color="w",
                            markerfacecolor="white" if hollow else color,
                            markeredgecolor=color if hollow else "black",
                            markeredgewidth=1.0 if hollow else 0.4,
                            markersize=8,
                            label=label,
                        )
                    )
        else:
            handles.append(
                Line2D(
                    [0], [0],
                    marker=dropnet_style["marker"],
                    color="w",
                    markerfacecolor=dropnet_style["color"],
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    markersize=8,
                    label="dropnet",
                )
            )
    if (df["kind"] == "other").any():
        handles.append(
            Line2D(
                [0], [0],
                marker=style_map["other"]["marker"],
                color="w",
                markerfacecolor=style_map["other"]["color"],
                markersize=8,
                label="other",
            )
        )
    return handles


def plot_param_count_vs_test_acc(
    save_path="results",
    experiments=None,
    experiment_glob="*",
    num_runs=None,
    aggregate_runs=True,
    x_col="Parameters",
    y_col="Test Acc",
    xlabel="Parameter count",
    ylabel="Test accuracy (%)",
    title="Parameter count vs test accuracy",
    save_path_out=None,
    show=True,
    figsize=(10, 7),
    dpi=150,
    alpha_individual=0.35,
    marker_size=70,
    annotate_points=False,
    group_by="experiment",
    style_map=None,
    lambda_palette=None,
    style_by="junctures_mode",
    show_pareto_frontier=False,
    pareto_scope="per_kind",
    pareto_y_goal=None,
    pareto_frontier_kinds=("baseline", "plasticity"),
    pareto_linestyle="--",
    pareto_linewidth=1.5,
):
    """
    Plot parameter count (x) against test accuracy (y) across experiments.
    Typical usage
    -------------
    # All plasticity + baseline runs, aggregated over 5 seeds:
    plot_param_count_vs_test_acc(
        save_path="results",
        experiment_glob="*",
        num_runs=5,
        aggregate_runs=True,
        save_path_out="param_count_vs_test_acc_multi.png",
    )
    # Only one penalty, multiple init widths:
    plot_param_count_vs_test_acc(
        save_path="results",
        experiment_glob="plasticity_*_1e-06",
        experiments=None,
        num_runs=5,
    )
    # Explicit list:
    plot_param_count_vs_test_acc(
        save_path="results",
        experiments=[
            "baseline_16", "baseline_32", "baseline_64",
            "plasticity_16_1e-06", "plasticity_32_1e-06",
        ],
        num_runs=5,
    )
    # Separate plasticity Pareto per initial width:
    plot_param_count_vs_test_acc(
        save_path="results",
        experiments=[..., "plasticity_20_1e-06", "plasticity_500_1e-06", ...],
        show_pareto_frontier=True,
        pareto_scope="per_init_width",
        pareto_frontier_kinds=("baseline", "plasticity"),
    )
    Parameters
    ----------
    aggregate_runs : bool
        If True, plot mean ± std per experiment group.
        If False, plot every run as its own point.
    group_by : str
        Column used for aggregation when aggregate_runs=True.
        Prefer "experiment" (default) so both/prune/grow ablations are not merged.
        Use "lambda_penalty" only when filtering to a single junctures_mode.
    style_by : str or None
        Column used to vary marker shape/fill for plasticity points.
        Defaults to "junctures_mode"; auto-disabled when only one mode is present.
    style_map : dict or None
        Optional override for baseline/other markers and fallback plasticity style.
    lambda_palette : list or None
        Colors assigned in order to distinct plasticity lambda_penalty values.
        Defaults to tab10 + Set2.
    show_pareto_frontier : bool
        If True, draw Pareto frontier line(s) (lower params; y direction auto-detected).
    pareto_scope : str
        "global" for one frontier over all points;
        "per_kind" for separate baseline/plasticity lines;
        "per_init_width" like per_kind, but plasticity / static_replay /
        plasticity_three_phase get a separate frontier per initial width;
        "per_junctures_mode" like per_kind, but plasticity-like kinds get a
        separate frontier per junctures_mode (both / prune / grow).
    pareto_y_goal : str or None
        "maximize" or "minimize" for the y-axis objective. If None, inferred from y_col/ylabel
        (e.g. Test Brier -> minimize, Test Acc -> maximize).
    pareto_frontier_kinds : tuple of str
        Which experiment kinds receive a frontier line when pareto_scope is
        "per_kind", "per_init_width", or "per_junctures_mode".
    pareto_linestyle : str or dict
        Line style for frontier(s); dict maps kind -> linestyle, or init_width /
        (kind, init_width) when using per_init_width, or junctures_mode /
        (kind, junctures_mode) when using per_junctures_mode.
    pareto_linewidth : float
        Width of frontier lines.
    """
    if pareto_scope not in ("global", "per_kind", "per_init_width", "per_junctures_mode"):
        raise ValueError(
            f"pareto_scope must be 'global', 'per_kind', 'per_init_width', or "
            f"'per_junctures_mode', got {pareto_scope!r}"
        )
    df = collect_experiment_summaries(
        save_path=save_path,
        experiments=experiments,
        experiment_glob=experiment_glob,
        num_runs=num_runs,
        x_col=x_col,
        y_col=y_col,
    )
    if style_map is None:
        style_map = {
            "baseline": {"color": "black", "marker": "s"},
            "three_phase": {"color": "#1f77b4", "marker": "D"},
            "plasticity_three_phase": {"color": "#9467bd", "marker": "P"},
            "plasticity": {"color": "#d62728", "marker": "o"},
            "static_replay": {"color": "#2ca02c", "marker": "X"},
            "nest": {"color": "#ff7f0e", "marker": "v"},
            "dropnet": {"color": "#17becf", "marker": "^"},
            "other": {"color": "gray", "marker": "x"},
        }
    lambda_colors = _lambda_penalty_color_map(df, palette=lambda_palette)
    nest_floor_colors = _nest_floor_color_map(df, palette=lambda_palette)
    dropnet_kappa_colors = _dropnet_kappa_color_map(df, palette=lambda_palette)
    plasticity_modes = df.loc[df["kind"] == "plasticity", "junctures_mode"].dropna().unique()
    style_by_junctures = (
        style_by == "junctures_mode"
        and len(plasticity_modes) > 1
    )
    dropnet_modes = df.loc[df["kind"] == "dropnet", "dropnet_mode"].dropna().unique()
    dropnet_style_by_mode = len(dropnet_modes) > 1
    fig, ax = plt.subplots(figsize=figsize)
    grouped = None
    if not aggregate_runs:
        for _, row in df.iterrows():
            style = _point_style(
                row,
                style_map,
                lambda_colors,
                style_by_junctures,
                nest_floor_colors,
                dropnet_kappa_colors,
                dropnet_style_by_mode,
            )
            _scatter_point(ax, row["x"], row["y"], style, marker_size)
    else:
        grouped = (
            df.groupby(group_by, dropna=False)
            .agg(
                x_mean=("x", "mean"),
                x_std=("x", "std"),
                y_mean=("y", "mean"),
                y_std=("y", "std"),
                kind=("kind", "first"),
                init_width=("init_width", "first"),
                lambda_penalty=("lambda_penalty", "first"),
                junctures_mode=("junctures_mode", "first"),
                nest_floor_acc=("nest_floor_acc", "first"),
                nest_p=("nest_p", "first"),
                dropnet_kappa=("dropnet_kappa", "first"),
                dropnet_mode=("dropnet_mode", "first"),
                dropnet_p=("dropnet_p", "first"),
                n_runs=("run", "nunique"),
            )
            .reset_index()
        )
        for _, row in grouped.iterrows():
            style = _point_style(
                row,
                style_map,
                lambda_colors,
                style_by_junctures,
                nest_floor_colors,
                dropnet_kappa_colors,
                dropnet_style_by_mode,
            )
            ax.errorbar(
                row["x_mean"], row["y_mean"],
                xerr=row["x_std"] if pd.notna(row["x_std"]) else None,
                yerr=row["y_std"] if pd.notna(row["y_std"]) else None,
                fmt=style["marker"],
                color=style["edgecolor"],
                markerfacecolor=style["facecolor"],
                markeredgecolor=style["edgecolor"],
                markeredgewidth=style["linewidth"],
                markersize=8,
                capsize=3,
                linestyle="none",
                alpha=0.95,
            )
            if annotate_points:
                ax.annotate(
                    str(row[group_by]),
                    (row["x_mean"], row["y_mean"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        for _, row in df.iterrows():
            style = _point_style(
                row,
                style_map,
                lambda_colors,
                style_by_junctures,
                nest_floor_colors,
                dropnet_kappa_colors,
                dropnet_style_by_mode,
            )
            _scatter_point(
                ax, row["x"], row["y"], style, marker_size * 0.5, alpha=alpha_individual
            )

    pareto_handles = []
    if show_pareto_frontier:
        if aggregate_runs:
            pareto_source = grouped.rename(columns={"x_mean": "x", "y_mean": "y"})
        else:
            pareto_source = (
                df.groupby(["kind", "experiment"], dropna=False)
                .agg(
                    x=("x", "mean"),
                    y=("y", "mean"),
                    init_width=("init_width", "first"),
                    junctures_mode=("junctures_mode", "first"),
                )
                .reset_index()
            )
        resolved_y_goal = pareto_y_goal or _infer_pareto_y_goal(y_col, ylabel)
        kind_colors = {
            "baseline": style_map["baseline"]["color"],
            "three_phase": style_map.get("three_phase", {"color": "#1f77b4"})["color"],
            "plasticity_three_phase": style_map.get(
                "plasticity_three_phase", {"color": "#9467bd"}
            )["color"],
            "plasticity": style_map["plasticity"]["color"],
            "static_replay": style_map.get("static_replay", {"color": "#2ca02c"})["color"],
            "nest": style_map.get("nest", {"color": "#ff7f0e"})["color"],
            "dropnet": style_map.get("dropnet", {"color": "#17becf"})["color"],
            "other": style_map["other"]["color"],
        }
        if pareto_scope == "global":
            handle = _draw_pareto_frontier(
                ax,
                pareto_source,
                color="0.25",
                label="Pareto frontier",
                y_goal=resolved_y_goal,
                linestyle=pareto_linestyle if not isinstance(pareto_linestyle, dict) else "--",
                linewidth=pareto_linewidth,
            )
            if handle is not None:
                pareto_handles.append(handle)
        else:
            split_by_init_width = pareto_scope == "per_init_width"
            split_by_junctures_mode = pareto_scope == "per_junctures_mode"
            width_colors = None
            if split_by_init_width and "init_width" in pareto_source.columns:
                width_colors = _init_width_pareto_colors(
                    pareto_source.loc[
                        pareto_source["kind"].isin(_PARETO_SPLIT_BY_INIT_WIDTH_KINDS),
                        "init_width",
                    ]
                    .dropna()
                    .unique()
                )
            for kind in pareto_frontier_kinds:
                if kind not in pareto_source["kind"].values:
                    continue
                pareto_handles.extend(
                    _draw_kind_pareto_frontiers(
                        ax,
                        pareto_source,
                        kind,
                        color=kind_colors.get(kind, "gray"),
                        y_goal=resolved_y_goal,
                        linestyle=pareto_linestyle,
                        linewidth=pareto_linewidth,
                        split_by_init_width=split_by_init_width,
                        split_by_junctures_mode=split_by_junctures_mode,
                        init_width_colors=width_colors,
                    )
                )

    handles = _legend_handles_for_test_acc_plot(
        df,
        style_map,
        lambda_colors,
        style_by_junctures,
        nest_floor_colors=nest_floor_colors,
        dropnet_kappa_colors=dropnet_kappa_colors,
        dropnet_style_by_mode=dropnet_style_by_mode,
    )
    handles.extend(pareto_handles)
    legend_title = "Model / λ / junctures" if style_by_junctures else "Model / λ / nest floor"
    ax.legend(handles=handles, title=legend_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path_out is not None:
        fig.savefig(save_path_out, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, df
