import ast
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metrics(metrics_dict, save_path="./results/metrics_comparison.png"):
    """Plot comparison of metrics across models."""
    colors = {
        "baseline": "blue",
        "plasticity": "red",
        "three_phase_baseline": "green",
        "static_replay": "purple",
    }

    fig, axs = plt.subplots(4, 3, figsize=(20, 15))

    ax = axs[0, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["train_loss_total"]))
        ax.plot(epochs, metrics["train_loss_total"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Training Loss (nll + kl)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axs[0, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["train_loss_nll"]))
        ax.plot(epochs, metrics["train_loss_nll"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Training Loss (nll)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axs[0, 2]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["train_loss_kl"]))
        ax.plot(epochs, metrics["train_loss_kl"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Training Loss (kl)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axs[1, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["train_acc"]))
        ax.plot(epochs, metrics["train_acc"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Training Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()

    ax = axs[1, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["train_brier"]))
        ax.plot(epochs, metrics["train_brier"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Training Brier")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Brier")
    ax.legend()

    axs[1, 2].axis("off")

    ax = axs[2, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["val_loss_total"]))
        ax.plot(epochs, metrics["val_loss_total"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Validation Loss (nll + kl)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axs[2, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["val_loss_nll"]))
        ax.plot(epochs, metrics["val_loss_nll"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Validation Loss (nll)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axs[2, 2]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["val_loss_kl"]))
        ax.plot(epochs, metrics["val_loss_kl"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Validation Loss (kl)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axs[3, 0]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["val_acc"]))
        ax.plot(epochs, metrics["val_acc"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()

    ax = axs[3, 1]
    for model_name, metrics in metrics_dict.items():
        epochs = range(1, 1 + len(metrics["val_brier"]))
        ax.plot(epochs, metrics["val_brier"], color=colors.get(model_name, "gray"), label=model_name)
    ax.set_title("Validation Brier")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Brier")
    ax.legend()

    axs[3, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _parse_experiment_dir_name(dirname):
    """
    Parse CNN experiment directory names, e.g.:
      baseline_20f_20f
      plasticity_20f_20f_5e-06
      plasticity_20f_20f_5e-06_prune_only
      static_replay_20f_20f_5e-06
      three_phase_baseline_20f_20f
    """
    name = os.path.basename(str(dirname).rstrip("/"))

    m = re.match(
        r"^plasticity_((?:\d+f_)*\d+f)_([0-9.e+-]+)_(prune|grow|both_gp)_only$",
        name,
    )
    if m:
        return {
            "kind": "plasticity",
            "conv_tag": m.group(1),
            "lambda_penalty": float(m.group(2)),
            "junctures_mode": m.group(3),
        }

    m = re.match(r"^plasticity_((?:\d+f_)*\d+f)_([0-9.e+-]+)$", name)
    if m:
        return {
            "kind": "plasticity",
            "conv_tag": m.group(1),
            "lambda_penalty": float(m.group(2)),
            "junctures_mode": "both",
        }

    m = re.match(
        r"^static_replay_((?:\d+f_)*\d+f)_([0-9.e+-]+)_(prune|grow|both_gp)_only$",
        name,
    )
    if m:
        return {
            "kind": "static_replay",
            "conv_tag": m.group(1),
            "lambda_penalty": float(m.group(2)),
            "junctures_mode": "static_replay",
        }

    m = re.match(r"^static_replay_((?:\d+f_)*\d+f)_([0-9.e+-]+)$", name)
    if m:
        return {
            "kind": "static_replay",
            "conv_tag": m.group(1),
            "lambda_penalty": float(m.group(2)),
            "junctures_mode": "static_replay",
        }

    m = re.match(r"^baseline_((?:\d+f_)*\d+f)$", name)
    if m:
        return {
            "kind": "baseline",
            "conv_tag": m.group(1),
            "lambda_penalty": None,
            "junctures_mode": "both",
        }

    m = re.match(r"^three_phase_baseline_((?:\d+f_)*\d+f)$", name)
    if m:
        return {
            "kind": "three_phase",
            "conv_tag": m.group(1),
            "lambda_penalty": None,
            "junctures_mode": "three_phase",
        }

    return {
        "kind": "unknown",
        "conv_tag": None,
        "lambda_penalty": None,
        "junctures_mode": None,
    }


def _pareto_frontier_xy(points, y_goal="maximize"):
    """points: list of (x, y); return frontier sorted by x ascending."""
    if not points:
        return [], []
    pts = sorted(points, key=lambda p: (p[0], -p[1] if y_goal == "maximize" else p[1]))
    frontier = []
    best_y = None
    for x, y in pts:
        if best_y is None:
            frontier.append((x, y))
            best_y = y
            continue
        if y_goal == "maximize":
            if y > best_y:
                frontier.append((x, y))
                best_y = y
        else:
            if y < best_y:
                frontier.append((x, y))
                best_y = y
    xs = [p[0] for p in frontier]
    ys = [p[1] for p in frontier]
    return xs, ys


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
    show_pareto_frontier=False,
    pareto_frontier_kinds=("baseline", "plasticity"),
    pareto_y_goal=None,
    pareto_linestyle="--",
    pareto_linewidth=1.5,
):
    """
    Collect experiment_summary.csv under save_path/run_*/{experiment}/ and plot
    parameter count vs test metric for CNN plasticity results.
    """
    save_path = os.path.normpath(str(save_path))
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(save_path, "run_*")) if os.path.isdir(d)
    )
    if num_runs is not None:
        run_dirs = [
            d for d in run_dirs
            if re.match(rf"run_([1-9]|[1-9]\d)$", os.path.basename(d))
            and int(os.path.basename(d).split("_")[1]) <= int(num_runs)
        ]

    rows = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        if experiments is None:
            exp_dirs = sorted(
                d for d in glob.glob(os.path.join(run_dir, experiment_glob))
                if os.path.isdir(d)
            )
        else:
            exp_dirs = [os.path.join(run_dir, e) for e in experiments]

        for exp_dir in exp_dirs:
            if not os.path.isdir(exp_dir):
                continue
            summary_path = os.path.join(exp_dir, "experiment_summary.csv")
            if not os.path.exists(summary_path):
                continue
            df = pd.read_csv(summary_path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            exp_name = os.path.basename(exp_dir)
            parsed = _parse_experiment_dir_name(exp_name)
            row["experiment"] = exp_name
            row["run"] = run_name
            row["kind"] = parsed["kind"]
            row["lambda_penalty"] = parsed.get("lambda_penalty")
            row["junctures_mode"] = parsed.get("junctures_mode")
            row["conv_tag"] = parsed.get("conv_tag")
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No experiment_summary.csv found under {save_path}/run_*/"
        )

    df_all = pd.DataFrame(rows)
    if x_col not in df_all.columns or y_col not in df_all.columns:
        raise ValueError(f"Columns {x_col!r}/{y_col!r} missing from summaries")

    style_map = {
        "baseline": {"color": "#1f77b4", "marker": "o"},
        "plasticity": {"color": "#d62728", "marker": "s"},
        "static_replay": {"color": "#2ca02c", "marker": "X"},
        "three_phase": {"color": "#9467bd", "marker": "^"},
        "unknown": {"color": "gray", "marker": "."},
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if aggregate_runs:
        group_cols = ["experiment", "kind"]
        agg = (
            df_all.groupby(group_cols, dropna=False)
            .agg(
                x_mean=(x_col, "mean"),
                x_std=(x_col, "std"),
                y_mean=(y_col, "mean"),
                y_std=(y_col, "std"),
                n=("run", "count"),
            )
            .reset_index()
        )
        for _, r in agg.iterrows():
            kind = r["kind"]
            style = style_map.get(kind, style_map["unknown"])
            ax.errorbar(
                r["x_mean"],
                r["y_mean"],
                xerr=0 if pd.isna(r["x_std"]) else r["x_std"],
                yerr=0 if pd.isna(r["y_std"]) else r["y_std"],
                fmt=style["marker"],
                color=style["color"],
                markersize=np.sqrt(marker_size),
                label=r["experiment"],
                capsize=3,
            )
    else:
        for _, r in df_all.iterrows():
            kind = r["kind"]
            style = style_map.get(kind, style_map["unknown"])
            ax.scatter(
                r[x_col],
                r[y_col],
                c=style["color"],
                marker=style["marker"],
                s=marker_size,
                alpha=alpha_individual,
                label=r["experiment"],
            )

    if show_pareto_frontier:
        if pareto_y_goal is None:
            pareto_y_goal = "minimize" if "brier" in y_col.lower() else "maximize"
        for kind in pareto_frontier_kinds:
            sub = df_all.loc[df_all["kind"] == kind]
            if sub.empty:
                continue
            if aggregate_runs:
                pts = list(
                    zip(
                        sub.groupby("experiment")[x_col].mean(),
                        sub.groupby("experiment")[y_col].mean(),
                    )
                )
            else:
                pts = list(zip(sub[x_col], sub[y_col]))
            xs, ys = _pareto_frontier_xy(pts, y_goal=pareto_y_goal)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    linestyle=pareto_linestyle,
                    linewidth=pareto_linewidth,
                    color=style_map.get(kind, style_map["unknown"])["color"],
                    label=f"{kind} Pareto",
                )

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            uniq.append((h, lab))
    if uniq:
        ax.legend([h for h, _ in uniq], [lab for _, lab in uniq], fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path_out is None:
        save_path_out = os.path.join(save_path, "param_count_vs_test_acc.png")
    fig.savefig(save_path_out)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path_out
