import ast
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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


_PARETO_SPLIT_BY_JUNCTURES_KINDS = frozenset(
    {"plasticity", "static_replay", "plasticity_three_phase"}
)
_DEFAULT_JUNCTURES_PARETO_LINESTYLES = {
    "both": "-",
    "prune": "--",
    "grow": ":",
    "both_gp": "-.",
}

_DEFAULT_STYLE_MAP = {
    "baseline": {"color": "black", "marker": "s"},
    "three_phase": {"color": "#1f77b4", "marker": "D"},
    "plasticity_three_phase": {"color": "#9467bd", "marker": "P"},
    "plasticity": {"color": "#d62728", "marker": "o"},
    "static_replay": {"color": "#2ca02c", "marker": "X"},
    "other": {"color": "gray", "marker": "x"},
    "unknown": {"color": "gray", "marker": "."},
}


def _normalize_junctures_mode(mode):
    if mode is None or (isinstance(mode, float) and pd.isna(mode)):
        return "both"
    mode = str(mode)
    return mode if mode else "both"


def _junctures_mode_label(mode):
    mode = _normalize_junctures_mode(mode)
    if mode == "both":
        return "grow+prune"
    if mode == "both_gp":
        return "grow+prune combo"
    if mode == "prune":
        return "prune only"
    if mode == "grow":
        return "grow only"
    return mode


def _pareto_legend_label(kind, junctures_mode=None):
    if junctures_mode is None:
        return f"{kind} Pareto"
    mode = _normalize_junctures_mode(junctures_mode)
    if kind == "static_replay" or mode == "static_replay":
        return f"{kind} Pareto"
    if mode == "both":
        return f"{kind} Pareto"
    if mode == "prune":
        return f"{kind} prune only Pareto"
    if mode == "grow":
        return f"{kind} grow only Pareto"
    if mode == "both_gp":
        return f"{kind} grow+prune combo Pareto"
    return f"{kind} {_junctures_mode_label(mode)} Pareto"


def _junctures_mode_marker(mode):
    """Match FNN: grow ^, both_gp D, otherwise o (both and prune)."""
    mode = _normalize_junctures_mode(mode)
    if mode == "grow":
        return "^"
    if mode == "both_gp":
        return "D"
    return "o"


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


def _plasticity_point_color(row, style_map, lambda_colors):
    lam = row.get("lambda_penalty")
    if lam is not None and not pd.isna(lam) and lam in lambda_colors:
        return lambda_colors[lam]
    kind = row.get("kind")
    if kind == "static_replay":
        return style_map.get("static_replay", style_map["plasticity"])["color"]
    return style_map["plasticity"]["color"]


def _point_style_for_row(row, style_map, lambda_colors, style_by_junctures=True):
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
    if kind == "plasticity":
        mode = _normalize_junctures_mode(row.get("junctures_mode"))
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
    style = style_map.get(kind, style_map["other"])
    color = style["color"]
    return {
        "color": color,
        "marker": style["marker"],
        "facecolor": color,
        "edgecolor": "black",
        "linewidth": 0.4,
    }


def _aggregated_points(sub, x_col, y_col, aggregate_runs):
    if aggregate_runs:
        return list(
            zip(
                sub.groupby("experiment")[x_col].mean(),
                sub.groupby("experiment")[y_col].mean(),
            )
        )
    return list(zip(sub[x_col], sub[y_col]))


def _build_point_legend_handles(df, style_map, lambda_colors, style_by_junctures):
    handles = []
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
    plasticity_df = df.loc[df["kind"] == "plasticity"]
    if style_by_junctures and not plasticity_df.empty:
        pairs = plasticity_df[["lambda_penalty", "junctures_mode"]].drop_duplicates()
        for _, pair in pairs.sort_values(["lambda_penalty", "junctures_mode"]).iterrows():
            lam = pair["lambda_penalty"]
            mode = _normalize_junctures_mode(pair["junctures_mode"])
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
    show_pareto_frontier=False,
    pareto_scope="per_kind",
    pareto_frontier_kinds=("baseline", "plasticity"),
    pareto_y_goal=None,
    pareto_linestyle="--",
    pareto_linewidth=1.5,
    legend_mode="full",
    style_map=None,
    lambda_palette=None,
):
    """
    Collect experiment_summary.csv under save_path/run_*/{experiment}/ and plot
    parameter count vs test metric for CNN plasticity results.

    Color/marker conventions match BayesianFNN: black baseline, tab10/Set2 λ colors.

    pareto_scope:
      - "per_kind": one frontier per kind in pareto_frontier_kinds
      - "per_junctures_mode": plasticity-like kinds get a frontier per junctures_mode
    legend_mode:
      - "full": point styles (λ / junctures) plus Pareto lines
      - "pareto": only Pareto frontier legend entries
    """
    if pareto_scope not in ("per_kind", "per_junctures_mode"):
        raise ValueError(
            f"pareto_scope must be 'per_kind' or 'per_junctures_mode', got {pareto_scope!r}"
        )
    if legend_mode not in ("full", "pareto"):
        raise ValueError(f"legend_mode must be 'full' or 'pareto', got {legend_mode!r}")

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
            row["junctures_mode"] = _normalize_junctures_mode(parsed.get("junctures_mode"))
            row["conv_tag"] = parsed.get("conv_tag")
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No experiment_summary.csv found under {save_path}/run_*/"
        )

    df_all = pd.DataFrame(rows)
    if x_col not in df_all.columns or y_col not in df_all.columns:
        raise ValueError(f"Columns {x_col!r}/{y_col!r} missing from summaries")

    if style_map is None:
        style_map = dict(_DEFAULT_STYLE_MAP)
    lambda_colors = _lambda_penalty_color_map(df_all, palette=lambda_palette)
    plasticity_modes = (
        df_all.loc[df_all["kind"] == "plasticity", "junctures_mode"].dropna().unique()
    )
    style_by_junctures = len(plasticity_modes) > 1

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pareto_handles = []

    if aggregate_runs:
        group_cols = ["experiment", "kind", "junctures_mode", "lambda_penalty"]
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
            pt = _point_style_for_row(r, style_map, lambda_colors, style_by_junctures)
            ax.errorbar(
                r["x_mean"],
                r["y_mean"],
                xerr=0 if pd.isna(r["x_std"]) else r["x_std"],
                yerr=0 if pd.isna(r["y_std"]) else r["y_std"],
                fmt=pt["marker"],
                color=pt["edgecolor"],
                markerfacecolor=pt["facecolor"],
                markeredgecolor=pt["edgecolor"],
                markeredgewidth=pt["linewidth"],
                markersize=8,
                linestyle="none",
                capsize=3,
                alpha=0.95,
            )
    else:
        for _, r in df_all.iterrows():
            pt = _point_style_for_row(r, style_map, lambda_colors, style_by_junctures)
            ax.scatter(
                r[x_col],
                r[y_col],
                c=pt["facecolor"],
                edgecolors=pt["edgecolor"],
                linewidths=pt["linewidth"],
                marker=pt["marker"],
                s=marker_size,
                alpha=alpha_individual,
            )

    if show_pareto_frontier:
        if pareto_y_goal is None:
            pareto_y_goal = "minimize" if "brier" in y_col.lower() else "maximize"
        split_by_mode = pareto_scope == "per_junctures_mode"
        for kind in pareto_frontier_kinds:
            sub = df_all.loc[df_all["kind"] == kind]
            if sub.empty:
                continue
            color = style_map.get(kind, style_map["other"])["color"]
            if split_by_mode and kind in _PARETO_SPLIT_BY_JUNCTURES_KINDS:
                modes = sorted(sub["junctures_mode"].map(_normalize_junctures_mode).unique())
                for mode in modes:
                    mode_sub = sub.loc[
                        sub["junctures_mode"].map(_normalize_junctures_mode) == mode
                    ]
                    pts = _aggregated_points(mode_sub, x_col, y_col, aggregate_runs)
                    xs, ys = _pareto_frontier_xy(pts, y_goal=pareto_y_goal)
                    if not xs:
                        continue
                    ls = _DEFAULT_JUNCTURES_PARETO_LINESTYLES.get(mode, pareto_linestyle)
                    (line,) = ax.plot(
                        xs,
                        ys,
                        linestyle=ls,
                        linewidth=pareto_linewidth,
                        color=color,
                        label=_pareto_legend_label(kind, mode),
                    )
                    pareto_handles.append(line)
            else:
                pts = _aggregated_points(sub, x_col, y_col, aggregate_runs)
                xs, ys = _pareto_frontier_xy(pts, y_goal=pareto_y_goal)
                if xs:
                    (line,) = ax.plot(
                        xs,
                        ys,
                        linestyle=pareto_linestyle,
                        linewidth=pareto_linewidth,
                        color=color,
                        label=_pareto_legend_label(kind),
                    )
                    pareto_handles.append(line)

    if legend_mode == "pareto":
        if pareto_handles:
            ax.legend(handles=pareto_handles, fontsize=8)
    else:
        handles = _build_point_legend_handles(
            df_all, style_map, lambda_colors, style_by_junctures
        )
        handles.extend(pareto_handles)
        if handles:
            ax.legend(handles=handles, fontsize=8)

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
