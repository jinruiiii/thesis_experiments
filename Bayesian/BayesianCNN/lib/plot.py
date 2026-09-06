import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lib.flops import parse_conv_channels


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
    axis_label_fontsize=None,
    tick_label_fontsize=None,
    title_fontsize=None,
    legend_fontsize=None,
    ylim=None,
    yoffset_fontsize=None,
    checkpoint_tail_epochs=None,
):
    base = Path(save_path)
    rows = []
    checkpoint_rows = []

    def _legend_label_for_experiment(meta):
        kind = meta.get("kind")
        conv_tag = meta.get("conv_tag")
        lam = meta.get("lambda_penalty")
        mode = meta.get("junctures_mode")
        if kind == "plasticity" and conv_tag is not None and lam is not None:
            label = f"{conv_tag} λ={lam:g}"
            if mode not in (None, "both") and not (isinstance(mode, float) and pd.isna(mode)):
                label = f"{label} ({_junctures_mode_label(mode)})"
            return label
        return meta["label"]

    for experiment in experiments:
        meta = _parse_experiment_dir_name(experiment)
        legend_label = _legend_label_for_experiment(meta)
        for run_dir in sorted(base.glob("run_*")):
            exp_dir = run_dir / experiment
            metrics_path = exp_dir / "metrics.csv"
            summary_path = exp_dir / "experiment_summary.csv"
            if not metrics_path.exists() or not summary_path.exists():
                continue

            run_id = int(run_dir.name.split("_", 1)[1])
            summary = pd.read_csv(summary_path)
            metrics_df = pd.read_csv(metrics_path)

            best_epoch = None
            if "Selected epoch" in summary.columns:
                try:
                    best_epoch = int(summary.iloc[0]["Selected epoch"])
                except (ValueError, TypeError):
                    best_epoch = None

            max_epoch = None
            if checkpoint_tail_epochs is not None and best_epoch is not None:
                max_epoch = best_epoch + int(checkpoint_tail_epochs)

            for _, mrow in metrics_df.iterrows():
                epoch = int(mrow["epoch"])
                if max_epoch is not None and epoch > max_epoch:
                    continue
                rows.append(
                    {
                        "run": run_id,
                        "init_width": meta["init_width"],
                        "experiment": experiment,
                        "experiment_label": legend_label,
                        "junctures_mode": meta.get("junctures_mode"),
                        "epoch": epoch,
                        "param_count": float(mrow["param_count"]),
                    }
                )

            if best_epoch is None:
                continue

            best_rows = metrics_df.loc[metrics_df["epoch"] == best_epoch]
            if not best_rows.empty:
                checkpoint_rows.append(
                    {
                        "run": run_id,
                        "experiment": experiment,
                        "experiment_label": legend_label,
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
        for (_, experiment), sub in df.groupby(["run", "experiment"]):
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
    legend_kwargs = {
        "handles": color_handles,
        "title": "Experiment",
        "loc": "upper right",
    }
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
        legend_kwargs["title_fontsize"] = legend_fontsize
    ax.legend(**legend_kwargs)
    ax.set_xlabel("Epoch", fontsize=axis_label_fontsize)
    ax.set_ylabel("Parameter count", fontsize=axis_label_fontsize)
    if tick_label_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax.set_title("Parameter count vs Epoch", fontsize=title_fontsize)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if yoffset_fontsize is not None:
        ax.yaxis.get_offset_text().set_fontsize(yoffset_fontsize)

    if save_path_out is None:
        tag = "_".join(experiments[0].split("_")[1:3]) if len(experiments) == 1 else "multi"
        save_path_out = f"param_count_{tag}.png"
    fig.savefig(save_path_out, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, stats


_ACTION_FROM_CSV = {"grow": "grow", "none": "retain", "prune": "prune"}
_ACTION_COLORS = {
    "grow": "#2ca02c",
    "retain": "#7f7f7f",
    "prune": "#d62728",
}
_HEATMAP_NO_DECISION = 0
_HEATMAP_GROW = 1
_HEATMAP_RETAIN = 2
_HEATMAP_PRUNE = 3
_HEATMAP_AFTER_CHECKPOINT = 4
_HEATMAP_ACTION_TO_CODE = {
    "grow": _HEATMAP_GROW,
    "retain": _HEATMAP_RETAIN,
    "prune": _HEATMAP_PRUNE,
}
_HEATMAP_COLORS = [
    "#eeeeee",
    _ACTION_COLORS["grow"],
    _ACTION_COLORS["retain"],
    _ACTION_COLORS["prune"],
    "#000000",
]
_HEATMAP_LEGEND = (
    ("grow", _ACTION_COLORS["grow"]),
    ("retain", _ACTION_COLORS["retain"]),
    ("prune", _ACTION_COLORS["prune"]),
    ("after checkpoint", "#000000"),
)


def plot_structural_decision_heatmap(
    save_path="results",
    experiments=None,
    save_path_out=None,
    show=True,
    figsize=None,
    dpi=150,
    axis_label_fontsize=None,
    tick_label_fontsize=None,
    title_fontsize=None,
    legend_fontsize=None,
    ylabels="lambda_blocks",
):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if not experiments:
        raise ValueError("experiments must be a non-empty list of experiment dir names")
    if ylabels not in ("lambda_blocks", "lambda_blocks_runs", "full"):
        raise ValueError(
            "ylabels must be 'lambda_blocks', 'lambda_blocks_runs', or 'full', "
            f"got {ylabels!r}"
        )

    base = Path(save_path)
    row_records = []
    t_max = 0
    phase1_epochs = None

    for experiment in experiments:
        meta = _parse_experiment_dir_name(experiment)
        for run_dir in sorted(base.glob("run_*")):
            exp_dir = run_dir / experiment
            decisions_path = exp_dir / "structural_decisions.csv"
            summary_path = exp_dir / "experiment_summary.csv"
            if not decisions_path.exists() or not summary_path.exists():
                continue

            run_id = int(run_dir.name.split("_", 1)[1])
            summary_row = pd.read_csv(summary_path).iloc[0]
            if "Selected epoch" in summary_row.index and pd.notna(summary_row["Selected epoch"]):
                selected_epoch = int(summary_row["Selected epoch"])
            elif "Best Phase 2 Epoch" in summary_row.index and pd.notna(
                summary_row["Best Phase 2 Epoch"]
            ):
                selected_epoch = int(summary_row["Best Phase 2 Epoch"])
            else:
                continue

            if phase1_epochs is None and "Phase 1 Epochs" in summary_row.index and pd.notna(
                summary_row["Phase 1 Epochs"]
            ):
                phase1_epochs = int(summary_row["Phase 1 Epochs"])

            decisions = pd.read_csv(decisions_path)
            if "epoch" not in decisions.columns or "action" not in decisions.columns:
                continue

            run_max_epoch = int(decisions["epoch"].max()) if len(decisions) else 0
            metrics_path = exp_dir / "metrics.csv"
            if metrics_path.exists():
                metrics_df = pd.read_csv(metrics_path)
                if "epoch" in metrics_df.columns and len(metrics_df):
                    run_max_epoch = max(run_max_epoch, int(metrics_df["epoch"].max()))
                if (
                    phase1_epochs is None
                    and "phase1_epochs" in metrics_df.columns
                    and len(metrics_df)
                ):
                    phase1_epochs = int(metrics_df["phase1_epochs"].iloc[0])
            t_max = max(t_max, run_max_epoch)

            action_by_epoch = {}
            for _, drow in decisions.iterrows():
                mapped = _ACTION_FROM_CSV.get(str(drow["action"]))
                if mapped is None:
                    continue
                action_by_epoch[int(drow["epoch"])] = mapped

            lam = meta.get("lambda_penalty")
            if lam is None or (isinstance(lam, float) and np.isnan(lam)):
                row_label = f"{meta.get('label', experiment)}, run {run_id}"
            else:
                row_label = f"λ={lam:g}, run {run_id}"

            row_records.append(
                {
                    "experiment": experiment,
                    "run": run_id,
                    "lambda_penalty": lam if lam is not None else np.nan,
                    "init_width": meta.get("init_width"),
                    "selected_epoch": selected_epoch,
                    "run_max_epoch": run_max_epoch,
                    "action_by_epoch": action_by_epoch,
                    "row_label": row_label,
                }
            )

    if not row_records or t_max < 1:
        raise FileNotFoundError(
            f"No structural_decisions.csv (with Selected epoch or Best Phase 2 Epoch) "
            f"found under {save_path} for experiments={experiments}"
        )

    row_meta = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "action_by_epoch"} for r in row_records]
    )
    for i, rec in enumerate(row_records):
        row_meta.loc[i, "_sort_idx"] = i

    sort_cols = ["lambda_penalty", "run", "experiment"]
    row_meta = row_meta.sort_values(
        sort_cols, ascending=[True, True, True], kind="mergesort", na_position="last"
    ).reset_index(drop=True)

    action_maps = []
    for _, row in row_meta.iterrows():
        orig = row_records[int(row["_sort_idx"])]
        action_maps.append(orig["action_by_epoch"])
    row_meta = row_meta.drop(columns=["_sort_idx"])

    n_rows = len(row_meta)
    H = np.full((n_rows, t_max), _HEATMAP_NO_DECISION, dtype=np.int32)
    tidy_rows = []

    for r_idx, ((_, row), action_by_epoch) in enumerate(
        zip(row_meta.iterrows(), action_maps)
    ):
        selected_epoch = int(row["selected_epoch"])
        for epoch in range(1, t_max + 1):
            if epoch > selected_epoch:
                code = _HEATMAP_AFTER_CHECKPOINT
                status = "after_checkpoint"
            elif epoch in action_by_epoch:
                action = action_by_epoch[epoch]
                code = _HEATMAP_ACTION_TO_CODE[action]
                status = action
            else:
                code = _HEATMAP_NO_DECISION
                status = "no_decision"
            H[r_idx, epoch - 1] = code
            tidy_rows.append(
                {
                    "experiment": row["experiment"],
                    "run": int(row["run"]),
                    "lambda_penalty": row["lambda_penalty"],
                    "epoch": epoch,
                    "selected_epoch": selected_epoch,
                    "status": status,
                }
            )

    if figsize is None:
        figsize = (14, max(4.0, 0.35 * n_rows))

    cmap = ListedColormap(_HEATMAP_COLORS)
    bounds = np.arange(-0.5, len(_HEATMAP_COLORS) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        H,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=(0.5, t_max + 0.5, n_rows - 0.5, -0.5),
    )

    if phase1_epochs is not None and 1 <= phase1_epochs < t_max:
        ax.axvline(
            phase1_epochs + 0.5,
            color="black",
            linestyle="--",
            linewidth=1.2,
            zorder=4,
        )

    ax.set_xlabel("Epoch", fontsize=axis_label_fontsize)
    ax.set_ylabel("Run (grouped by λ)", fontsize=axis_label_fontsize)
    ytick_fs = 8 if tick_label_fontsize is None else tick_label_fontsize

    def _same_lam(a, b):
        a_nan = isinstance(a, float) and np.isnan(a)
        b_nan = isinstance(b, float) and np.isnan(b)
        if a_nan or b_nan:
            return a_nan and b_nan
        return a == b

    def _lam_label(lam):
        if isinstance(lam, float) and np.isnan(lam):
            return "?"
        return f"λ={lam:g}"

    lam_values = row_meta["lambda_penalty"].tolist()
    blocks = []
    start = 0
    for i in range(1, n_rows + 1):
        if i == n_rows or not _same_lam(lam_values[i], lam_values[start]):
            blocks.append((start, i - 1, lam_values[start]))
            start = i

    for start_i, _end_i, _lam in blocks[1:]:
        ax.axhline(start_i - 0.5, color="white", linewidth=1.5, zorder=3)

    if ylabels == "full":
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_meta["row_label"].tolist(), fontsize=ytick_fs)
    else:
        centers = [0.5 * (s + e) for s, e, _ in blocks]
        lam_tick_labels = [_lam_label(lam) for _, _, lam in blocks]
        ax.set_yticks(centers)
        ax.set_yticklabels(lam_tick_labels, fontsize=ytick_fs)
        if ylabels == "lambda_blocks_runs":
            run_positions = []
            run_labels = []
            for s, e, _lam in blocks:
                for k, row_i in enumerate(range(s, e + 1)):
                    run_positions.append(row_i)
                    run_labels.append(str(k + 1))
            ax.set_yticks(run_positions, minor=True)
            ax.set_yticklabels(run_labels, minor=True, fontsize=max(6, ytick_fs - 2))
            ax.tick_params(axis="y", which="minor", length=2, pad=2)
            ax.tick_params(axis="y", which="major", pad=12)

    if t_max <= 40:
        xticks = list(range(1, t_max + 1))
    else:
        step = max(1, int(np.ceil(t_max / 20)))
        xticks = list(range(1, t_max + 1, step))
        if xticks[-1] != t_max:
            xticks.append(t_max)
    ax.set_xticks(xticks)
    if tick_label_fontsize is not None:
        ax.tick_params(axis="x", labelsize=tick_label_fontsize)
    ax.set_xlim(0.5, t_max + 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_title("Structural decisions by epoch", fontsize=title_fontsize)

    legend_fs = 8 if legend_fontsize is None else legend_fontsize
    ax.legend(
        handles=[
            Patch(facecolor=color, edgecolor="0.3", label=label)
            for label, color in _HEATMAP_LEGEND
        ],
        title="Decision",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=legend_fs,
        title_fontsize=legend_fs,
    )
    fig.tight_layout()

    if save_path_out is None:
        save_path_out = "structural_decision_heatmap.png"
    fig.savefig(save_path_out, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    matrix_df = pd.DataFrame(tidy_rows)
    return fig, ax, matrix_df


def _init_width_from_conv_tag(conv_tag):
    if conv_tag is None:
        return None
    m = re.match(r"^(\d+)f", str(conv_tag))
    return int(m.group(1)) if m else None


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


_OPTIONAL_EXPERIMENT_DIR_TAGS = ("_vcl", "_p2junct", "_regrow", "_random_grow")


def _strip_optional_experiment_dir_tags(name):
    """
    Strip trailing shift/plasticity tags (_vcl, _p2junct, _regrow, _random_grow) in any order.
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
    Parse CNN experiment directory names, e.g.:
      baseline_20f_20f
      plasticity_20f_20f_5e-06
      plasticity_20f_20f_5e-06_prune_only
      plasticity_20f_20f_5e-06_random_grow
      static_replay_20f_20f_5e-06
      three_phase_baseline_20f_20f
      baseline_300f_300f_vcl
    Returns dict with keys: kind, conv_tag, init_width, lambda_penalty, junctures_mode, label, dir_tags
    """
    raw_name = os.path.basename(str(dirname).rstrip("/"))
    name, dir_tags = _strip_optional_experiment_dir_tags(raw_name)

    def _finalize(meta):
        meta["dir_tags"] = list(dir_tags)
        if meta.get("kind") == "plasticity" and "random_grow" in meta["dir_tags"]:
            meta["kind"] = "plasticity_random"
        return meta

    m = re.match(
        r"^plasticity_((?:\d+f_)*\d+f)_([0-9.e+-]+)_(prune|grow|both_gp)_only$",
        name,
    )
    if m:
        conv_tag = m.group(1)
        lam = float(m.group(2))
        mode = m.group(3)
        mode_label = _junctures_mode_label(mode)
        return _finalize({
            "kind": "plasticity",
            "conv_tag": conv_tag,
            "init_width": _init_width_from_conv_tag(conv_tag),
            "lambda_penalty": lam,
            "junctures_mode": mode,
            "label": f"plasticity {conv_tag} λ={lam:g} ({mode_label})",
        })

    m = re.match(r"^plasticity_((?:\d+f_)*\d+f)_([0-9.e+-]+)$", name)
    if m:
        conv_tag = m.group(1)
        lam = float(m.group(2))
        return _finalize({
            "kind": "plasticity",
            "conv_tag": conv_tag,
            "init_width": _init_width_from_conv_tag(conv_tag),
            "lambda_penalty": lam,
            "junctures_mode": "both",
            "label": f"plasticity {conv_tag} λ={lam:g}",
        })

    m = re.match(
        r"^static_replay_((?:\d+f_)*\d+f)_([0-9.e+-]+)_(prune|grow|both_gp)_only$",
        name,
    )
    if m:
        conv_tag = m.group(1)
        lam = float(m.group(2))
        mode = m.group(3)
        mode_label = _junctures_mode_label(mode)
        return _finalize({
            "kind": "static_replay",
            "conv_tag": conv_tag,
            "init_width": _init_width_from_conv_tag(conv_tag),
            "lambda_penalty": lam,
            "junctures_mode": "static_replay",
            "label": f"static replay {conv_tag} λ={lam:g} ({mode_label})",
        })

    m = re.match(r"^static_replay_((?:\d+f_)*\d+f)_([0-9.e+-]+)$", name)
    if m:
        conv_tag = m.group(1)
        lam = float(m.group(2))
        return _finalize({
            "kind": "static_replay",
            "conv_tag": conv_tag,
            "init_width": _init_width_from_conv_tag(conv_tag),
            "lambda_penalty": lam,
            "junctures_mode": "static_replay",
            "label": f"static replay {conv_tag} λ={lam:g}",
        })

    m = re.match(r"^baseline_((?:\d+f_)*\d+f)$", name)
    if m:
        conv_tag = m.group(1)
        return _finalize({
            "kind": "baseline",
            "conv_tag": conv_tag,
            "init_width": _init_width_from_conv_tag(conv_tag),
            "lambda_penalty": None,
            "junctures_mode": "both",
            "label": f"baseline (init {conv_tag})",
        })

    m = re.match(r"^three_phase_baseline_((?:\d+f_)*\d+f)$", name)
    if m:
        conv_tag = m.group(1)
        return _finalize({
            "kind": "three_phase",
            "conv_tag": conv_tag,
            "init_width": _init_width_from_conv_tag(conv_tag),
            "lambda_penalty": None,
            "junctures_mode": "three_phase",
            "label": f"three-phase baseline (init {conv_tag})",
        })

    m = re.match(
        r"^nest_((?:\d+f_)*\d+f)_p([0-9.e+-]+)_refacc([0-9.e+-]+)_flooracc([0-9.e+-]+)$",
        name,
    )
    if m:
        conv_tag = m.group(1)
        nest_p = float(m.group(2))
        nest_ref_acc = float(m.group(3))
        nest_floor_acc = float(m.group(4))
        conv_display = conv_tag.replace("_", "/")
        return _finalize({
            "kind": "nest",
            "conv_tag": conv_tag,
            "init_width": _init_width_from_conv_tag(conv_tag),
            "lambda_penalty": None,
            "junctures_mode": "nest",
            "nest_p": nest_p,
            "nest_ref_acc": nest_ref_acc,
            "nest_floor_acc": nest_floor_acc,
            "label": f"nest {conv_display} p={nest_p:g} floor={nest_floor_acc:g}",
        })

    return _finalize({
        "kind": "unknown",
        "conv_tag": None,
        "init_width": None,
        "lambda_penalty": None,
        "junctures_mode": None,
        "label": name,
    })


def _resolve_summary_x_value(row0, x_col):
    """Map CSV columns onto the plot x value."""
    def _has(col):
        return col in row0.index and not pd.isna(row0[col])

    if x_col in ("FLOPs", "FLOPs_theoretical", "FLOPs_realized"):
        if not _has("FLOPs"):
            raise KeyError(
                "experiment_summary.csv is missing 'FLOPs'. "
                "Re-run the experiment with an updated summary writer."
            )
        return float(row0["FLOPs"])

    if x_col == "Parameters":
        if _has("Sparse Parameters"):
            return float(row0["Sparse Parameters"])
        return float(row0["Parameters"])
    if x_col == "Sparse Parameters":
        if _has("Sparse Parameters"):
            return float(row0["Sparse Parameters"])
        return float(row0["Parameters"])
    if x_col == "Dense Parameters":
        if _has("Dense Parameters"):
            return float(row0["Dense Parameters"])
        return float(row0["Parameters"])
    return float(row0[x_col])


def _csv_params_sparse_dense(row0):
    """Return (params_sparse, params_dense) from a summary row."""
    def _has(col):
        return col in row0.index and not pd.isna(row0[col])

    if _has("Parameters"):
        fallback = float(row0["Parameters"])
    elif _has("Sparse Parameters"):
        fallback = float(row0["Sparse Parameters"])
    elif _has("Dense Parameters"):
        fallback = float(row0["Dense Parameters"])
    else:
        fallback = float("nan")

    sparse = float(row0["Sparse Parameters"]) if _has("Sparse Parameters") else fallback
    dense = float(row0["Dense Parameters"]) if _has("Dense Parameters") else fallback
    return sparse, dense


def collect_experiment_summaries(
    save_path="results",
    experiments=None,
    experiment_glob="*",
    num_runs=None,
    x_col="Parameters",
    y_col="Test Acc",
):
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
            if meta.get("kind") != "nest":
                if csv_junctures is not None and not pd.isna(csv_junctures):
                    junctures_mode = str(csv_junctures)

            conv_raw = row0.get("Conv Channels")
            if conv_raw is None or (isinstance(conv_raw, float) and pd.isna(conv_raw)):
                raise ValueError(f"'Conv Channels' missing in {summary_path}")
            conv_channels = parse_conv_channels(conv_raw)
            params_sparse, params_dense = _csv_params_sparse_dense(row0)
            flops_csv = None
            if "FLOPs" in row0.index and not pd.isna(row0["FLOPs"]):
                flops_csv = float(row0["FLOPs"])

            rows.append(
                {
                    "run": run_id,
                    "experiment": exp_dir.name,
                    "kind": meta["kind"],
                    "init_width": meta["init_width"],
                    "conv_tag": meta.get("conv_tag"),
                    "conv_channels": conv_channels,
                    "lambda_penalty": meta["lambda_penalty"],
                    "junctures_mode": junctures_mode,
                    "nest_p": meta.get("nest_p"),
                    "nest_ref_acc": meta.get("nest_ref_acc"),
                    "nest_floor_acc": meta.get("nest_floor_acc"),
                    "model_label": str(row0.get("Model", meta["kind"])),
                    "params_sparse": params_sparse,
                    "params_dense": params_dense,
                    "FLOPs": flops_csv,
                    "x": _resolve_summary_x_value(row0, x_col),
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


_PARETO_SPLIT_BY_INIT_WIDTH_KINDS = frozenset(
    {"plasticity", "plasticity_random", "static_replay"}
)
_PARETO_SPLIT_BY_JUNCTURES_KINDS = frozenset(
    {"plasticity", "plasticity_random", "static_replay"}
)
_DEFAULT_JUNCTURES_PARETO_LINESTYLES = {
    "both": "-",
    "prune": "--",
    "grow": ":",
    "both_gp": "-.",
}
_PARETO_PLASTICITY_PRUNE_COLOR = "#ff7f0e"

_DEFAULT_STYLE_MAP = {
    "baseline": {"color": "black", "marker": "s"},
    "three_phase": {"color": "#1f77b4", "marker": "D"},
    "plasticity": {"color": "#d62728", "marker": "o"},
    "plasticity_random": {"color": "#17becf", "marker": "^"},
    "static_replay": {"color": "#2ca02c", "marker": "X"},
    "nest": {"color": "#d946ef", "marker": "v"},
    "nest_dense": {"color": "#d946ef", "marker": "v"},
    "other": {"color": "gray", "marker": "x"},
}


def _normalize_junctures_mode_for_pareto(mode):
    if mode is None or (isinstance(mode, float) and pd.isna(mode)):
        return "both"
    mode = str(mode)
    return mode if mode else "both"


def _pareto_color(kind, style_map, junctures_mode=None):
    mode = _normalize_junctures_mode_for_pareto(junctures_mode)
    if kind in ("plasticity", "plasticity_three_phase") and mode == "prune":
        return _PARETO_PLASTICITY_PRUNE_COLOR
    return style_map.get(kind, style_map["other"])["color"]


def _kind_display_name_for_pareto(kind):
    return {
        "baseline": "Symmetric Baseline",
        "static_replay": "Static Replay",
        "three_phase": "Three-Phase",
        "plasticity": "Plasticity",
        "plasticity_random": "Random Growth Plasticity",
        "nest": "NeST Sparse",
        "nest_dense": "NeST Dense",
    }.get(kind, kind)


def _pareto_legend_label(kind, junctures_mode=None, init_width=None):
    if kind == "plasticity_random":
        return "Random Growth Plasticity"

    name = _kind_display_name_for_pareto(kind)
    if init_width is not None and not (isinstance(init_width, float) and pd.isna(init_width)):
        return f"{name} init={int(init_width)} Pareto"

    if junctures_mode is None:
        return f"{name} Pareto"

    mode = _normalize_junctures_mode_for_pareto(junctures_mode)
    if kind == "static_replay" or mode == "static_replay":
        return f"{name} Pareto"
    if kind in ("nest", "nest_dense") or mode == "nest":
        return f"{name} Pareto"
    if mode == "both":
        return f"{name} Pareto"
    if mode == "prune":
        return f"{name} Prune-Only Pareto"
    if mode == "grow":
        return f"{name} Grow-Only Pareto"
    if mode == "both_gp":
        return f"{name} Grow+Prune Combo Pareto"
    pretty = _junctures_mode_label(mode) or mode
    return f"{name} {pretty} Pareto"


def _junctures_mode_marker(mode):
    mode = _normalize_junctures_mode(mode)
    if mode == "grow":
        return "^"
    if mode == "both_gp":
        return "D"
    return "o"


def _lambda_penalty_color_map(df, palette=None):
    lambdas = sorted(
        df.loc[
            df["kind"].isin(["plasticity", "plasticity_random", "static_replay"]),
            "lambda_penalty",
        ]
        .dropna()
        .unique()
    )
    if palette is None:
        palette = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)
    return {lam: palette[i % len(palette)] for i, lam in enumerate(lambdas)}


def _normalize_nest_x_modes(nest_x_modes):
    if nest_x_modes is None:
        modes = ("sparse",)
    elif isinstance(nest_x_modes, str):
        modes = (nest_x_modes,)
    else:
        modes = tuple(nest_x_modes)
    allowed = {"sparse", "dense"}
    if not modes or any(m not in allowed for m in modes):
        raise ValueError(
            "nest_x_modes must be a non-empty subset of ('sparse', 'dense'), "
            f"got {nest_x_modes!r}"
        )
    seen = set()
    ordered = []
    for m in modes:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return tuple(ordered)


def _expand_nest_x_modes(df, nest_x_modes, x_col):
    modes = _normalize_nest_x_modes(nest_x_modes)
    if x_col not in ("Parameters", "Sparse Parameters", "Dense Parameters"):
        return df
    nest = df.loc[df["kind"] == "nest"].copy()
    if nest.empty:
        return df

    others = df.loc[df["kind"] != "nest"].copy()
    parts = [others] if not others.empty else []

    if "sparse" in modes:
        sparse = nest.copy()
        sparse["x"] = sparse["params_sparse"]
        sparse["kind"] = "nest"
        parts.append(sparse)
    if "dense" in modes:
        dense = nest.copy()
        dense["x"] = dense["params_dense"]
        dense["kind"] = "nest_dense"
        dense["experiment"] = dense["experiment"].astype(str) + "__dense"
        parts.append(dense)

    if not parts:
        return df
    return pd.concat(parts, ignore_index=True)


def _nest_floor_color_map(df, palette=None):
    floors = sorted(
        df.loc[df["kind"].isin(["nest", "nest_dense"]), "nest_floor_acc"]
        .dropna()
        .unique()
    )
    if palette is None:
        palette = [
            "#d946ef",
            "#c084fc",
            "#f472b6",
            "#e879f9",
            "#a855f7",
            "#ec4899",
        ]
    return {floor: palette[i % len(palette)] for i, floor in enumerate(floors)}


def _plasticity_point_color(row, style_map, lambda_colors):
    lam = row.get("lambda_penalty")
    if lam is not None and not pd.isna(lam) and lam in lambda_colors:
        return lambda_colors[lam]
    kind = row.get("kind")
    if kind == "static_replay":
        return style_map.get("static_replay", style_map["plasticity"])["color"]
    return style_map["plasticity"]["color"]


def _point_style(row, style_map, lambda_colors, style_by_junctures=False, nest_floor_colors=None):
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
    if kind == "nest":
        style = style_map.get("nest", {"color": "#d946ef", "marker": "v"})
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
    if kind == "nest_dense":
        style = style_map.get("nest_dense", {"color": "#d946ef", "marker": "v"})
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
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.0,
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
    style = style_map.get(kind, style_map["other"])
    color = style["color"]
    return {
        "color": color,
        "marker": style["marker"],
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


def _pareto_mean_point_style(row, style_map):
    color = _pareto_color(row["kind"], style_map, row.get("junctures_mode"))
    return {
        "marker": "s",
        "facecolor": color,
        "edgecolor": "black",
        "linewidth": 0.4,
        "color": color,
    }


def _infer_pareto_y_goal(y_col, ylabel=None):
    text = f"{y_col} {ylabel or ''}".lower()
    if any(k in text for k in ("brier", "loss", "error", "nll", "mse", "rmse")):
        return "minimize"
    return "maximize"


def _pareto_frontier_xy(points, y_goal="maximize"):
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


def _init_width_pareto_colors(widths):
    palette = [
        "#d62728",
        "#1f77b4",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]
    widths_sorted = sorted(
        {
            int(w)
            for w in widths
            if w is not None and not (isinstance(w, float) and pd.isna(w))
        }
    )
    return {w: palette[i % len(palette)] for i, w in enumerate(widths_sorted)}


def _resolve_pareto_linestyle(pareto_linestyle, kind, init_width=None, junctures_mode=None):
    if not isinstance(pareto_linestyle, dict):
        if kind == "nest_dense" and pareto_linestyle == "--":
            return ":"
        if junctures_mode is not None and pareto_linestyle == "--":
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
    style_map=None,
):
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
                label=_pareto_legend_label(kind),
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
            label = _pareto_legend_label(kind, junctures_mode=mode_label)
            ls = _resolve_pareto_linestyle(linestyle, kind, junctures_mode=mode_label)
            line_color = (
                _pareto_color(kind, style_map, mode_label)
                if style_map is not None
                else color
            )
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

    if split_by_init_width and kind in _PARETO_SPLIT_BY_INIT_WIDTH_KINDS:
        if "init_width" not in subset.columns:
            handle = _draw_pareto_frontier(
                ax,
                subset,
                color=color,
                label=_pareto_legend_label(kind),
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
                label = _pareto_legend_label(kind)
                line_color = color
            else:
                width_int = int(width)
                label = _pareto_legend_label(kind, init_width=width_int)
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
        label=_pareto_legend_label(kind),
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
):
    handles = []
    if nest_floor_colors is None:
        nest_floor_colors = {}
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
        nest_style = style_map.get("nest", {"color": "#d946ef", "marker": "v"})
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
    nest_dense_df = df.loc[df["kind"] == "nest_dense"]
    if not nest_dense_df.empty:
        nest_dense_style = style_map.get("nest_dense", {"color": "#d946ef", "marker": "v"})
        floors = sorted(nest_dense_df["nest_floor_acc"].dropna().unique())
        if floors:
            for floor in floors:
                color = nest_floor_colors.get(floor, nest_dense_style["color"])
                handles.append(
                    Line2D(
                        [0], [0],
                        marker=nest_dense_style["marker"],
                        color="w",
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markeredgewidth=1.0,
                        markersize=8,
                        label=f"nest dense floor={floor:g}",
                    )
                )
        else:
            handles.append(
                Line2D(
                    [0], [0],
                    marker=nest_dense_style["marker"],
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor=nest_dense_style["color"],
                    markeredgewidth=1.0,
                    markersize=8,
                    label="nest dense",
                )
            )
    if (df["kind"] == "other").any() or (df["kind"] == "unknown").any():
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
    group_by="experiment",
    x_col="Parameters",
    y_col="Test Acc",
    xlabel=None,
    ylabel="Test accuracy (%)",
    title="Parameter count vs test accuracy",
    save_path_out=None,
    show=True,
    figsize=(10, 7),
    dpi=150,
    alpha_individual=0.35,
    marker_size=70,
    annotate_points=False,
    style_map=None,
    lambda_palette=None,
    style_by="junctures_mode",
    show_pareto_frontier=False,
    show_points=True,
    pareto_scope="per_kind",
    pareto_y_goal=None,
    pareto_frontier_kinds=("baseline", "plasticity"),
    pareto_linestyle="--",
    pareto_linewidth=1.5,
    legend_mode="full",
    axis_label_fontsize=None,
    tick_label_fontsize=None,
    title_fontsize=None,
    legend_fontsize=None,
    nest_x_modes=("sparse",),
    x_tick_interval=None,
):
    """
    Plot parameter count (x) against test metric (y) across CNN experiments.
    Styling matches BayesianFNN: black baseline, blue diamond three-phase, tab10 λ colors.

    show_points : bool
        If False, skip error bars and per-run scatter; when ``aggregate_runs=True``,
        still plot one mean marker per experiment configuration. Requires
        ``show_pareto_frontier=True`` (or mean markers) for a non-empty figure.
    """
    if legend_mode not in ("full", "pareto"):
        raise ValueError(f"legend_mode must be 'full' or 'pareto', got {legend_mode!r}")
    nest_x_modes = _normalize_nest_x_modes(nest_x_modes)
    if pareto_scope not in ("global", "per_kind", "per_init_width", "per_junctures_mode"):
        raise ValueError(
            f"pareto_scope must be 'global', 'per_kind', 'per_init_width', or "
            f"'per_junctures_mode', got {pareto_scope!r}"
        )

    if xlabel is None:
        if x_col in ("FLOPs", "FLOPs_realized", "FLOPs_theoretical"):
            xlabel = "FLOPs"
        elif x_col == "Sparse Parameters":
            xlabel = "Sparse parameter count"
        elif x_col == "Dense Parameters":
            xlabel = "Dense parameter count"
        else:
            xlabel = "Parameter count"

    df = collect_experiment_summaries(
        save_path=save_path,
        experiments=experiments,
        experiment_glob=experiment_glob,
        num_runs=num_runs,
        x_col=x_col,
        y_col=y_col,
    )
    df = _expand_nest_x_modes(df, nest_x_modes=nest_x_modes, x_col=x_col)

    if style_map is None:
        style_map = dict(_DEFAULT_STYLE_MAP)
    lambda_colors = _lambda_penalty_color_map(df, palette=lambda_palette)
    nest_floor_colors = _nest_floor_color_map(df, palette=lambda_palette)
    plasticity_modes = df.loc[df["kind"] == "plasticity", "junctures_mode"].dropna().unique()
    style_by_junctures = style_by == "junctures_mode" and len(plasticity_modes) > 1

    fig, ax = plt.subplots(figsize=figsize)
    pareto_handles = []
    grouped = None

    if aggregate_runs:
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
                n_runs=("run", "nunique"),
            )
            .reset_index()
        )
        if show_points:
            for _, row in grouped.iterrows():
                style = _point_style(
                    row, style_map, lambda_colors, style_by_junctures, nest_floor_colors
                )
                ax.errorbar(
                    row["x_mean"],
                    row["y_mean"],
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
                    row, style_map, lambda_colors, style_by_junctures, nest_floor_colors
                )
                _scatter_point(
                    ax, row["x"], row["y"], style, marker_size * 0.5, alpha=alpha_individual
                )
        else:
            for _, row in grouped.iterrows():
                style = _pareto_mean_point_style(row, style_map)
                _scatter_point(ax, row["x_mean"], row["y_mean"], style, marker_size)
                if annotate_points:
                    ax.annotate(
                        str(row[group_by]),
                        (row["x_mean"], row["y_mean"]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                    )
    elif show_points:
        for _, row in df.iterrows():
            style = _point_style(
                row, style_map, lambda_colors, style_by_junctures, nest_floor_colors
            )
            _scatter_point(ax, row["x"], row["y"], style, marker_size)

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
            "plasticity": style_map["plasticity"]["color"],
            "static_replay": style_map.get("static_replay", {"color": "#2ca02c"})["color"],
            "nest": style_map.get("nest", {"color": "#d946ef"})["color"],
            "nest_dense": style_map.get("nest_dense", {"color": "#d946ef"})["color"],
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
                        style_map=style_map,
                    )
                )

    if legend_mode == "full" and (show_points or aggregate_runs):
        handles = _legend_handles_for_test_acc_plot(
            df, style_map, lambda_colors, style_by_junctures, nest_floor_colors
        )
        handles.extend(pareto_handles)
        legend_title = (
            "Model / λ / junctures" if style_by_junctures else "Model / λ"
        )
    else:
        handles = list(pareto_handles)
        legend_title = "Pareto"

    if handles:
        legend_kwargs = {"handles": handles, "title": legend_title}
        if legend_fontsize is not None:
            legend_kwargs["fontsize"] = legend_fontsize
            legend_kwargs["title_fontsize"] = legend_fontsize
        ax.legend(**legend_kwargs)

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    if x_tick_interval is not None:
        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(float(x_tick_interval)))
    if tick_label_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path_out is not None:
        fig.savefig(save_path_out, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, df


def _parse_gamma_label_from_save_path(save_path):
    """Extract a display label from a results root like ..._gamma0.01."""
    name = Path(save_path).name
    match = re.search(r"gamma([0-9.e-]+)", name, re.I)
    if match:
        return f"γ={match.group(1)}"
    return name


def _default_gamma_color_map(gamma_labels, gamma_colors=None):
    """Assign a distinct color to each gamma / growth-mode label."""
    if gamma_colors is not None:
        return dict(gamma_colors)
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(gamma_labels)}


def plot_gamma_pareto_comparison(
    save_paths,
    gamma_labels=None,
    experiments=None,
    experiment_glob="*",
    num_runs=5,
    aggregate_runs=True,
    x_col="Parameters",
    y_col="Test Acc",
    xlabel=None,
    ylabel=None,
    title=None,
    save_path_out=None,
    show=True,
    figsize=(10, 7),
    dpi=150,
    pareto_frontier_kinds=("plasticity",),
    show_points=True,
    gamma_colors=None,
    pareto_linestyle="--",
    pareto_linewidth=1.5,
    pareto_y_goal=None,
    alpha_individual=0.35,
    marker_size=70,
    legend_mode="pareto",
    axis_label_fontsize=None,
    tick_label_fontsize=None,
    title_fontsize=None,
    legend_fontsize=None,
    x_tick_interval=None,
):
    """
    Compare plasticity Pareto frontiers across result roots (e.g. growth modes).

    Loads the same leaf experiment names from each parent ``save_path``, tags
    rows with ``gamma_label``, then plots Acc/Brier vs parameter count.

    Typical usage
    -------------
    plot_gamma_pareto_comparison(
        save_paths=["results_cifar10", "results_cifar10_random_growth"],
        gamma_labels=["Uncertainty growth", "Random growth"],
        experiments=["plasticity_300f_300f_1e-07", ...],
        y_col="Test Acc",
        save_path_out="cifar10_plots/random_growth/acc.pdf",
        show=False,
    )
    """
    if not save_paths:
        raise ValueError("save_paths must be a non-empty list of result roots")
    if legend_mode not in ("full", "pareto"):
        raise ValueError(
            f"legend_mode must be 'full' or 'pareto', got {legend_mode!r}"
        )

    save_paths = [Path(p) for p in save_paths]
    if gamma_labels is None:
        gamma_labels = [_parse_gamma_label_from_save_path(p) for p in save_paths]
    if len(gamma_labels) != len(save_paths):
        raise ValueError(
            f"gamma_labels length ({len(gamma_labels)}) must match "
            f"save_paths ({len(save_paths)})"
        )

    frames = []
    for save_path, gamma_label in zip(save_paths, gamma_labels):
        frame = collect_experiment_summaries(
            save_path=str(save_path),
            experiments=experiments,
            experiment_glob=experiment_glob,
            num_runs=num_runs,
            x_col=x_col,
            y_col=y_col,
        )
        frame = frame.copy()
        frame["gamma_label"] = gamma_label
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    if xlabel is None:
        if x_col in ("FLOPs", "FLOPs_realized", "FLOPs_theoretical"):
            xlabel = "FLOPs"
        elif x_col == "Sparse Parameters":
            xlabel = "Sparse parameter count"
        elif x_col == "Dense Parameters":
            xlabel = "Dense parameter count"
        else:
            xlabel = "Parameter count"
    if ylabel is None:
        ylabel = y_col
    if title is None:
        title = f"{ylabel} vs parameter count by growth rate"

    resolved_y_goal = pareto_y_goal or _infer_pareto_y_goal(y_col, ylabel)
    color_map = _default_gamma_color_map(gamma_labels, gamma_colors=gamma_colors)
    kinds = tuple(pareto_frontier_kinds)
    resolved_pareto_linestyle = _resolve_pareto_linestyle(
        pareto_linestyle, kind="plasticity", junctures_mode="both"
    )

    fig, ax = plt.subplots(figsize=figsize)
    pareto_handles = []
    grouped = None

    if aggregate_runs:
        grouped = (
            df.groupby(["gamma_label", "experiment"], dropna=False)
            .agg(
                x_mean=("x", "mean"),
                x_std=("x", "std"),
                y_mean=("y", "mean"),
                y_std=("y", "std"),
                kind=("kind", "first"),
                n_runs=("run", "nunique"),
            )
            .reset_index()
        )
        plot_source = grouped
        x_plot_col, y_plot_col = "x_mean", "y_mean"
    else:
        plot_source = df
        x_plot_col, y_plot_col = "x", "y"

    if show_points and aggregate_runs and grouped is not None:
        for _, row in grouped.iterrows():
            if row["kind"] not in kinds:
                continue
            color = color_map[row["gamma_label"]]
            ax.errorbar(
                row["x_mean"],
                row["y_mean"],
                xerr=row["x_std"] if pd.notna(row["x_std"]) else None,
                yerr=row["y_std"] if pd.notna(row["y_std"]) else None,
                fmt="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=8,
                capsize=3,
                linestyle="none",
                alpha=0.95,
                zorder=2,
            )
        for _, row in df.iterrows():
            if row["kind"] not in kinds:
                continue
            color = color_map[row["gamma_label"]]
            ax.scatter(
                row["x"],
                row["y"],
                color=color,
                alpha=alpha_individual,
                s=marker_size * 0.5,
                zorder=1,
            )

    for gamma_label in gamma_labels:
        color = color_map[gamma_label]
        subset = plot_source[
            (plot_source["gamma_label"] == gamma_label)
            & (plot_source["kind"].isin(kinds))
        ]
        if subset.empty:
            continue

        if show_points and not aggregate_runs:
            ax.scatter(
                subset[x_plot_col],
                subset[y_plot_col],
                color=color,
                alpha=alpha_individual,
                s=marker_size * 0.5,
                zorder=2,
            )

        pareto_df = subset.rename(columns={x_plot_col: "x", y_plot_col: "y"})
        handle = _draw_pareto_frontier(
            ax,
            pareto_df,
            color=color,
            label=f"{gamma_label} Pareto",
            y_goal=resolved_y_goal,
            linestyle=resolved_pareto_linestyle,
            linewidth=pareto_linewidth,
        )
        if handle is not None:
            pareto_handles.append(handle)

    if legend_mode == "full":
        point_handles = []
        for gamma_label in gamma_labels:
            if gamma_label not in df["gamma_label"].values:
                continue
            color = color_map[gamma_label]
            point_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=8,
                    label=gamma_label,
                )
            )
        handles = point_handles + pareto_handles
        legend_title = "Growth mode / Pareto"
    else:
        handles = list(pareto_handles)
        legend_title = "Pareto"

    if handles:
        legend_kwargs = {"handles": handles, "title": legend_title}
        if legend_fontsize is not None:
            legend_kwargs["fontsize"] = legend_fontsize
            legend_kwargs["title_fontsize"] = legend_fontsize
        ax.legend(**legend_kwargs)

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    if x_tick_interval is not None:
        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(float(x_tick_interval)))
    if tick_label_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path_out is not None:
        out = Path(save_path_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, df


def _parse_gamma_label_from_save_path(save_path):
    """Extract a display label from a results root like ..._gamma0.01."""
    name = Path(save_path).name
    match = re.search(r"gamma([0-9.e-]+)", name, re.I)
    if match:
        return f"γ={match.group(1)}"
    return name


def _default_gamma_color_map(gamma_labels, gamma_colors=None):
    """Assign a distinct color to each gamma / growth-mode label."""
    if gamma_colors is not None:
        return dict(gamma_colors)
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(gamma_labels)}


def plot_gamma_pareto_comparison(
    save_paths,
    gamma_labels=None,
    experiments=None,
    experiment_glob="*",
    num_runs=5,
    aggregate_runs=True,
    x_col="Parameters",
    y_col="Test Acc",
    xlabel=None,
    ylabel=None,
    title=None,
    save_path_out=None,
    show=True,
    figsize=(10, 7),
    dpi=150,
    pareto_frontier_kinds=("plasticity",),
    show_points=True,
    gamma_colors=None,
    pareto_linestyle="--",
    pareto_linewidth=1.5,
    pareto_y_goal=None,
    alpha_individual=0.35,
    marker_size=70,
    legend_mode="pareto",
    axis_label_fontsize=None,
    tick_label_fontsize=None,
    title_fontsize=None,
    legend_fontsize=None,
    x_tick_interval=None,
):
    """
    Compare plasticity Pareto frontiers across result roots (e.g. growth modes).

    Loads the same leaf experiment names from each parent ``save_path``, tags
    rows with ``gamma_label``, then plots Acc/Brier vs parameter count.

    Typical usage
    -------------
    plot_gamma_pareto_comparison(
        save_paths=["results_cifar10", "results_cifar10_random_growth"],
        gamma_labels=["Uncertainty growth", "Random growth"],
        experiments=["plasticity_300f_300f_1e-07", ...],
        y_col="Test Acc",
        save_path_out="cifar10_plots/random_growth/acc.pdf",
        show=False,
    )
    """
    if not save_paths:
        raise ValueError("save_paths must be a non-empty list of result roots")
    if legend_mode not in ("full", "pareto"):
        raise ValueError(
            f"legend_mode must be 'full' or 'pareto', got {legend_mode!r}"
        )

    save_paths = [Path(p) for p in save_paths]
    if gamma_labels is None:
        gamma_labels = [_parse_gamma_label_from_save_path(p) for p in save_paths]
    if len(gamma_labels) != len(save_paths):
        raise ValueError(
            f"gamma_labels length ({len(gamma_labels)}) must match "
            f"save_paths ({len(save_paths)})"
        )

    frames = []
    for save_path, gamma_label in zip(save_paths, gamma_labels):
        frame = collect_experiment_summaries(
            save_path=str(save_path),
            experiments=experiments,
            experiment_glob=experiment_glob,
            num_runs=num_runs,
            x_col=x_col,
            y_col=y_col,
        )
        frame = frame.copy()
        frame["gamma_label"] = gamma_label
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    if xlabel is None:
        if x_col in ("FLOPs", "FLOPs_realized", "FLOPs_theoretical"):
            xlabel = "FLOPs"
        elif x_col == "Sparse Parameters":
            xlabel = "Sparse parameter count"
        elif x_col == "Dense Parameters":
            xlabel = "Dense parameter count"
        else:
            xlabel = "Parameter count"
    if ylabel is None:
        ylabel = y_col
    if title is None:
        title = f"{ylabel} vs parameter count by growth rate"

    resolved_y_goal = pareto_y_goal or _infer_pareto_y_goal(y_col, ylabel)
    color_map = _default_gamma_color_map(gamma_labels, gamma_colors=gamma_colors)
    kinds = tuple(pareto_frontier_kinds)
    resolved_pareto_linestyle = _resolve_pareto_linestyle(
        pareto_linestyle, kind="plasticity", junctures_mode="both"
    )

    fig, ax = plt.subplots(figsize=figsize)
    pareto_handles = []
    grouped = None

    if aggregate_runs:
        grouped = (
            df.groupby(["gamma_label", "experiment"], dropna=False)
            .agg(
                x_mean=("x", "mean"),
                x_std=("x", "std"),
                y_mean=("y", "mean"),
                y_std=("y", "std"),
                kind=("kind", "first"),
                n_runs=("run", "nunique"),
            )
            .reset_index()
        )
        plot_source = grouped
        x_plot_col, y_plot_col = "x_mean", "y_mean"
    else:
        plot_source = df
        x_plot_col, y_plot_col = "x", "y"

    if show_points and aggregate_runs and grouped is not None:
        for _, row in grouped.iterrows():
            if row["kind"] not in kinds:
                continue
            color = color_map[row["gamma_label"]]
            ax.errorbar(
                row["x_mean"],
                row["y_mean"],
                xerr=row["x_std"] if pd.notna(row["x_std"]) else None,
                yerr=row["y_std"] if pd.notna(row["y_std"]) else None,
                fmt="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=8,
                capsize=3,
                linestyle="none",
                alpha=0.95,
                zorder=2,
            )
        for _, row in df.iterrows():
            if row["kind"] not in kinds:
                continue
            color = color_map[row["gamma_label"]]
            ax.scatter(
                row["x"],
                row["y"],
                color=color,
                alpha=alpha_individual,
                s=marker_size * 0.5,
                zorder=1,
            )

    for gamma_label in gamma_labels:
        color = color_map[gamma_label]
        subset = plot_source[
            (plot_source["gamma_label"] == gamma_label)
            & (plot_source["kind"].isin(kinds))
        ]
        if subset.empty:
            continue

        if show_points and not aggregate_runs:
            ax.scatter(
                subset[x_plot_col],
                subset[y_plot_col],
                color=color,
                alpha=alpha_individual,
                s=marker_size * 0.5,
                zorder=2,
            )

        pareto_df = subset.rename(columns={x_plot_col: "x", y_plot_col: "y"})
        handle = _draw_pareto_frontier(
            ax,
            pareto_df,
            color=color,
            label=f"{gamma_label} Pareto",
            y_goal=resolved_y_goal,
            linestyle=resolved_pareto_linestyle,
            linewidth=pareto_linewidth,
        )
        if handle is not None:
            pareto_handles.append(handle)

    if legend_mode == "full":
        point_handles = []
        for gamma_label in gamma_labels:
            if gamma_label not in df["gamma_label"].values:
                continue
            color = color_map[gamma_label]
            point_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=8,
                    label=gamma_label,
                )
            )
        handles = point_handles + pareto_handles
        legend_title = "Growth mode / Pareto"
    else:
        handles = list(pareto_handles)
        legend_title = "Pareto"

    if handles:
        legend_kwargs = {"handles": handles, "title": legend_title}
        if legend_fontsize is not None:
            legend_kwargs["fontsize"] = legend_fontsize
            legend_kwargs["title_fontsize"] = legend_fontsize
        ax.legend(**legend_kwargs)

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    if x_tick_interval is not None:
        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(float(x_tick_interval)))
    if tick_label_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path_out is not None:
        out = Path(save_path_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, df
