# Bayesian CNN experiments

Code for the Bayesian convolutional network experiments (structural plasticity on Fashion-MNIST / CIFAR-10).

Run **all commands from this directory** (`thesis_experiments/Bayesian/BayesianCNN`).

## Setup

```bash
pip install -r requirements.txt
```

Datasets are read from `../../Datasets` (Fashion-MNIST / CIFAR-10 via torchvision, `download=True`). That folder is gitignored.

Experiment outputs under `results*` / `results_*/` are also gitignored (see `thesis_experiments/.gitignore`: `**/results/`, `**/results*/`) and are **not** pushed to GitHub. Regenerate them locally by running the experiments and `python -m scripts.make_figures`.

## Seed contract

- Sweeps call `set_seed(42 + i)` for runs `i = 1..5`, then `main(...)`.

## Experiments

| Thesis comparison | Module | Typical invocation |
|---|---|---|
| Baseline / plasticity / three-phase / static replay | `experiments.run_plasticity` | `python -m experiments.run_plasticity` |

`experiments.run_plasticity.main(..., run_mode=...)` selects `baseline`, `plasticity`, `three_phase`, or `static_replay`.

`dataset` is `fashion_mnist` or `cifar10`. Architecture knobs: `conv_channels` (list of conv widths) and `fc_hidden`.

### Plasticity knobs

- `junctures_mode`: `both`, `grow`, `prune`, or `both_gp`
- Growth layer score: `mean` or `mad` (`growth_mad_percentile`)
- Prune: `prune_mode="global_param"` with `global_prune_normalize` in `{percentile, zscore, mad, raw}` and budget `filters` / `params`
- `static_replay` requires `resume_from_plasticity_dir` (reads `Conv Channels` from that run’s `experiment_summary.csv`)

## Figures

From saved `experiment_summary.csv` trees (no retraining):

```bash
python -m scripts.make_figures
```

This writes:

- Fashion-MNIST CNN Pareto plots from `results_fashionmnist_cnn/`: `results_fashionmnist_cnn_acc.png`, `results_fashionmnist_cnn_brier.png`

## Layout

```text
models/       BayesianCNN (+ vendored BayesianLinear)
lib/          seed, data, train/eval, grow/prune, plots
experiments/  plasticity runner
scripts/      figure assembly from saved summaries
```
