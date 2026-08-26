# Bayesian FNN experiments

Code for the Bayesian fully-connected network experiments (structural plasticity, NeST, and Fashion-MNIST prior shift).

Run **all commands from this directory** (`thesis_experiments/Bayesian/BayesianFNN`).

## Setup

```bash
pip install -r requirements.txt
```

Datasets are read from `../../Datasets` (Fashion-MNIST / KMNIST via torchvision, `download=True`). That folder is gitignored.

Experiment outputs under `results*` / `results_*/` are also gitignored (see `thesis_experiments/.gitignore`: `**/results/`, `**/results*/`) and are **not** pushed to GitHub. Regenerate them locally by running the experiments and `python -m scripts.make_figures`.

## Seed contract

- Sweeps call `set_seed(42 + i)` for runs `i = 1..5`, then `main(...)`.

## Experiments

| Thesis comparison | Module | Typical invocation |
|---|---|---|
| Baseline / plasticity / three-phase / static replay | `experiments.run_plasticity` | `python -m experiments.run_plasticity` |
| NeST | `experiments.run_nest` | `python -m experiments.run_nest` |
| Prior shift (VCL) | `experiments.run_shift` | `python -m experiments.run_shift` |

`experiments.run_plasticity.main(..., run_mode=...)` selects `baseline`, `plasticity`, `three_phase`, or `static_replay`.

### Plasticity knobs

- `junctures_mode`: `both`, `grow`, `prune`, or `both_gp`
- Growth layer score: `mean` or `mad` (`growth_mad_percentile`)
- Prune: `prune_mode="global_param"` with `global_prune_normalize` in `{percentile, zscore, mad, raw}` and budget `neurons` / `params`
- `static_replay` requires `resume_from_plasticity_dir` (reads `Hidden Sizes` from that run’s `experiment_summary.csv`)

## Figures

From saved `experiment_summary.csv` trees (no retraining):

```bash
python -m scripts.make_figures
```

This writes:

- Fashion-MNIST Pareto / param plots from `results_fashionmnist/`: `results_fashionmnist_acc.png`, `results_fashionmnist_brier.png`, `results_fashionmnist_param_count_vs_epoch.png`
- Prior-shift decision heatmap from `results_prior_shift_fashionmnist/`: `decision_heatmap.png`

## Layout

```text
models/       BayesianFNN and sparse NeST layers
lib/          seed, data, train/eval, grow/prune, FLOPs, plots
experiments/  plasticity, NeST, and prior-shift runners
scripts/      figure assembly from saved summaries
```
