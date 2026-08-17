# Bayesian FNN experiments

Code for the Bayesian fully-connected network experiments (structural plasticity, DropNet, NeST, and Fashion-MNIST prior shift).

Run **all commands from this directory** (`thesis_experiments/Bayesian/BayesianFNN`).

## Setup

```bash
pip install -r requirements.txt
```

Datasets are read from `../../Datasets` (Fashion-MNIST / KMNIST via torchvision, `download=True`). That folder is gitignored.

## Seed contract

- Sweeps call `set_seed(42 + i)` for runs `i = 1..5`, then `main(...)`.

## Experiments

| Thesis comparison | Module | Typical invocation |
|---|---|---|
| Baseline / plasticity / three-phase / static replay | `experiments.run_plasticity` | `python -m scripts.fashion_mnist_pareto` |
| DropNet | `experiments.run_dropnet` | `python -m experiments.run_dropnet` |
| NeST | `experiments.run_nest` | `python -m experiments.run_nest` |
| Prior shift (VCL) | `experiments.run_shift` | `python -m scripts.prior_shift` |

`experiments.run_plasticity.main(..., run_mode=...)` selects `baseline`, `plasticity`, `three_phase`, `static_replay`, `hybrid`, or `plasticity_and_hybrid`.

## Figures

From saved `experiment_summary.csv` trees (no retraining):

```bash
python -m scripts.make_figures
```

This writes Fashion-MNIST Pareto plots (`results_FashionMnist_FNN_acc.png`, `results_FashionMnist_FNN_brier.png`) and prior-shift VCL plots under `results_prior_shift_FashionMnist_FNN/`.

## Layout

```text
models/       BayesianFNN and sparse NeST layers
lib/          seed, data, train/eval, grow/prune, plots
experiments/  method runners
scripts/      multi-seed sweeps and figure assembly
```
