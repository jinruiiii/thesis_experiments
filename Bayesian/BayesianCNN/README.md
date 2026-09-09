# Bayesian CNN experiments

Code for the Bayesian convolutional network experiments.

Run **all commands from this directory** (`thesis_experiments/Bayesian/BayesianCNN`).

## Setup

```bash
pip install -r ../requirements.txt
```

Datasets are read from `../../Datasets` (CIFAR-10 via torchvision, `download=True`). That folder is gitignored.

Experiment outputs under `results*` / `results_*/` are also gitignored (see `thesis_experiments/.gitignore`: `**/results/`, `**/results*/`) and are **not** pushed to GitHub.

## Seed contract

- Sweeps call `set_seed(42 + i)` for runs `i = 1..5`, then `main(...)`.

## Reproduce experiments

### CIFAR-10 CNN

```bash
python -m scripts.script_cifar10
```

After results are written under `results_cifar10/` (and related trees), build plots:

```bash
python -m scripts.make_figures_cifar10
```

## Layout

```text
models/       BayesianCNN and sparse NeST layers
lib/          seed, data, train/eval, grow/prune, plots
experiments/  plasticity, NeST, and prior-shift runners
scripts/      experiment sweeps and figure assembly
```
