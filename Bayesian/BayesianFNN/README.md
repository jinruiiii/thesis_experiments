# Bayesian FNN experiments

Code for the Bayesian fully-connected network experiments.

Run **all commands from this directory** (`thesis_experiments/Bayesian/BayesianFNN`).

## Setup

```bash
pip install -r ../requirements.txt
```

Datasets are read from `../../Datasets` (Fashion-MNIST / CIFAR-10 via torchvision, `download=True`). That folder is gitignored.

Experiment outputs under `results*` / `results_*/` are also gitignored (see `thesis_experiments/.gitignore`: `**/results/`, `**/results*/`) and are **not** pushed to GitHub.

## Seed contract

- Sweeps call `set_seed(42 + i)` for runs `i = 1..5`, then `main(...)`.

## Reproduce experiments

### Fashion-MNIST FNN

```bash
python -m scripts.script_fashionmnist
```

After results are written under `results_fashionmnist/` (and related trees), build plots:

```bash
python -m scripts.make_figures_fashionmnist
```

### CIFAR-10 FNN

```bash
python -m scripts.script_cifar10
```

After results are written under `results_cifar10/` (and related trees), build plots:

```bash
python -m scripts.make_figures_cifar10
```

## Layout

```text
models/       BayesianFNN and sparse NeST layers
lib/          seed, data, train/eval, grow/prune, FLOPs, plots
experiments/  plasticity, NeST, and prior-shift runners
scripts/      experiment sweeps and figure assembly
```
