# Bayesian plasticity experiments

Code for structural plasticity experiments with Bayesian neural networks.

## Setup

From this directory (`thesis_experiments/Bayesian`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## What’s included

This repo contains the codebase for three plasticity configurations:

| Configuration | Folder |
|---|---|
| Fashion-MNIST FNN | [`BayesianFNN/`](BayesianFNN/) |
| CIFAR-10 FNN | [`BayesianFNN/`](BayesianFNN/) |
| CIFAR-10 CNN | [`BayesianCNN/`](BayesianCNN/) |

Read the README in each folder for how to reproduce experiments and figures.

## Results and datasets

Experiment outputs under `results*` / `results_*/` are gitignored (see `thesis_experiments/.gitignore`: `**/results/`, `**/results*/`) and are **not** pushed to GitHub. Regenerate them locally by running the experiment scripts and figure scripts described in the subfolder READMEs.

Datasets are downloaded under `thesis_experiments/Datasets` (also gitignored).
