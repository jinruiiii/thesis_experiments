from BayesianFNN import BayesianFNN
import ast
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import copy
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import re


SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()  # initial seed at import

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Random seed set to: {SEED} for full reproducibility")

def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_checkpoint(
    path,
    state_dict,
    epoch,
    hidden_sizes=None,
    selection_metric=None,
    selection_value=None,
):
    payload = {
        "state_dict": state_dict,
        "epoch": int(epoch),
    }
    if hidden_sizes is not None:
        payload["hidden_sizes"] = list(hidden_sizes)
    if selection_metric is not None:
        payload["selection_metric"] = str(selection_metric)
    if selection_value is not None:
        payload["selection_value"] = float(selection_value)
    torch.save(payload, path)


def load_checkpoint(path, in_features, out_features, device=None):
    """Load a BayesianFNN from a checkpoint saved by save_checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(path, map_location=device)
    hidden_sizes = payload.get("hidden_sizes")
    if hidden_sizes is None:
        raise ValueError(f"checkpoint at {path!r} is missing hidden_sizes")
    model = BayesianFNN(in_features, list(hidden_sizes), out_features).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    metadata = {
        "epoch": payload.get("epoch"),
        "selection_metric": payload.get("selection_metric"),
        "selection_value": payload.get("selection_value"),
    }
    return model, list(hidden_sizes), metadata


def write_hybrid_provenance(output_dir, provenance):
    """Write hybrid experiment lineage metadata as JSON."""
    path = os.path.join(output_dir, "hybrid_provenance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    return path


def write_metrics_csv(output_dir, metrics):
    metrics_df = pd.DataFrame({
        'epoch': range(1, 1 + len(metrics['train_loss_total'])),
        'train_loss_total': metrics['train_loss_total'],
        'train_loss_nll': metrics['train_loss_nll'],
        'train_loss_kl': metrics['train_loss_kl'],
        'train_loss_penalty': metrics['train_loss_penalty'],
        'train_acc': metrics['train_acc'],
        'train_brier': metrics['train_brier'],
        'val_loss_total': metrics['val_loss_total'],
        'val_loss_nll': metrics['val_loss_nll'],
        'val_loss_kl': metrics['val_loss_kl'],
        'val_loss_penalty': metrics['val_loss_penalty'],
        'val_acc': metrics['val_acc'],
        'val_brier': metrics['val_brier'],
        'param_count': metrics['param_count_history'],
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

def write_experiment_summary_csv(
    output_dir,
    model_label,
    params,
    trainable_params,
    hidden_sizes,
    best_epoch,
    best_val_total,
    best_val_acc,
    best_val_brier,
    test_acc,
    test_brier,
    lambda_penalty,
    selected_checkpoint_metric= "val_total",
    junctures_mode="both",
):
    summary_df = pd.DataFrame([{
        "Model": model_label,
        "Parameters": int(params),
        "Trainable Params": int(trainable_params),
        "Best Val Acc": float(best_val_acc),
        "Best Val Brier": float(best_val_brier),
        "Test Acc": float(test_acc),
        "Test Brier": float(test_brier),
        "Hidden Sizes": str(list(hidden_sizes)),
        "Lambda Penalty": float(lambda_penalty),
        "Junctures Mode": str(junctures_mode),
        "Selected checkpoint metric": str(selected_checkpoint_metric),
        "Selected epoch": int(best_epoch),
        "Selected val_total": float(best_val_total),
    }])
    summary_df.to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)


def _normalize_checkpoint_metric(checkpoint_metric):
    if checkpoint_metric in ("val_loss_nll", "val_nll"):
        return "val_loss_nll"
    if checkpoint_metric in ("val_loss_total", "val_total"):
        return "val_loss_total"
    raise ValueError(
        f"checkpoint_metric must be 'val_loss_nll' or 'val_loss_total', got {checkpoint_metric!r}"
    )


def _checkpoint_metric_label(checkpoint_metric, phase2=False):
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    prefix = "phase2_mixed_" if phase2 else ""
    if checkpoint_metric == "val_loss_nll":
        return f"{prefix}val_nll"
    return f"{prefix}val_total"


def _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric):
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    if checkpoint_metric == "val_loss_nll":
        return val_loss_nll
    return val_loss_total


def _is_better_checkpoint_score(candidate, best, checkpoint_metric):
    return candidate < best


def seed_worker(worker_id):
    """Function to ensure DataLoader workers use different seeds derived from the base seed"""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


DATASET_ROOT = "../../Datasets"
SUPPORTED_DATASETS = {
    "fashion_mnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
}


def _dataset_to_flat_tensors(raw_dataset):
    """Vectorized ToTensor + flatten over the whole dataset (done once)."""
    x = raw_dataset.data.float().div_(255.0).flatten(1)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def build_dataloaders(dataset_name, batch_size, train_frac=0.8, seed=SEED):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {sorted(SUPPORTED_DATASETS)}, got {dataset_name!r}"
        )

    dataset_cls = SUPPORTED_DATASETS[dataset_name]

    training_data_raw = dataset_cls(root=DATASET_ROOT, train=True, download=True)
    test_data_raw = dataset_cls(root=DATASET_ROOT, train=False, download=True)

    x_all, y_all = _dataset_to_flat_tensors(training_data_raw)
    x_test, y_test = _dataset_to_flat_tensors(test_data_raw)

    train_size = int(train_frac * len(training_data_raw))
    val_size = len(training_data_raw) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(training_data_raw)), [train_size, val_size], generator=generator
    )
    train_idx = torch.tensor(train_indices.indices, dtype=torch.long)
    val_idx = torch.tensor(val_indices.indices, dtype=torch.long)
    train_dataset = TensorDataset(x_all[train_idx], y_all[train_idx])
    val_dataset = TensorDataset(x_all[val_idx], y_all[val_idx])
    test_dataset = TensorDataset(x_test, y_test)

    pin_memory = torch.cuda.is_available()
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=pin_memory,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        generator=g,
    )
    return train_loader, val_loader, test_loader


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

def loss_function(outputs, labels, kl_loss, beta):
    criterion = nn.CrossEntropyLoss()
    nll = criterion(outputs, labels)
    # normalise to per sample
    return nll + kl_loss*beta, nll, kl_loss*beta


def count_params(model):
    """Total variational parameters |theta| for the penalised ELBO."""
    if hasattr(model, 'get_param_stats'):
        return model.get_param_stats()['total_params']
    return sum(p.numel() for p in model.parameters())


def penalised_loss_function(outputs, labels, kl_loss, beta, lambda_penalty, param_count):
    """Minimised objective: NLL + beta*KL + lambda*|theta| (negative penalised ELBO)."""
    _, nll, kl_scaled = loss_function(outputs, labels, kl_loss, beta)
    penalty = lambda_penalty * param_count
    total = nll + kl_scaled + penalty
    return total, nll, kl_scaled, penalty


def penalised_elbo_on_batch(model, inputs, labels, beta_scaled, lambda_penalty):
    """Penalised ELBO L_pen on a single batch (higher is better)."""
    model.eval()
    param_count = count_params(model)
    with torch.no_grad():
        outputs = model(inputs)
        loss, nll, kl_scaled, penalty = penalised_loss_function(
            outputs, labels, model.kl_loss(), beta_scaled, lambda_penalty, param_count
        )
    return -(loss.item())

def penalised_elbo_on_batches(model, batches_val, beta_scaled, lambda_penalty):
    """
    Mean penalised ELBO over several validation batches (higher is better).
    """
    vals = []
    for inputs, labels in batches_val:
        vals.append(penalised_elbo_on_batch(model, inputs, labels, beta_scaled, lambda_penalty))
    return float(np.mean(vals))


def sample_batch(loader, device):
    """Draw one random mini-batch (shuffle=True for stochastic B_ws / B_val)."""
    g = torch.Generator()
    g.manual_seed(SEED + random.randint(0, 1_000_000))
    shuffled_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
    )
    inputs, labels = next(iter(shuffled_loader))
    return inputs.to(device), labels.to(device)

# def sample_batches(loader, device, num_batches):
#     """
#     Draw num_batches random mini-batches from loader.dataset (like sample_batch, but repeated).
#     Returns list[(inputs, labels)].
#     """
#     batches = []
#     for _ in range(num_batches):
#         batches.append(sample_batch(loader, device))
#     return batches

def sample_batches(loader, device, num_batches):
    """
    Draw num_batches consecutive mini-batches from a single shuffled pass
    (no replacement within this chunk).
    """
    g = torch.Generator()
    g.manual_seed(SEED + random.randint(0, 1_000_000))
    shuffled_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
        num_workers=loader.num_workers,
        worker_init_fn=getattr(loader, "worker_init_fn", None),
    )
    batches = []
    it = iter(shuffled_loader)
    for _ in range(num_batches):
        inputs, labels = next(it)
        batches.append((inputs.to(device), labels.to(device)))
    return batches


def train(model, train_dataloader, optimizer, epoch, device, beta_scaled, lambda_penalty=0):
    model.train()
    running_loss_total = 0.0
    running_loss_nll = 0.0
    running_loss_kl = 0.0
    running_loss_penalty = 0.0
    running_brier = 0.0
    correct = 0
    total = 0
    param_count = count_params(model)

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')

    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        if lambda_penalty > 0:
            loss, nll, kl, penalty = penalised_loss_function(
                outputs, labels, model.kl_loss(), beta_scaled, lambda_penalty, param_count
            )
        else:
            loss, nll, kl = loss_function(outputs, labels, model.kl_loss(), beta_scaled)
            penalty = torch.tensor(0.0)
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss_total += loss.item()
        running_loss_nll += nll.item()
        running_loss_kl += kl.item()
        running_loss_penalty += penalty.item() if hasattr(penalty, 'item') else penalty

        # Accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Brier Score
        probs = torch.softmax(outputs, dim=1)
        num_classes = outputs.size(1)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        
        brier = torch.sum((probs - one_hot) ** 2, dim=1).sum()
        running_brier += brier.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss_total / (progress_bar.n + 1),
            'acc': 100. * correct / total,
            'brier': running_brier / total
        })
    train_loss_total = running_loss_total / len(train_dataloader)
    train_acc = 100. * correct / total
    train_loss_nll = running_loss_nll / len(train_dataloader)
    train_loss_kl = running_loss_kl / len(train_dataloader)
    train_loss_penalty = running_loss_penalty / len(train_dataloader)
    train_brier = running_brier / total

    return train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty


def validate(model, val_dataloader, device, beta_scaled, lambda_penalty=0):
    model.eval()
    val_loss_total = 0.0
    val_loss_nll = 0.0
    val_loss_kl = 0.0
    val_loss_penalty = 0.0
    running_brier = 0.0
    correct = 0
    total = 0
    param_count = count_params(model)

    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc='Validating'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            if lambda_penalty > 0:
                loss, nll, kl, penalty = penalised_loss_function(
                    outputs, labels, model.kl_loss(), beta_scaled, lambda_penalty, param_count
                )
            else:
                loss, nll, kl = loss_function(outputs, labels, model.kl_loss(), beta_scaled)
                penalty = torch.tensor(0.0)

            # Track Statistics
            val_loss_total += loss.item()
            val_loss_nll += nll.item()
            val_loss_kl += kl.item()
            val_loss_penalty += penalty.item() if hasattr(penalty, 'item') else penalty
            
            # Accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Brier Score
            probs = torch.softmax(outputs, dim=1)
            num_classes = outputs.size(1)
            one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            
            brier = torch.sum((probs - one_hot) ** 2, dim=1).sum()
            running_brier += brier.item()
            

    val_loss_total = val_loss_total / len(val_dataloader)
    val_acc = 100. * correct / total
    val_loss_nll = val_loss_nll / len(val_dataloader)
    val_loss_kl = val_loss_kl / len(val_dataloader)
    val_loss_penalty = val_loss_penalty / len(val_dataloader)
    val_brier = running_brier / total

    return val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty


def neurogenesis(
    plasticity_original,
    hidden_sizes,
    exclude=None,
    gamma=0.1,
    uncertainty_combine="geometric",
    growth_layer_score="mean",
    growth_mad_percentile=100.0,
):
    """Growth candidate: expand layer l* with highest growth score."""
    growth_layer_score = _normalize_growth_layer_score(growth_layer_score)
    growth_mad_percentile = _validate_growth_mad_percentile(growth_mad_percentile)
    if exclude is None:
        exclude = []
    n_layers = len(hidden_sizes)
    eligible = [i for i in range(n_layers) if i not in exclude]
    if not eligible:
        print("[neurogenesis] All layers excluded; ignoring exclude list for this step.")
        eligible = list(range(n_layers))

    if growth_layer_score == "mean":
        uncertainty = plasticity_original.get_average_bidirectional_uncertainty_per_layer(
            combine=uncertainty_combine
        )
        layer_scores = {
            i: uncertainty[i].item() / (hidden_sizes[i] ** 0.5)
            for i in range(n_layers)
        }
        print("\n Average Bidirectional Normalised Uncertainty per Hidden Layer:")
        for i, val in enumerate(uncertainty):
            print(f"  Layer {i+1}: {val.item()/(hidden_sizes[i]**0.5):.6f}")
        score_label = "normalised mean uncertainty"
    else:
        raw = _collect_hidden_neuron_uncertainty_scores(
            plasticity_original, uncertainty_combine=uncertainty_combine
        )
        layer_scores = _layer_growth_scores_from_mad(raw, growth_mad_percentile)
        pct_label = "max" if growth_mad_percentile >= 100.0 else f"p{growth_mad_percentile:g}"
        print(
            f"\n Growth layer MAD scores (combine={uncertainty_combine}, "
            f"percentile={growth_mad_percentile:g}):"
        )
        for i in sorted(layer_scores.keys()):
            print(f"  Layer {i+1}: mad_{pct_label}={layer_scores[i]:.6f}")
        score_label = f"mad_{pct_label}"

    layer_to_expand = max(eligible, key=lambda i: layer_scores[i])
    neurons_to_add = max(1, math.ceil(gamma * hidden_sizes[layer_to_expand]))
    old_width = hidden_sizes[layer_to_expand]
    print(
        f"Expanding Layer {layer_to_expand+1} "
        f"(highest {score_label}: {layer_scores[layer_to_expand]:.6f}) "
        f"by {neurons_to_add} neurons"
    )

    expanded_hidden_sizes = hidden_sizes.copy()
    expanded_hidden_sizes[layer_to_expand] += neurons_to_add
    model_device = next(plasticity_original.parameters()).device
    plasticity_neurogenesis = BayesianFNN(
        plasticity_original.in_features, expanded_hidden_sizes, plasticity_original.out_features
    ).to(model_device)
    return plasticity_neurogenesis, expanded_hidden_sizes, layer_to_expand, old_width

def expand_and_load_encoder_layer(old_sd, new_layer):
    new_sd = new_layer.state_dict()
    for k in new_sd.keys():
        if k not in old_sd:
            print(f"[skip] {k} not found in old layer")
            continue

        old_param = old_sd[k]
        new_param = new_sd[k]

        if old_param.shape == new_param.shape:
            new_sd[k] = old_param
        elif len(old_param.shape) == 2:
            # Linear weights: expand top-left corner
            new_sd[k][:old_param.shape[0], :old_param.shape[1]] = old_param
        elif len(old_param.shape) == 1:
            # Bias / LayerNorm
            new_sd[k][:old_param.shape[0]] = old_param
        else:
            print(f"[warn] Shape mismatch for {k}: old {old_param.shape}, new {new_param.shape}")

    new_layer.load_state_dict(new_sd, strict=True)

def _weight_snr(mu, rho, eps=1e-8):
    """Element-wise |mu / sigma| for weight tensors."""
    sigma = F.softplus(rho)

    return torch.abs(mu) / (sigma + eps)
def _neuron_snr_incoming(layer, eps=1e-8):
    """Mean incoming SNR per output neuron (row j of mu_w)."""
    snr = _weight_snr(layer.mu_w, layer.rho_w, eps)
    return snr.mean(dim=1)

def _neuron_snr_outgoing(next_layer, neuron_idx, eps=1e-8):
    """Mean outgoing SNR for hidden neuron j via column j of the next layer."""
    snr = _weight_snr(next_layer.mu_w, next_layer.rho_w, eps)
    return snr[:, neuron_idx].mean()

def _neuron_snr_bidirectional(model, layer_idx, neuron_idx, eps=1e-8, combine="geometric"):
    """
    Combined SNR for neuron j in hidden layer layer_idx.
    combine: 'geometric' (sqrt(in*out)), 'min', or 'mean'
    """
    layer = model.layers[layer_idx]
    snr_in = _neuron_snr_incoming(layer, eps)[neuron_idx]
    if layer_idx + 1 < len(model.layers):
        snr_out = _neuron_snr_outgoing(model.layers[layer_idx + 1], neuron_idx, eps)
    else:
        snr_out = _neuron_snr_outgoing(model.out, neuron_idx, eps)
    if combine == "min":
        return torch.min(snr_in, snr_out)
    if combine == "mean":
        return 0.5 * (snr_in + snr_out)
    # geometric mean (default): penalises neurons weak on either path
    return torch.sqrt(snr_in * snr_out + eps)


def _hidden_neuron_param_cost(model, layer_idx):
    """
    Variational parameter count removed when pruning one neuron from hidden layer layer_idx.
    Matches truncate_and_load_encoder_layer: row at layer i plus downstream column.
    """
    layer = model.layers[layer_idx]
    if layer_idx + 1 < len(model.layers):
        downstream_out = model.layers[layer_idx + 1].out_features
    else:
        downstream_out = model.out.out_features
    return 2 * layer.in_features + 4 + 2 * downstream_out


def _hidden_neuron_snr_percentile_ranks(raw_scores_by_layer):
    """
    Per-layer percentile rank from precomputed raw SNR scores.

    raw_scores_by_layer: dict layer_idx -> list of raw SNR per neuron.
    Returns dict (layer_idx, neuron_idx) -> rank in [0, 1] where 0 is worst
    within that layer.
    """
    ranks = {}
    for layer_idx, raw_scores in raw_scores_by_layer.items():
        n_neurons = len(raw_scores)
        if n_neurons == 0:
            continue
        if n_neurons == 1:
            ranks[(layer_idx, 0)] = 0.0
            continue
        order = sorted(range(n_neurons), key=lambda j: (raw_scores[j], j))
        for rank, j in enumerate(order):
            ranks[(layer_idx, j)] = rank / (n_neurons - 1)
    return ranks


def _hidden_neuron_snr_zscores(raw_scores_by_layer, eps=1e-8):
    """
    Per-layer z-score from precomputed raw SNR scores.

    raw_scores_by_layer: dict layer_idx -> list of raw SNR per neuron.
    Returns dict (layer_idx, neuron_idx) -> z where lower is worse within layer.
    """
    zscores = {}
    for layer_idx, raw_scores in raw_scores_by_layer.items():
        n_neurons = len(raw_scores)
        if n_neurons == 0:
            continue
        if n_neurons == 1:
            zscores[(layer_idx, 0)] = 0.0
            continue
        mean = sum(raw_scores) / n_neurons
        variance = sum((score - mean) ** 2 for score in raw_scores) / n_neurons
        std = math.sqrt(variance) + eps
        for j, score in enumerate(raw_scores):
            zscores[(layer_idx, j)] = (score - mean) / std
    return zscores


def _median(values):
    """Return the median of a non-empty list of numbers."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _percentile(values, p):
    """Linear-interpolation percentile; p=100 returns max, p=0 returns min."""
    if not values:
        raise ValueError("values must be non-empty for percentile")
    p = float(p)
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _validate_growth_mad_percentile(growth_mad_percentile):
    p = float(growth_mad_percentile)
    if not (0.0 <= p <= 100.0):
        raise ValueError(
            f"growth_mad_percentile must be in [0, 100], got {growth_mad_percentile!r}"
        )
    return p


def _normalize_growth_layer_score(growth_layer_score):
    if growth_layer_score == "mean":
        return "mean"
    if growth_layer_score == "mad":
        return "mad"
    raise ValueError(
        f"growth_layer_score must be 'mean' or 'mad', got {growth_layer_score!r}"
    )


def _robust_mad_zscores_by_layer(raw_scores_by_layer, eps=1e-8):
    """
    Per-layer robust z-scores using median/MAD.

    Returns dict layer_idx -> list of robust z-scores (one per neuron).
    """
    mad_scale = 1.4826
    by_layer = {}
    for layer_idx, raw_scores in raw_scores_by_layer.items():
        n_neurons = len(raw_scores)
        if n_neurons == 0:
            by_layer[layer_idx] = []
            continue
        if n_neurons == 1:
            by_layer[layer_idx] = [0.0]
            continue
        layer_median = _median(raw_scores)
        abs_deviations = [abs(score - layer_median) for score in raw_scores]
        mad = _median(abs_deviations)
        scaled_mad = mad_scale * mad + eps
        by_layer[layer_idx] = [
            (score - layer_median) / scaled_mad for score in raw_scores
        ]
    return by_layer


def _hidden_neuron_snr_mad_scores(raw_scores_by_layer, eps=1e-8):
    """
    Per-layer robust z-score from precomputed raw SNR scores using median/MAD.

    raw_scores_by_layer: dict layer_idx -> list of raw SNR per neuron.
    Returns dict (layer_idx, neuron_idx) -> robust z where lower is worse within layer.
    MAD is scaled by 1.4826 so it is comparable to std under normality.
    """
    mad_scores = {}
    by_layer = _robust_mad_zscores_by_layer(raw_scores_by_layer, eps=eps)
    for layer_idx, zscores in by_layer.items():
        for j, z in enumerate(zscores):
            mad_scores[(layer_idx, j)] = z
    return mad_scores


def _collect_hidden_neuron_uncertainty_scores(plasticity_model, uncertainty_combine="geometric"):
    """Return dict layer_idx -> list of raw bidirectional uncertainty per neuron."""
    raw_by_layer = {}
    for layer_idx, layer in enumerate(plasticity_model.layers):
        n_neurons = layer.mu_w.shape[0]
        raw_by_layer[layer_idx] = [
            float(
                plasticity_model._neuron_uncertainty_bidirectional(
                    layer_idx, j, combine=uncertainty_combine
                ).item()
            )
            for j in range(n_neurons)
        ]
    return raw_by_layer


def _layer_growth_scores_from_mad(raw_by_layer, percentile=100.0):
    """Layer growth scores from within-layer MAD z-scores at the given percentile."""
    percentile = _validate_growth_mad_percentile(percentile)
    by_layer_z = _robust_mad_zscores_by_layer(raw_by_layer)
    return {
        layer_idx: _percentile(zscores, percentile)
        for layer_idx, zscores in by_layer_z.items()
        if zscores
    }


def _hidden_neuron_snr_raw(raw_scores_by_layer):
    """
    Raw bidirectional SNR per neuron (no within-layer normalization).

    Returns dict (layer_idx, neuron_idx) -> snr; lower values are pruned first
    when used for global ranking.
    """
    scores = {}
    for layer_idx, raw_scores in raw_scores_by_layer.items():
        for j, score in enumerate(raw_scores):
            scores[(layer_idx, j)] = score
    return scores


def _layer_normalized_global_snr_scores(raw_scores_by_layer, global_prune_normalize="percentile"):
    """
    Per-layer scores for cross-layer global SNR ranking (lower = prune first).

    global_prune_normalize:
      - "percentile": rank / (n - 1) in [0, 1] (order-only within layer).
      - "zscore": (snr - layer_mean) / layer_std (magnitude-aware within layer).
      - "mad": (snr - layer_median) / scaled_MAD (robust within-layer outlier score).
      - "raw": use raw bidirectional SNR globally (lowest SNR pruned first).
    """
    if global_prune_normalize == "percentile":
        return _hidden_neuron_snr_percentile_ranks(raw_scores_by_layer), "layer_percentile"
    if global_prune_normalize == "zscore":
        return _hidden_neuron_snr_zscores(raw_scores_by_layer), "layer_zscore"
    if global_prune_normalize == "mad":
        return _hidden_neuron_snr_mad_scores(raw_scores_by_layer), "layer_mad"
    if global_prune_normalize == "raw":
        return _hidden_neuron_snr_raw(raw_scores_by_layer), "global_raw_snr"
    raise ValueError(
        f"global_prune_normalize must be 'percentile', 'zscore', 'mad', or 'raw', "
        f"got {global_prune_normalize!r}"
    )


def _collect_hidden_neuron_snr_scores(plasticity_model, snr_combine="geometric"):
    """Return dict layer_idx -> list of raw bidirectional SNR per neuron."""
    raw_by_layer = {}
    for layer_idx, layer in enumerate(plasticity_model.layers):
        n_neurons = layer.mu_w.shape[0]
        raw_by_layer[layer_idx] = [
            float(
                _neuron_snr_bidirectional(
                    plasticity_model, layer_idx, j, combine=snr_combine
                ).item()
            )
            for j in range(n_neurons)
        ]
    return raw_by_layer


def _print_neuron_snr_by_layer(plasticity_model, snr_combine="geometric"):
    """Print raw bidirectional SNR for every neuron in every hidden layer."""
    raw_by_layer = _collect_hidden_neuron_snr_scores(plasticity_model, snr_combine)
    print(f"\nNeuron bidirectional SNR (combine={snr_combine}):")
    for layer_idx in sorted(raw_by_layer.keys()):
        scores = raw_by_layer[layer_idx]
        n_neurons = len(scores)
        if n_neurons == 0:
            print(f"  Layer {layer_idx + 1} (0 neurons)")
            continue
        mean = sum(scores) / n_neurons
        variance = sum((s - mean) ** 2 for s in scores) / n_neurons
        std = math.sqrt(variance)
        print(
            f"  Layer {layer_idx + 1} ({n_neurons} neurons, "
            f"mean={mean:.6f}, std={std:.6f}):"
        )
        for j, s in enumerate(scores):
            print(f"    neuron {j:>3}: {s:.6f}")


def _prune_stats_from_keep_dict(plasticity_model, keep_dict, extra=None):
    """Build prune_stats with per-layer neuron removal counts."""
    neurons_removed_per_layer = {}
    total_removed = 0
    for layer_idx, layer in enumerate(plasticity_model.layers):
        old_n = layer.mu_w.shape[0]
        new_n = len(keep_dict[layer_idx])
        removed = old_n - new_n
        neurons_removed_per_layer[layer_idx] = removed
        total_removed += removed
    stats = {
        "neurons_removed_per_layer": neurons_removed_per_layer,
        "neurons_pruned": total_removed,
    }
    if extra:
        stats.update(extra)
    return stats


def _print_prune_candidate_summary(plasticity_model, keep_dict, prune_mode, prune_rate):
    """Log how many neurons would be removed from each hidden layer."""
    print(f"\nPruning candidate (mode={prune_mode}, rate={prune_rate}):")
    for layer_idx, layer in enumerate(plasticity_model.layers):
        old_n = layer.mu_w.shape[0]
        new_n = len(keep_dict[layer_idx])
        removed = old_n - new_n
        if removed > 0:
            print(f"  Layer {layer_idx + 1}: removing {removed} neurons ({old_n} -> {new_n})")
        else:
            print(f"  Layer {layer_idx + 1}: no neurons removed ({old_n})")


def _neuroapoptosis_per_layer(
    plasticity_model,
    prune_rate,
    exclude=None,
    snr_combine="geometric",
    min_neurons_per_layer=2,
):
    """
    Per-layer structured pruning: prune bottom prune_rate fraction *within each layer*
    by bidirectional SNR.

    Returns (keep_dict, prune_stats) or (None, {}) if no pruning occurred.
    """
    if exclude is None:
        exclude = []

    keep_dict = {}

    for i, layer in enumerate(plasticity_model.layers):
        n_neurons = layer.mu_w.shape[0]

        if i in exclude:
            keep_dict[i] = list(range(n_neurons))
            continue

        if n_neurons <= min_neurons_per_layer:
            keep_dict[i] = list(range(n_neurons))
            continue

        scores = []
        for j in range(n_neurons):
            s = _neuron_snr_bidirectional(plasticity_model, i, j, combine=snr_combine)
            scores.append(float(s.item()))

        num_prune = int(prune_rate * n_neurons)
        if num_prune < 1:
            keep_dict[i] = list(range(n_neurons))
            continue
        max_prune_allowed = n_neurons - min_neurons_per_layer
        if max_prune_allowed <= 0:
            keep_dict[i] = list(range(n_neurons))
            continue
        num_prune = min(num_prune, max_prune_allowed)

        neuron_indices = list(range(n_neurons))
        neuron_indices.sort(key=lambda j: scores[j])
        pruned = set(neuron_indices[:num_prune])

        kept = [j for j in range(n_neurons) if j not in pruned]
        if len(kept) < min_neurons_per_layer:
            return None, {}

        keep_dict[i] = kept

    if all(
        len(keep_dict[i]) == plasticity_model.layers[i].mu_w.shape[0]
        for i in range(len(plasticity_model.layers))
    ):
        return None, {}

    if any(len(v) == 0 for v in keep_dict.values()):
        return None, {}

    prune_stats = _prune_stats_from_keep_dict(
        plasticity_model, keep_dict, extra={"snr_normalize": "per_layer"}
    )
    return keep_dict, prune_stats


def _neuroapoptosis_global_param(
    plasticity_model,
    prune_rate,
    exclude=None,
    snr_combine="geometric",
    min_neurons_per_layer=2,
    global_prune_budget="params",
    global_prune_normalize="percentile",
):
    """
    Global SNR pruning with layer-normalized ranking: each neuron's SNR is converted
    to a within-layer score (percentile rank or z-score), then all neurons are ranked
    globally and the lowest-scoring are pruned until roughly prune_rate fraction of
    the chosen budget is consumed.

    global_prune_budget:
      - "params": stop after removing ~prune_rate * total variational parameters.
      - "neurons": stop after removing ~prune_rate * total hidden neurons.

    global_prune_normalize:
      - "percentile": within-layer rank in [0, 1] (order only).
      - "zscore": within-layer standard deviations below the mean (SNR gap aware).
      - "mad": within-layer robust z-score using median/MAD (outlier-resistant).
      - "raw": rank all neurons globally by raw bidirectional SNR (no layer norm).

    Returns (keep_dict, prune_stats) or (None, {}) if no pruning occurred.
    """
    if global_prune_budget not in ("params", "neurons"):
        raise ValueError(
            f"global_prune_budget must be 'params' or 'neurons', got {global_prune_budget!r}"
        )
    if exclude is None:
        exclude = []

    if global_prune_budget == "neurons":
        budget_total = sum(layer.mu_w.shape[0] for layer in plasticity_model.layers)
    else:
        budget_total = count_params(plasticity_model)
    target_remove = int(round(prune_rate * budget_total))
    if target_remove < 1:
        return None, {}

    raw_scores_by_layer = _collect_hidden_neuron_snr_scores(plasticity_model, snr_combine)
    norm_scores, snr_normalize = _layer_normalized_global_snr_scores(
        raw_scores_by_layer, global_prune_normalize
    )

    candidates = []
    for layer_idx, layer in enumerate(plasticity_model.layers):
        n_neurons = layer.mu_w.shape[0]
        if layer_idx in exclude or n_neurons <= min_neurons_per_layer:
            continue
        param_cost = _hidden_neuron_param_cost(plasticity_model, layer_idx)
        for neuron_idx in range(n_neurons):
            snr = raw_scores_by_layer[layer_idx][neuron_idx]
            norm_score = norm_scores[(layer_idx, neuron_idx)]
            candidates.append((norm_score, snr, layer_idx, neuron_idx, param_cost))

    if not candidates:
        return None, {}

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    pruned_per_layer = {i: set() for i in range(len(plasticity_model.layers))}
    params_removed = 0
    neurons_pruned = 0
    budget_consumed = 0
    unit_cost = 1 if global_prune_budget == "neurons" else None

    for norm_score, snr, layer_idx, neuron_idx, param_cost in candidates:
        step_cost = unit_cost if unit_cost is not None else param_cost
        if budget_consumed + step_cost > target_remove:
            continue
        n_neurons = plasticity_model.layers[layer_idx].mu_w.shape[0]
        n_pruned = len(pruned_per_layer[layer_idx])
        if n_neurons - n_pruned <= min_neurons_per_layer:
            continue
        pruned_per_layer[layer_idx].add(neuron_idx)
        params_removed += param_cost
        neurons_pruned += 1
        budget_consumed += step_cost

    if neurons_pruned == 0:
        return None, {}

    keep_dict = {}
    for layer_idx, layer in enumerate(plasticity_model.layers):
        n_neurons = layer.mu_w.shape[0]
        pruned = pruned_per_layer[layer_idx]
        kept = [j for j in range(n_neurons) if j not in pruned]
        if len(kept) < min_neurons_per_layer:
            return None, {}
        keep_dict[layer_idx] = kept

    prune_stats = _prune_stats_from_keep_dict(
        plasticity_model,
        keep_dict,
        extra={
            "target_remove": target_remove,
            "params_removed": params_removed,
            "snr_normalize": snr_normalize,
            "budget_unit": global_prune_budget,
        },
    )
    return keep_dict, prune_stats


def neuroapoptosis(
    plasticity_model,
    prune_rate,
    exclude=None,
    snr_combine="geometric",
    min_neurons_per_layer=0,
    prune_mode="per_layer",
    global_prune_budget="params",
    global_prune_normalize="percentile",
):
    """
    Structured pruning by bidirectional SNR.

    prune_mode:
      - "per_layer": prune int(prune_rate * n) neurons within each layer independently.
      - "global_param": rank all hidden neurons globally by layer-normalized SNR;
        prune lowest until roughly prune_rate fraction of the budget is consumed.

    global_prune_budget (only for prune_mode="global_param"):
      - "params": budget is prune_rate * total variational parameters.
      - "neurons": budget is prune_rate * total hidden neurons.

    global_prune_normalize (only for prune_mode="global_param"):
      - "percentile": within-layer rank in [0, 1].
      - "zscore": within-layer z-score (magnitude-aware).
      - "mad": within-layer robust z-score using median/MAD.
      - "raw": global rank by raw bidirectional SNR (lowest SNR pruned first).

    Returns (keep_dict, prune_stats). keep_dict is None if no pruning occurred.
    """
    if prune_mode == "per_layer":
        keep_dict, prune_stats = _neuroapoptosis_per_layer(
            plasticity_model,
            prune_rate,
            exclude=exclude,
            snr_combine=snr_combine,
            min_neurons_per_layer=min_neurons_per_layer,
        )
    elif prune_mode == "global_param":
        keep_dict, prune_stats = _neuroapoptosis_global_param(
            plasticity_model,
            prune_rate,
            exclude=exclude,
            snr_combine=snr_combine,
            min_neurons_per_layer=min_neurons_per_layer,
            global_prune_budget=global_prune_budget,
            global_prune_normalize=global_prune_normalize,
        )
    else:
        raise ValueError(
            f"prune_mode must be 'per_layer' or 'global_param', got {prune_mode!r}"
        )
    if keep_dict is not None:
        _print_prune_candidate_summary(
            plasticity_model, keep_dict, prune_mode, prune_rate
        )
    return keep_dict, prune_stats


def _mask_warm_start_grads_grow_new_only(model, layer_idx, old_width):
    """
    Growth warm-start gradient mask (new-only).

    Allows updates only to:
    - grown layer's new neurons: rows [old_width:]
    - its LayerNorm entries: indices [old_width:]
    - immediate downstream weights connected to new neurons: columns [old_width:]

    Everything else is frozen (gradients zeroed).
    """
    if layer_idx is None or old_width is None:
        raise ValueError("layer_idx and old_width must be provided for grow_new_only masking")
    if old_width <= 0:
        return

    def _zero_grad_full(p):
        if p is not None and getattr(p, "grad", None) is not None:
            p.grad.zero_()

    def _zero_grad_rows(p, n_rows):
        if p is not None and getattr(p, "grad", None) is not None:
            p.grad[:n_rows].zero_()

    def _zero_grad_cols(p, n_cols):
        if p is not None and getattr(p, "grad", None) is not None:
            p.grad[:, :n_cols].zero_()

    num_hidden = len(model.layers)

    # Hidden layers
    for i, layer in enumerate(model.layers):
        if i == layer_idx:
            # Freeze old neurons in grown layer; allow new neuron rows [old_width:].
            _zero_grad_rows(getattr(layer, "mu_w", None), old_width)
            _zero_grad_rows(getattr(layer, "rho_w", None), old_width)
            _zero_grad_rows(getattr(layer, "mu_b", None), old_width)
            _zero_grad_rows(getattr(layer, "rho_b", None), old_width)
        elif i == layer_idx + 1:
            # Allow adapting fan-out from new neurons only (columns [old_width:]).
            _zero_grad_cols(getattr(layer, "mu_w", None), old_width)
            _zero_grad_cols(getattr(layer, "rho_w", None), old_width)
            # Keep next-layer biases fixed (output-neuron biases) under new-only policy.
            _zero_grad_full(getattr(layer, "mu_b", None))
            _zero_grad_full(getattr(layer, "rho_b", None))
        else:
            # Completely freeze unrelated layers.
            _zero_grad_full(getattr(layer, "mu_w", None))
            _zero_grad_full(getattr(layer, "rho_w", None))
            _zero_grad_full(getattr(layer, "mu_b", None))
            _zero_grad_full(getattr(layer, "rho_b", None))

    # LayerNorm layers
    for i, ln in enumerate(getattr(model, "ln_layers", [])):
        if i == layer_idx:
            _zero_grad_rows(getattr(ln, "weight", None), old_width)
            _zero_grad_rows(getattr(ln, "bias", None), old_width)
        else:
            _zero_grad_full(getattr(ln, "weight", None))
            _zero_grad_full(getattr(ln, "bias", None))

    # Output layer
    if layer_idx == num_hidden - 1:
        # Growing last hidden layer: allow adapting out weights connected to new neurons.
        _zero_grad_cols(getattr(model.out, "mu_w", None), old_width)
        _zero_grad_cols(getattr(model.out, "rho_w", None), old_width)
        _zero_grad_full(getattr(model.out, "mu_b", None))
        _zero_grad_full(getattr(model.out, "rho_b", None))
    else:
        _zero_grad_full(getattr(model.out, "mu_w", None))
        _zero_grad_full(getattr(model.out, "rho_w", None))
        _zero_grad_full(getattr(model.out, "mu_b", None))
        _zero_grad_full(getattr(model.out, "rho_b", None))


def warm_start_model_on_batches(
    model,
    batches_ws,
    K,
    eta_ws,
    beta_scaled,
    lambda_penalty,
    mask_mode= None,
    layer_idx= None,
    old_width= None,
    grow_new_only_steps=None,
):
    """
    K gradient steps cycling through a list of warm-start batches.
    Trains all params (as in your current warm_start_model), but on multiple batches.

    When mask_mode == "grow_new_only", the new-neuron gradient mask is applied only for
    the first grow_new_only_steps steps (default: all K steps). Remaining steps update
    all parameters with no mask.
    """
    K = int(K)
    if grow_new_only_steps is None:
        new_only_limit = K
    else:
        new_only_limit = int(grow_new_only_steps)
        if new_only_limit < 0 or new_only_limit > K:
            raise ValueError(
                f"grow_new_only_steps must be in [0, K={K}], got {grow_new_only_steps!r}"
            )

    optimizer = optim.Adam(model.parameters(), lr=eta_ws)
    model.train()
    for step in range(K):
        inputs, labels = batches_ws[step % len(batches_ws)]
        param_count = count_params(model) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, _, _, _ = penalised_loss_function(
            outputs, labels, model.kl_loss(), beta_scaled, lambda_penalty, param_count
        )
        loss.backward()
        use_new_only = mask_mode == "grow_new_only" and step < new_only_limit
        if use_new_only:
            _mask_warm_start_grads_grow_new_only(model, layer_idx=layer_idx, old_width=old_width)
        optimizer.step()

def truncate_and_load_encoder_layer(old_sd, keep_dict, new_layer):
    num_layers = len(keep_dict)
    new_sd = {}
    weight_keys = ["mu_w", "rho_w", "mu_b", "rho_b", "prior_mu_w", "prior_sigma_w", "prior_mu_b", "prior_sigma_b"]
    for i in range(num_layers):
        keep_i = keep_dict.get(i, None)
        keep_prev = keep_dict.get(i - 1, None)
        for p in weight_keys:
            key = f"layers.{i}.{p}"
            if key not in old_sd:
                # Older checkpoints may lack prior buffers; skip and keep new-layer defaults.
                if p.startswith("prior_"):
                    continue
                raise ValueError(f"{key} is missing in the plasticity model")
            w = old_sd[key]
            if w.ndim == 2:
                if keep_i is not None:
                    w = w[keep_i, :]
                if keep_prev is not None:
                    w = w[:, keep_prev]
            # bias / 1D prior bias
            else:
                if keep_i is not None:
                    w = w[keep_i]
            new_sd[key] = w

        for p in ["weight", "bias"]:
            key = f"ln_layers.{i}.{p}"
            if key not in old_sd:
                raise ValueError(f"{key} is missing in the plasticity model")
            w = old_sd[key]
            if keep_i is not None:
                w = w[keep_i]
            new_sd[key] = w

    for p in weight_keys:
        key = f"out.{p}"
        if key not in old_sd:
            if p.startswith("prior_"):
                continue
            raise ValueError(f"{key} is missing in the plasticity model")
        w = old_sd[key]
        keep_last = keep_dict.get(num_layers - 1, None)
        if w.ndim == 2 and keep_last is not None:
            w = w[:, keep_last]
        new_sd[key] = w


    new_layer.load_state_dict(new_sd, strict=False)

def build_pruned_model(model, keep_dict):
    """Construct pruned BayesianFNN from keep_dict."""
    hidden_sizes = [len(v) for v in keep_dict.values()]
    device = next(model.parameters()).device
    pruned_model = BayesianFNN(model.in_features, hidden_sizes, model.out_features).to(device)
    truncate_and_load_encoder_layer(model.state_dict(), keep_dict, pruned_model)
    return pruned_model, hidden_sizes


def structural_decision_juncture(
    model,
    hidden_sizes,
    train_loader,
    val_loader,
    device,
    beta,
    lambda_penalty,
    gamma,
    rho,
    K,
    eta_ws,
    grow_exclude_layers=None,
    uncertainty_combine="geometric",
    growth_layer_score="mean",
    growth_mad_percentile=100.0,
    snr_combine="geometric",
    junctures_mode="both",
    prune_mode="per_layer",
    global_prune_budget="params",
    global_prune_normalize="percentile",
    grow_new_only_steps=None,
):
    """
    Evaluate growth and/or prune candidates via delta penalised ELBO on B_val.
    Deltas are relative to a warm-started baseline (same K steps as candidates).
    junctures_mode: "both" (grow+prune), "grow" (grow only), or "prune" (prune only).
    grow_new_only_steps: for grow warm-start, apply new-only mask for this many initial
    steps (default None = all K steps); remaining steps update all parameters.
    Returns (action, model, hidden_sizes, info_dict).
    """
    if junctures_mode not in ("both", "grow", "prune"):
        raise ValueError(
            f"junctures_mode must be 'both', 'grow', or 'prune', got {junctures_mode!r}"
        )
    train_dataset_size = len(train_loader.dataset)
    beta_scaled = (1 / train_dataset_size) * beta
    K = int(K)
    if grow_new_only_steps is None:
        resolved_grow_new_only_steps = K
    else:
        resolved_grow_new_only_steps = int(grow_new_only_steps)
        if resolved_grow_new_only_steps < 0 or resolved_grow_new_only_steps > K:
            raise ValueError(
                f"grow_new_only_steps must be in [0, K={K}], got {grow_new_only_steps!r}"
            )

    #_print_neuron_snr_by_layer(model, snr_combine=snr_combine)

    # choose how many batches
    M_ws = 32     # warm-start batches
    M_val = 20    # evaluation batches

    batches_ws = sample_batches(train_loader, device, M_ws)
    batches_val = sample_batches(val_loader, device, M_val)

    L_before = penalised_elbo_on_batches(model, batches_val, beta_scaled, lambda_penalty)

    none_model = copy.deepcopy(model)
    warm_start_model_on_batches(
        none_model,
        batches_ws,
        K,
        eta_ws,
        beta_scaled,
        lambda_penalty,
    )
    L_after_none = penalised_elbo_on_batches(
        none_model, batches_val, beta_scaled, lambda_penalty
    )

    info = {
        'L_before': L_before,
        'L_after_none': L_after_none,
        'delta_grow': None,
        'delta_prune': None,
        'param_count_before': count_params(model),
        'prune_mode': prune_mode,
        'global_prune_budget': global_prune_budget,
        'global_prune_normalize': global_prune_normalize,
        'prune_target_remove': None,
        'prune_params_removed': None,
        'prune_neurons_pruned': None,
        'growth_layer_score': growth_layer_score,
        'growth_mad_percentile': growth_mad_percentile,
        'grow_new_only_steps': resolved_grow_new_only_steps,
    }
    if grow_exclude_layers is None:
        grow_exclude_layers = []

    grow_model = None
    hidden_sizes_g = None
    layer_idx = None
    delta_grow = float('-inf')

    if junctures_mode in ("both", "grow"):
        grow_model, hidden_sizes_g, layer_idx, old_width = neurogenesis(
            model,
            hidden_sizes,
            exclude=list(grow_exclude_layers),
            gamma=gamma,
            uncertainty_combine=uncertainty_combine,
            growth_layer_score=growth_layer_score,
            growth_mad_percentile=growth_mad_percentile,
        )
        expand_and_load_encoder_layer(model.state_dict(), grow_model)
        warm_start_model_on_batches(
            grow_model,
            batches_ws,
            K,
            eta_ws,
            beta_scaled,
            lambda_penalty,
            mask_mode="grow_new_only",
            layer_idx=layer_idx,
            old_width=old_width,
            grow_new_only_steps=resolved_grow_new_only_steps,
        )
        L_after_grow = penalised_elbo_on_batches(
            grow_model, batches_val, beta_scaled, lambda_penalty
        )
        delta_grow = L_after_grow - L_after_none
        info['delta_grow'] = delta_grow

    # Prune candidate
    delta_prune = float('-inf')
    prune_model = None
    hidden_sizes_p = None
    keep_dict = None

    if junctures_mode in ("both", "prune"):
        keep_dict, prune_stats = neuroapoptosis(
            model,
            rho,
            snr_combine=snr_combine,
            prune_mode=prune_mode,
            min_neurons_per_layer=2,
            global_prune_budget=global_prune_budget,
            global_prune_normalize=global_prune_normalize,
        )
        if prune_stats:
            info['prune_target_remove'] = prune_stats.get('target_remove')
            info['prune_params_removed'] = prune_stats.get('params_removed')
            info['prune_neurons_pruned'] = prune_stats.get('neurons_pruned')
            info['prune_budget_unit'] = prune_stats.get('budget_unit', global_prune_budget)
            info['prune_snr_normalize'] = prune_stats.get('snr_normalize', global_prune_normalize)
        if keep_dict is not None:
            prune_model, hidden_sizes_p = build_pruned_model(model, keep_dict)
            warm_start_model_on_batches(
                prune_model,
                batches_ws,
                K,
                eta_ws,
                beta_scaled,
                lambda_penalty,
            )
            L_after_prune = penalised_elbo_on_batches(
                prune_model, batches_val, beta_scaled, lambda_penalty
            )
            delta_prune = L_after_prune - L_after_none
        info['delta_prune'] = delta_prune if keep_dict is not None else None

    best_action = 'none'
    best_model = model
    best_hidden_sizes = hidden_sizes

    delta_prune_val = delta_prune if keep_dict is not None else float('-inf')
    if junctures_mode == "both":
        if delta_grow > 0 or delta_prune_val > 0:
            if delta_grow >= delta_prune_val:
                best_action = 'grow'
                best_model = grow_model
                best_hidden_sizes = hidden_sizes_g
            else:
                best_action = 'prune'
                best_model = prune_model
                best_hidden_sizes = hidden_sizes_p
    elif junctures_mode == "grow" and delta_grow > 0:
        best_action = 'grow'
        best_model = grow_model
        best_hidden_sizes = hidden_sizes_g
    elif junctures_mode == "prune" and delta_prune_val > 0:
        best_action = 'prune'
        best_model = prune_model
        best_hidden_sizes = hidden_sizes_p

    info['grow_layer_idx'] = layer_idx
    info['grow_exclude_layers'] = list(grow_exclude_layers)
    info['junctures_mode'] = junctures_mode
    info['action'] = best_action
    info['param_count_after'] = count_params(best_model) if best_action != 'none' else info['param_count_before']
    delta_grow_str = f"{delta_grow:.4f}" if info['delta_grow'] is not None else "N/A"
    prune_log = ""
    if info.get('prune_neurons_pruned') is not None:
        budget_unit = info.get('prune_budget_unit', 'params')
        if budget_unit == "neurons":
            prune_log = (
                f", neuron_budget={info['prune_target_remove']}, "
                f"pruned={info['prune_neurons_pruned']} neurons "
                f"({info['prune_params_removed']} params)"
            )
        else:
            prune_log = (
                f", param_budget={info['prune_target_remove']}, "
                f"pruned={info['prune_params_removed']} params "
                f"({info['prune_neurons_pruned']} neurons)"
            )
    print(
        f"\nStructural decision (mode={junctures_mode}, prune_mode={prune_mode}, "
        f"global_prune_budget={global_prune_budget}, "
        f"global_prune_normalize={global_prune_normalize}, "
        f"grow_new_only_steps={resolved_grow_new_only_steps}/{K}): "
        f"L_before={L_before:.4f}, L_after_none={L_after_none:.4f}, "
        f"delta_grow={delta_grow_str}, delta_prune={info['delta_prune']}, "
        f"action={best_action}{prune_log}"
    )
    return best_action, best_model, best_hidden_sizes, info


def _hidden_sizes_from_model(model):
    return [layer.mu_w.shape[0] for layer in model.layers]


def build_grown_model_at_layer(model, hidden_sizes, growth_layer_idx, growth_gamma):
    """Construct a wider BayesianFNN by appending neurons to one hidden layer."""
    # The live model is the source of truth; callers may pass stale initial widths.
    hidden_sizes = _hidden_sizes_from_model(model)
    growth_layer_idx = int(growth_layer_idx)
    growth_gamma = float(growth_gamma)
    if growth_layer_idx < 0 or growth_layer_idx >= len(hidden_sizes):
        raise ValueError(
            f"growth_layer_idx must be in [0, {len(hidden_sizes) - 1}], got {growth_layer_idx}"
        )
    if growth_gamma <= 0:
        raise ValueError(f"growth_gamma must be positive, got {growth_gamma!r}")

    old_width = hidden_sizes[growth_layer_idx]
    neurons_to_add = max(1, math.ceil(growth_gamma * old_width))
    grown_hidden_sizes = hidden_sizes.copy()
    grown_hidden_sizes[growth_layer_idx] += neurons_to_add

    model_device = next(model.parameters()).device
    grown_model = BayesianFNN(
        model.in_features,
        grown_hidden_sizes,
        model.out_features,
    ).to(model_device)
    expand_and_load_encoder_layer(model.state_dict(), grown_model)
    return grown_model, grown_hidden_sizes, old_width, neurons_to_add


def build_three_phase_pruned_model(model, grown_hidden_sizes, growth_layer_idx, old_width):
    """Remove the appended neurons from the grown layer, keeping original neurons."""
    keep_dict = {}
    for layer_idx, width in enumerate(grown_hidden_sizes):
        if layer_idx == growth_layer_idx:
            keep_dict[layer_idx] = list(range(old_width))
        else:
            keep_dict[layer_idx] = list(range(width))
    return build_pruned_model(model, keep_dict)


def _initialise_standard_metrics():
    return {
        'train_loss_total': [],
        'train_loss_nll': [],
        'train_loss_kl': [],
        'train_loss_penalty': [],
        'train_acc': [],
        'train_brier': [],
        'val_loss_total': [],
        'val_loss_nll': [],
        'val_loss_kl': [],
        'val_loss_penalty': [],
        'val_acc': [],
        'val_brier': [],
        'param_count_history': [],
    }


def run_three_phase_baseline(
    experiment_name,
    model,
    hidden_sizes,
    train_loader,
    val_loader,
    test_loader,
    phase1_epochs,
    phase2_epochs,
    phase3_epochs,
    growth_layer_idx,
    growth_gamma,
    learning_rate=0.001,
    beta=0.1,
    lambda_penalty=0,
    output_dir=None,
    checkpoint_metric="val_loss_total",
    junctures_mode="three_phase",
):
    """Baseline with optional static train, deterministic growth, then deterministic removal."""
    if output_dir is None:
        output_dir = os.path.join('./results', experiment_name)
    ensure_output_dir(output_dir)

    phase1_epochs = int(phase1_epochs)
    phase2_epochs = int(phase2_epochs)
    phase3_epochs = int(phase3_epochs)
    if phase1_epochs < 0:
        raise ValueError("phase1_epochs must be non-negative")
    if phase2_epochs <= 0 or phase3_epochs <= 0:
        raise ValueError("phase2_epochs and phase3_epochs must be positive")
    total_epochs = phase1_epochs + phase2_epochs + phase3_epochs
    if total_epochs <= 0:
        raise ValueError("at least one phase must have a positive epoch count")

    hidden_sizes = _hidden_sizes_from_model(model)
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    loss_label = 'Penalised ELBO' if lambda_penalty > 0 else 'ELBO'

    print(f"\n{'-'*20} Running {experiment_name} experiment {'-'*20}")
    if phase1_epochs == 0:
        print(
            "Grow-prune refinement: "
            f"phase2={phase2_epochs}, phase3={phase3_epochs}, "
            f"growth_layer_idx={growth_layer_idx}, growth_gamma={growth_gamma}"
        )
    else:
        print(
            "Three-phase baseline: "
            f"phase1={phase1_epochs}, phase2={phase2_epochs}, phase3={phase3_epochs}, "
            f"growth_layer_idx={growth_layer_idx}, growth_gamma={growth_gamma}"
        )
    print(f"Initial hidden sizes: {hidden_sizes}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    metrics = _initialise_standard_metrics()
    metrics['phase_history'] = []
    metrics['hidden_size_history'] = []

    best_checkpoint_score = float("inf")
    best_model_state = None
    best_hidden_sizes = list(hidden_sizes)
    best_epoch = 0
    last_epoch = 0
    old_width = None
    neurons_to_add = 0
    phase = "phase1_static"

    for epoch in range(1, total_epochs + 1):
        if epoch == phase1_epochs + 1 and phase2_epochs > 0:
            print("-" * 20 + f" Three-phase growth boundary (epoch {epoch}) " + "-" * 20)
            hidden_sizes = _hidden_sizes_from_model(model)
            model, hidden_sizes, old_width, neurons_to_add = build_grown_model_at_layer(
                model,
                hidden_sizes,
                growth_layer_idx,
                growth_gamma,
            )
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            print(
                f"Added {neurons_to_add} neurons to hidden layer {growth_layer_idx} "
                f"using base width {old_width} "
                f"({old_width} -> {hidden_sizes[growth_layer_idx]})."
            )
        if epoch == phase1_epochs + phase2_epochs + 1 and phase3_epochs > 0:
            print("-" * 20 + f" Three-phase prune boundary (epoch {epoch}) " + "-" * 20)
            if old_width is None:
                raise RuntimeError("cannot enter phase 3 before phase 2 growth has occurred")
            model, hidden_sizes = build_three_phase_pruned_model(
                model,
                hidden_sizes,
                growth_layer_idx,
                old_width,
            )
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            print(
                f"Removed the {neurons_to_add} appended neurons from hidden layer "
                f"{growth_layer_idx}; hidden sizes restored to {hidden_sizes}."
            )

        if epoch <= phase1_epochs:
            phase = "phase1_static"
        elif epoch <= phase1_epochs + phase2_epochs:
            phase = "phase2_grown"
        else:
            phase = "phase3_pruned"

        last_epoch = epoch
        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty = train(
            model, train_loader, optimizer, epoch, device, beta_scaled, lambda_penalty
        )
        metrics['train_loss_total'].append(train_loss_total)
        metrics['train_loss_nll'].append(train_loss_nll)
        metrics['train_loss_kl'].append(train_loss_kl)
        metrics['train_loss_penalty'].append(train_loss_penalty)
        metrics['train_acc'].append(train_acc)
        metrics['train_brier'].append(train_brier)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics['val_loss_total'].append(val_loss_total)
        metrics['val_loss_nll'].append(val_loss_nll)
        metrics['val_loss_kl'].append(val_loss_kl)
        metrics['val_loss_penalty'].append(val_loss_penalty)
        metrics['val_acc'].append(val_acc)
        metrics['val_brier'].append(val_brier)
        metrics['param_count_history'].append(count_params(model))
        metrics['phase_history'].append(phase)
        metrics['hidden_size_history'].append(list(hidden_sizes))

        penalty_str = f', Penalty={train_loss_penalty:.4f}' if lambda_penalty > 0 else ''
        print(
            f'Epoch {epoch} ({phase}): Train Loss({loss_label})={train_loss_total:.4f}, '
            f'Train Loss(NLL)={train_loss_nll:.4f}, Train Loss(KL)={train_loss_kl:.4f}'
            f'{penalty_str}, Params={count_params(model):,}, Train Acc={train_acc:.2f}%, '
            f'Val Loss({loss_label})={val_loss_total:.4f}, Val Acc={val_acc:.2f}%'
        )

        # Only phase 3 (the final pruned architecture) is eligible for the best checkpoint.
        if phase == "phase3_pruned":
            val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
            if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
                best_checkpoint_score = val_score
                best_epoch = epoch
                best_hidden_sizes = list(hidden_sizes)
                best_model_state = copy.deepcopy(model.state_dict())
                save_checkpoint(
                    os.path.join(output_dir, "best_checkpoint.pth"),
                    state_dict=best_model_state,
                    epoch=best_epoch,
                    hidden_sizes=best_hidden_sizes,
                    selection_metric=checkpoint_metric_label,
                    selection_value=best_checkpoint_score,
                )

    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, 'metrics.png'),
    )

    if best_model_state is not None:
        eval_model = BayesianFNN(
            model.in_features,
            best_hidden_sizes,
            model.out_features,
        ).to(device)
        eval_model.load_state_dict(best_model_state, strict=True)
        print(
            f"Loaded best model from epoch {best_epoch} "
            f"(hidden_sizes={best_hidden_sizes}) for final testing."
        )
    else:
        eval_model = model

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f'Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}')

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else last_epoch,
        hidden_sizes=[layer.mu_w.shape[0] for layer in eval_model.layers],
    )

    final_hidden_sizes = [layer.mu_w.shape[0] for layer in eval_model.layers]
    metrics.update({
        'test_acc': test_acc,
        'test_loss_total': test_loss_total,
        'test_loss_nll': test_loss_nll,
        'test_loss_kl': test_loss_kl,
        'test_brier': test_brier,
        'param_count': count_params(eval_model),
        'trainable_param_count': eval_model.get_param_stats()['trainable_params'],
        'hidden_sizes': final_hidden_sizes,
    })

    # Restrict best-validation reporting to phase 3 epochs, matching checkpoint selection.
    phase3_val_acc = [
        acc for acc, ph in zip(metrics['val_acc'], metrics['phase_history'])
        if ph == "phase3_pruned"
    ]
    phase3_val_brier = [
        brier for brier, ph in zip(metrics['val_brier'], metrics['phase_history'])
        if ph == "phase3_pruned"
    ]

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        hidden_sizes=metrics["hidden_sizes"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(phase3_val_acc),
        best_val_brier=min(phase3_val_brier),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=lambda_penalty,
        selected_checkpoint_metric=checkpoint_metric_label,
        junctures_mode=junctures_mode,
    )

    phase_df = pd.DataFrame({
        'epoch': range(1, total_epochs + 1),
        'phase': metrics['phase_history'],
        'hidden_sizes': [str(hs) for hs in metrics['hidden_size_history']],
        'param_count': metrics['param_count_history'],
    })
    phase_df.to_csv(os.path.join(output_dir, 'phase_history.csv'), index=False)

    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy (phase 3): {max(phase3_val_acc):.2f}%")
    print(f"Best checkpoint score (phase 3): {best_checkpoint_score:.4f} at epoch {best_epoch}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Selected hidden sizes: {final_hidden_sizes}")
    print(f"Phase history written to {os.path.join(output_dir, 'phase_history.csv')}")

    return metrics, model, total_epochs


def _experiment_dir_suffix(junctures_mode):
    return "" if junctures_mode == "both" else f"_{junctures_mode}_only"


def _format_lambda_dir(lambda_penalty):
    return f"{float(lambda_penalty):g}"


def _resolve_plasticity_checkpoint_dir(resume_from_plasticity_dir):
    """Validate a plasticity results folder and return (dir, best_checkpoint.pth path)."""
    plasticity_dir = os.path.normpath(str(resume_from_plasticity_dir))
    if not os.path.isdir(plasticity_dir):
        raise FileNotFoundError(f"Plasticity directory not found: {plasticity_dir}")
    ckpt_path = os.path.join(plasticity_dir, "best_checkpoint.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"best_checkpoint.pth not found in plasticity directory: {plasticity_dir}"
        )
    return plasticity_dir, ckpt_path


def _load_plasticity_summary_metrics(plasticity_output_dir):
    """Load test metrics from a prior plasticity experiment_summary.csv if present."""
    summary_path = os.path.join(plasticity_output_dir, "experiment_summary.csv")
    if not os.path.exists(summary_path):
        return {}
    df = pd.read_csv(summary_path)
    if df.empty:
        return {}
    row = df.iloc[0]
    metrics = {}
    if "Test Acc" in row and not pd.isna(row["Test Acc"]):
        metrics["test_acc"] = float(row["Test Acc"])
    if "Parameters" in row and not pd.isna(row["Parameters"]):
        metrics["param_count"] = int(float(row["Parameters"]))
    if "Hidden Sizes" in row and not pd.isna(row["Hidden Sizes"]):
        metrics["hidden_sizes"] = row["Hidden Sizes"]
    return metrics


def _hybrid_output_dir_for_plasticity(plasticity_output_dir, save_path, init_width, lam_tag, suffix):
    """Place hybrid outputs beside the plasticity folder when resuming, else under save_path."""
    plasticity_dir_name = os.path.basename(plasticity_output_dir)
    if plasticity_dir_name.startswith("plasticity_"):
        hybrid_name = "plasticity_three_phase_" + plasticity_dir_name[len("plasticity_"):]
        return os.path.join(os.path.dirname(plasticity_output_dir), hybrid_name)
    return os.path.join(
        save_path,
        f"plasticity_three_phase_{init_width}_{lam_tag}{suffix}",
    )


def _resolve_plasticity_dir_for_static_replay(resume_from_plasticity_dir):
    """Validate a plasticity results folder and return (dir, experiment_summary.csv path)."""
    plasticity_dir = os.path.normpath(str(resume_from_plasticity_dir))
    if not os.path.isdir(plasticity_dir):
        raise FileNotFoundError(f"Plasticity directory not found: {plasticity_dir}")
    summary_path = os.path.join(plasticity_dir, "experiment_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"experiment_summary.csv not found in plasticity directory: {plasticity_dir}"
        )
    return plasticity_dir, summary_path


def _parse_hidden_sizes_from_summary(summary_path):
    """Parse Hidden Sizes from a plasticity experiment_summary.csv into a list of ints."""
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"Empty experiment summary: {summary_path}")
    if "Hidden Sizes" not in df.columns:
        raise ValueError(f"'Hidden Sizes' column missing in {summary_path}")
    raw = df.iloc[0]["Hidden Sizes"]
    if pd.isna(raw):
        raise ValueError(f"'Hidden Sizes' is empty in {summary_path}")
    sizes = ast.literal_eval(str(raw))
    if not isinstance(sizes, (list, tuple)) or not sizes:
        raise ValueError(f"Invalid Hidden Sizes value {raw!r} in {summary_path}")
    sizes = [int(s) for s in sizes]
    if any(s <= 0 for s in sizes):
        raise ValueError(f"Hidden Sizes must be positive ints, got {sizes} from {summary_path}")
    return sizes


def _static_replay_output_dir(plasticity_output_dir):
    """Place static-replay outputs beside the source plasticity folder."""
    plasticity_dir = os.path.normpath(plasticity_output_dir)
    plasticity_dir_name = os.path.basename(plasticity_dir)
    if not plasticity_dir_name.startswith("plasticity_"):
        raise ValueError(
            f"Expected plasticity_* directory name for static replay, got {plasticity_dir_name!r}"
        )
    return os.path.join(
        os.path.dirname(plasticity_dir),
        "static_replay_" + plasticity_dir_name[len("plasticity_"):],
    )


def write_static_replay_provenance(output_dir, provenance):
    """Write static-replay lineage metadata as JSON."""
    path = os.path.join(output_dir, "static_replay_provenance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    return path


def run_plasticity_then_three_phase(
    save_path,
    hidden_sizes,
    train_loader,
    val_loader,
    test_loader,
    lambda_penalty,
    num_epochs,
    learning_rate,
    beta,
    gamma,
    rho,
    warm_start_steps,
    warm_start_lr,
    decision_interval_min,
    decision_interval_max,
    decision_interval_power,
    decision_warmup_epochs,
    decision_cooldown_epochs,
    growth_layer_score,
    growth_mad_percentile,
    junctures_mode,
    prune_mode,
    global_prune_budget,
    global_prune_normalize,
    checkpoint_metric,
    grow_epochs,
    prune_epochs,
    growth_layer_idx,
    growth_gamma,
    resume_from_plasticity_dir=None,
    grow_new_only_steps=None,
):
    """
    Run plasticity, then grow-train-prune-train refinement on its best checkpoint.
    If resume_from_plasticity_dir is set, skip plasticity training and load
    best_checkpoint.pth from that existing results folder instead.
    Returns (plasticity_metrics, hybrid_metrics).
    """
    hidden_sizes = list(hidden_sizes)
    suffix = _experiment_dir_suffix(junctures_mode)
    lam_tag = _format_lambda_dir(lambda_penalty)
    init_width = hidden_sizes[0]

    if resume_from_plasticity_dir is not None:
        plasticity_output_dir, plasticity_ckpt_path = _resolve_plasticity_checkpoint_dir(
            resume_from_plasticity_dir
        )
        plasticity_metrics = _load_plasticity_summary_metrics(plasticity_output_dir)
        print(
            f"\nResuming hybrid refinement from existing plasticity run: "
            f"{plasticity_output_dir}"
        )
    else:
        plasticity_output_dir = os.path.join(
            save_path,
            f"plasticity_{init_width}_{lam_tag}{suffix}",
        )
        base_model = BayesianFNN(784, hidden_sizes, 10).to(device)
        plasticity_metrics, _, _ = run_adaptive_experiment(
            "plasticity",
            base_model,
            hidden_sizes,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            learning_rate,
            beta,
            lambda_penalty=lambda_penalty,
            gamma=gamma,
            rho=rho,
            warm_start_steps=warm_start_steps,
            warm_start_lr=warm_start_lr,
            decision_interval=None,
            decision_interval_min=decision_interval_min,
            decision_interval_max=decision_interval_max,
            decision_interval_power=decision_interval_power,
            decision_warmup_epochs=decision_warmup_epochs,
            decision_cooldown_epochs=decision_cooldown_epochs,
            output_dir=plasticity_output_dir,
            uncertainty_combine="mean",
            growth_layer_score=growth_layer_score,
            growth_mad_percentile=growth_mad_percentile,
            junctures_mode=junctures_mode,
            prune_mode=prune_mode,
            global_prune_budget=global_prune_budget,
            global_prune_normalize=global_prune_normalize,
            checkpoint_metric=checkpoint_metric,
            grow_new_only_steps=grow_new_only_steps,
        )
        plasticity_ckpt_path = os.path.join(plasticity_output_dir, "best_checkpoint.pth")
        if not os.path.exists(plasticity_ckpt_path):
            raise FileNotFoundError(
                f"Plasticity best checkpoint not found: {plasticity_ckpt_path}"
            )

    hybrid_output_dir = _hybrid_output_dir_for_plasticity(
        plasticity_output_dir,
        save_path,
        init_width,
        lam_tag,
        suffix,
    )

    refinement_model, refinement_hidden_sizes, ckpt_meta = load_checkpoint(
        plasticity_ckpt_path,
        in_features=784,
        out_features=10,
        device=device,
    )
    growth_layer_idx = int(growth_layer_idx)
    growth_gamma = float(growth_gamma)
    growth_base_hidden_sizes = _hidden_sizes_from_model(refinement_model)
    if growth_layer_idx < 0 or growth_layer_idx >= len(growth_base_hidden_sizes):
        raise ValueError(
            f"growth_layer_idx must be in [0, {len(growth_base_hidden_sizes) - 1}], "
            f"got {growth_layer_idx}"
        )
    growth_layer_width_at_growth = growth_base_hidden_sizes[growth_layer_idx]
    expected_neurons_to_add = max(1, math.ceil(growth_gamma * growth_layer_width_at_growth))

    hybrid_metrics, _, _ = run_three_phase_baseline(
        "plasticity_three_phase",
        refinement_model,
        refinement_hidden_sizes,
        train_loader,
        val_loader,
        test_loader,
        phase1_epochs=0,
        phase2_epochs=grow_epochs,
        phase3_epochs=prune_epochs,
        growth_layer_idx=growth_layer_idx,
        growth_gamma=growth_gamma,
        learning_rate=learning_rate,
        beta=beta,
        lambda_penalty=0,
        output_dir=hybrid_output_dir,
        checkpoint_metric=checkpoint_metric,
        junctures_mode="plasticity_three_phase",
    )

    provenance = {
        "plasticity_parent_dir": plasticity_output_dir,
        "plasticity_best_checkpoint": plasticity_ckpt_path,
        "plasticity_best_epoch": ckpt_meta.get("epoch"),
        "plasticity_hidden_sizes_before_refinement": refinement_hidden_sizes,
        "plasticity_test_acc": plasticity_metrics.get("test_acc"),
        "plasticity_param_count": plasticity_metrics.get("param_count"),
        "growth_base_hidden_sizes": growth_base_hidden_sizes,
        "growth_layer_idx": growth_layer_idx,
        "growth_layer_width_at_growth": growth_layer_width_at_growth,
        "growth_gamma": growth_gamma,
        "expected_neurons_to_add": expected_neurons_to_add,
        "grow_epochs": int(grow_epochs),
        "prune_epochs": int(prune_epochs),
        "hybrid_test_acc": hybrid_metrics.get("test_acc"),
        "hybrid_param_count": hybrid_metrics.get("param_count"),
        "hybrid_hidden_sizes": hybrid_metrics.get("hidden_sizes"),
        "resumed_from_plasticity_dir": resume_from_plasticity_dir is not None,
    }
    provenance_path = write_hybrid_provenance(hybrid_output_dir, provenance)
    print(f"Hybrid provenance written to {provenance_path}")

    return plasticity_metrics, hybrid_metrics


def run_experiment(experiment_name, model, train_loader, val_loader, test_loader, num_epochs,
                   learning_rate=0.001, start_epoch=1, metrics=None, beta=0.1,
                   lambda_penalty=0, output_dir=None, checkpoint_metric="val_loss_total",
                   junctures_mode="both", summary_lambda_penalty=None):
    """Run a complete training experiment and return metrics"""
    if output_dir is None:
        output_dir = os.path.join('./results', experiment_name)
    ensure_output_dir(output_dir)

    print(f"\n{'-'*20} Running {experiment_name} experiment {'-'*20}")
    
    # Display model parameters
    param_stats = model.get_param_stats() if hasattr(model, 'get_param_stats') else {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    print(f"Model parameters: {param_stats['total_params']:,}")
    print(f"Trainable parameters: {param_stats.get('trainable_params', param_stats['total_params']):,}")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Track metrics
    if not metrics:
        metrics = {}
        metrics['train_loss_total'] = []
        metrics['train_loss_nll'] = []
        metrics['train_loss_kl'] = []
        metrics['train_acc'] = []
        metrics['train_brier'] = []
        metrics['val_loss_total'] = []
        metrics['val_loss_nll'] = []
        metrics['val_loss_kl'] = []
        metrics['val_acc'] = []
        metrics['val_brier'] = []
        metrics['train_loss_penalty'] = []
        metrics['val_loss_penalty'] = []
        metrics['param_count_history'] = []

    loss_label = 'Penalised ELBO' if lambda_penalty > 0 else 'ELBO'
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    best_checkpoint_score = float("inf")
    best_model_state = None
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    csv_lambda_penalty = (
        float(summary_lambda_penalty)
        if summary_lambda_penalty is not None
        else float(lambda_penalty)
    )
    
    # Training loop
    last_epoch = best_epoch = start_epoch - 1
    for epoch in range(start_epoch, start_epoch + num_epochs):
        last_epoch = epoch
        
        # Train
        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty = train(
            model, train_loader, optimizer, epoch, device, beta_scaled, lambda_penalty
        )
        metrics['train_loss_total'].append(train_loss_total)
        metrics['train_loss_nll'].append(train_loss_nll)
        metrics['train_loss_kl'].append(train_loss_kl)
        metrics['train_acc'].append(train_acc)
        metrics['train_brier'].append(train_brier)
        metrics['train_loss_penalty'].append(train_loss_penalty)

        # Validate
        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics['val_loss_total'].append(val_loss_total)
        metrics['val_loss_nll'].append(val_loss_nll)
        metrics['val_loss_kl'].append(val_loss_kl)
        metrics['val_acc'].append(val_acc)
        metrics['val_brier'].append(val_brier)
        metrics['val_loss_penalty'].append(val_loss_penalty)
        metrics['param_count_history'].append(count_params(model))

        penalty_str = f', Penalty={train_loss_penalty:.4f}' if lambda_penalty > 0 else ''
        print(
            f'Epoch {epoch}: Train Loss({loss_label})={train_loss_total:.4f}, Train Loss(NLL)={train_loss_nll:.4f}, '
            f'Train Loss(KL)={train_loss_kl:.4f}{penalty_str}, Train Acc={train_acc:.2f}%, Train Brier={train_brier:.3f}, '
            f'Val Loss({loss_label})={val_loss_total:.4f}, Val Loss(NLL)={val_loss_nll:.4f}, '
            f'Val Loss(KL)={val_loss_kl:.4f}, Val Acc={val_acc:.2f}%, Val Brier={val_brier:.3f}'
        )
        
        val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
        if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
            best_checkpoint_score = val_score
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                hidden_sizes=[layer.mu_w.shape[0] for layer in model.layers],
                selection_metric=checkpoint_metric_label,
                selection_value=best_checkpoint_score,
            )
        
    # Plot and save metrics
    plot_metrics(
        {experiment_name: {
            'train_loss_total': metrics['train_loss_total'],
            'train_loss_nll': metrics['train_loss_nll'],
            'train_loss_kl': metrics['train_loss_kl'],
            'train_acc': metrics['train_acc'],
            'train_brier': metrics['train_brier'],
            'val_loss_total': metrics['val_loss_total'],
            'val_loss_nll': metrics['val_loss_nll'],
            'val_loss_kl': metrics['val_loss_kl'],
            'val_acc': metrics['val_acc'],
            'val_brier': metrics['val_brier']
        }}, 
        save_path=os.path.join(output_dir, 'metrics.png')
    )

    # Load best model for test
    best_model_for_eval = None
    if best_model_state is not None:
        best_model_for_eval = copy.deepcopy(model)
        best_model_for_eval.load_state_dict(best_model_state)
        print(f"Loaded best model from Epoch {best_epoch} based on validation loss for final testing.")

    eval_model = best_model_for_eval if best_model_for_eval is not None else model
    
    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f'Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}')
    
    # Save best checkpoint (same weights used for test eval)
    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else last_epoch,
        hidden_sizes=[layer.mu_w.shape[0] for layer in eval_model.layers],
    )

    # get final architecture
    hidden_sizes = []
    for i, layer in enumerate(eval_model.layers):
        hidden_sizes.append(layer.mu_w.shape[0])
    
    # Update metrics
    metrics.update({
        'test_acc': test_acc,
        'test_loss_total': test_loss_total,
        'test_loss_nll': test_loss_nll,
        'test_loss_kl': test_loss_kl,
        'test_brier': test_brier,
        'param_count': count_params(eval_model),
        'trainable_param_count': eval_model.get_param_stats()['trainable_params'],
        'hidden_sizes': hidden_sizes
    })
    
    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        hidden_sizes=metrics["hidden_sizes"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=csv_lambda_penalty,
        selected_checkpoint_metric=checkpoint_metric_label,
        junctures_mode=junctures_mode,
    )
    
    # Print summary
    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(metrics['val_acc'][start_epoch-1:]):.2f}%")
    print(f"Best validation loss: {min(metrics['val_loss_total'][start_epoch-1:]):.4f}")
    print(f"Best validation loss (NLL): {min(metrics['val_loss_nll'][start_epoch-1:]):.4f}")
    print(f"Best validation loss (KL): {min(metrics['val_loss_kl'][start_epoch-1:]):.4f}")
    print(f"Best validation brier: {min(metrics['val_brier'][start_epoch-1:]):.3f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final test loss: {test_loss_total:.4f}")
    print(f"Final test loss (NLL): {test_loss_nll:.4f}")
    print(f"Final test loss (KL): {test_loss_kl:.4f}")
    print(f"Final test brier: {test_brier:.3f}")
    
    epochs_run = last_epoch - start_epoch + 1
    return metrics, model, epochs_run


def _last_structural_epoch(num_epochs, decision_cooldown_epochs=0):
    """Last epoch at which a structural juncture may occur."""
    return max(0, num_epochs - max(0, int(decision_cooldown_epochs)))


def _annealed_decision_interval_length(
    epoch,
    num_epochs,
    decision_interval_min,
    decision_interval_max,
    decision_interval_power,
    decision_warmup_epochs=0,
    decision_cooldown_epochs=0,
):
    """Epochs until the next juncture; annealing is measured over the structural window."""
    structural_epochs = max(
        1,
        num_epochs - decision_warmup_epochs - max(0, int(decision_cooldown_epochs)),
    )
    frac = min(1.0, max(0.0, (epoch - decision_warmup_epochs) / structural_epochs))
    raw = (
        decision_interval_min
        + (decision_interval_max - decision_interval_min) * (frac ** decision_interval_power)
    )
    return max(1, int(math.floor(raw)))


def _initial_decision_epoch(
    num_epochs,
    decision_interval,
    decision_interval_min,
    decision_warmup_epochs=0,
):
    """First epoch at which a structural juncture may occur."""
    warmup = max(0, int(decision_warmup_epochs))
    if warmup >= num_epochs:
        return num_epochs
    if decision_interval is not None:
        return warmup + int(decision_interval)
    return warmup + decision_interval_min


def run_adaptive_experiment(
    experiment_name,
    model,
    hidden_sizes,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    learning_rate,
    beta,
    lambda_penalty,
    gamma,
    rho,
    warm_start_steps,
    warm_start_lr,
    decision_interval=None,
    decision_interval_min=2,
    decision_interval_max=20,
    decision_interval_power=2.0,
    decision_warmup_epochs=0,
    decision_cooldown_epochs=0,
    output_dir=None,
    growth_cooldown_junctures=0,
    uncertainty_combine="mean",
    growth_layer_score="mean",
    growth_mad_percentile=100.0,
    snr_combine="geometric",
    junctures_mode="both",
    prune_mode="per_layer",
    global_prune_budget="params",
    global_prune_normalize="percentile",
    checkpoint_metric="val_loss_total",
    grow_new_only_steps=None,
):
    """
    Dynamic structural adaptation via penalised ELBO.
    Structural decisions happen at annealed intervals (or fixed if decision_interval is set).
    No structural junctures occur for the first decision_warmup_epochs epochs or the
    last decision_cooldown_epochs epochs (weight-only convergence at the end).
    junctures_mode: "both" (grow+prune), "grow" (grow only), or "prune" (prune only).
    global_prune_budget: "params" or "neurons" (only used when prune_mode="global_param").
    global_prune_normalize: "percentile", "zscore", "mad", or "raw" (only used when prune_mode="global_param").
    growth_layer_score: "mean" or "mad" for which layer to expand on grow.
    growth_mad_percentile: percentile of within-layer MAD z-scores when growth_layer_score="mad".
    checkpoint_metric: "val_loss_nll" or "val_loss_total" for best-checkpoint selection.
    grow_new_only_steps: first N grow warm-start steps use new-only mask; rest update all
    (default None = all warm_start_steps are new-only).
    """
    if output_dir is None:
        output_dir = os.path.join('./results', experiment_name)
    ensure_output_dir(output_dir)

    print("\n" + "=" * 50)
    print(f"Training {experiment_name.upper()} (Penalised ELBO, lambda={lambda_penalty}, junctures_mode={junctures_mode})")
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    print(f"Checkpoint selection metric: {checkpoint_metric_label}")
    resolved_grow_new_only_steps = (
        warm_start_steps if grow_new_only_steps is None else int(grow_new_only_steps)
    )
    print(
        f"Grow warm-start: new_only_steps={resolved_grow_new_only_steps}/{warm_start_steps} "
        f"(remaining steps update all params)"
    )
    if decision_warmup_epochs > 0:
        print(f"Structural juncture warmup: {decision_warmup_epochs} epoch(s) of weight-only training")
    if decision_cooldown_epochs > 0:
        print(
            f"Structural juncture cooldown: {decision_cooldown_epochs} epoch(s) of "
            "weight-only convergence at the end"
        )
    print("=" * 50)

    param_stats = model.get_param_stats()
    print(f"Model parameters: {param_stats['total_params']:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    metrics = {
        'train_loss_total': [],
        'train_loss_nll': [],
        'train_loss_kl': [],
        'train_loss_penalty': [],
        'train_acc': [],
        'train_brier': [],
        'val_loss_total': [],
        'val_loss_nll': [],
        'val_loss_kl': [],
        'val_loss_penalty': [],
        'val_acc': [],
        'val_brier': [],
        'structural_epochs': [],
        'structural_actions': [],
        'structural_delta_grow': [],
        'structural_delta_prune': [],
        'structural_L_before': [],
        'structural_L_after_none': [],
        'structural_hidden_sizes': [],
        'structural_prune_mode': [],
        'structural_prune_target_remove': [],
        'structural_prune_params_removed': [],
        'structural_prune_neurons_pruned': [],
        'structural_global_prune_budget': [],
        'structural_global_prune_normalize': [],
        'structural_grow_new_only_steps': [],
        'param_count_history': [],
        'junctures_mode': junctures_mode,
    }

    growth_cooldown = {}

    best_checkpoint_score = float('inf')
    best_model_state = None
    best_epoch = 0
    best_hidden_sizes = list(hidden_sizes)
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    decision_warmup_epochs = max(0, int(decision_warmup_epochs))
    decision_cooldown_epochs = max(0, int(decision_cooldown_epochs))
    last_structural_epoch = _last_structural_epoch(num_epochs, decision_cooldown_epochs)
    if decision_warmup_epochs + decision_cooldown_epochs >= num_epochs:
        print(
            "Warning: decision_warmup_epochs + decision_cooldown_epochs >= num_epochs; "
            "no structural junctures will occur."
        )

    next_decision_epoch = min(
        last_structural_epoch,
        _initial_decision_epoch(
            num_epochs,
            decision_interval,
            decision_interval_min,
            decision_warmup_epochs,
        ),
    )

    for epoch in range(1, num_epochs + 1):
        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty = train(
            model, train_loader, optimizer, epoch, device, beta_scaled, lambda_penalty
        )
        metrics['train_loss_total'].append(train_loss_total)
        metrics['train_loss_nll'].append(train_loss_nll)
        metrics['train_loss_kl'].append(train_loss_kl)
        metrics['train_loss_penalty'].append(train_loss_penalty)
        metrics['train_acc'].append(train_acc)
        metrics['train_brier'].append(train_brier)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics['val_loss_total'].append(val_loss_total)
        metrics['val_loss_nll'].append(val_loss_nll)
        metrics['val_loss_kl'].append(val_loss_kl)
        metrics['val_loss_penalty'].append(val_loss_penalty)
        metrics['val_acc'].append(val_acc)
        metrics['val_brier'].append(val_brier)
        metrics['param_count_history'].append(count_params(model))

        print(
            f'Epoch {epoch}: Train Loss(Penalised ELBO)={train_loss_total:.4f}, Train Loss(NLL)={train_loss_nll:.4f}, '
            f'Train Loss(KL)={train_loss_kl:.4f}, Penalty={train_loss_penalty:.4f}, Params={count_params(model):,}, '
            f'Train Acc={train_acc:.2f}%, Val Loss(Penalised ELBO)={val_loss_total:.4f}, Val Acc={val_acc:.2f}%'
        )

        val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
        if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
            best_checkpoint_score = val_score
            best_epoch = epoch
            best_hidden_sizes = list(hidden_sizes)
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                hidden_sizes=best_hidden_sizes,
                selection_metric=checkpoint_metric_label,
                selection_value=best_checkpoint_score,
            )

        if (
            epoch == next_decision_epoch
            and epoch <= last_structural_epoch
            and epoch > decision_warmup_epochs
        ):
            print("-" * 20 + f" Decision juncture (epoch {epoch}) " + "-" * 20)
            exclude_layers = [i for i, rem in growth_cooldown.items() if rem > 0]
            if exclude_layers:
                print(f"Growth cooldown: excluding layer indices {exclude_layers} (0-based)")
            action, model, hidden_sizes, info = structural_decision_juncture(
                model,
                hidden_sizes,
                train_loader,
                val_loader,
                device,
                beta,
                lambda_penalty,
                gamma,
                rho,
                warm_start_steps,
                warm_start_lr,
                grow_exclude_layers=exclude_layers,
                uncertainty_combine=uncertainty_combine,
                growth_layer_score=growth_layer_score,
                growth_mad_percentile=growth_mad_percentile,
                snr_combine=snr_combine,
                junctures_mode=junctures_mode,
                prune_mode=prune_mode,
                global_prune_budget=global_prune_budget,
                global_prune_normalize=global_prune_normalize,
                grow_new_only_steps=grow_new_only_steps,
            )
            metrics['structural_epochs'].append(epoch)
            metrics['structural_actions'].append(action)
            metrics['structural_delta_grow'].append(info.get('delta_grow'))
            metrics['structural_delta_prune'].append(info.get('delta_prune'))
            metrics['structural_L_before'].append(info.get('L_before'))
            metrics['structural_L_after_none'].append(info.get('L_after_none'))
            metrics['structural_hidden_sizes'].append(list(hidden_sizes))
            metrics['structural_prune_mode'].append(info.get('prune_mode'))
            metrics['structural_prune_target_remove'].append(info.get('prune_target_remove'))
            metrics['structural_prune_params_removed'].append(info.get('prune_params_removed'))
            metrics['structural_prune_neurons_pruned'].append(info.get('prune_neurons_pruned'))
            metrics['structural_global_prune_budget'].append(info.get('global_prune_budget'))
            metrics['structural_global_prune_normalize'].append(
                info.get('global_prune_normalize')
            )
            metrics['structural_grow_new_only_steps'].append(
                info.get('grow_new_only_steps')
            )

            if action != 'none':
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
                )
        

            if action == "grow" and growth_cooldown_junctures > 0 and junctures_mode in ("both", "grow"):
                grown = info["grow_layer_idx"]
                growth_cooldown[grown] = growth_cooldown_junctures

                for i in list(growth_cooldown.keys()):
                    if i != grown: 
                        growth_cooldown[i] -= 1
                        if growth_cooldown[i] <= 0:
                            del growth_cooldown[i]

            # Schedule next decision juncture.
            if decision_interval is not None:
                next_decision_epoch = min(
                    last_structural_epoch, epoch + int(decision_interval)
                )
            else:
                gap = _annealed_decision_interval_length(
                    epoch,
                    num_epochs,
                    decision_interval_min,
                    decision_interval_max,
                    decision_interval_power,
                    decision_warmup_epochs,
                    decision_cooldown_epochs,
                )
                next_decision_epoch = min(last_structural_epoch, epoch + gap)

    # Final test evaluation with best checkpoint
    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, 'metrics.png'),
    )

    if best_model_state is not None:
        eval_model = BayesianFNN(
            model.in_features,
            best_hidden_sizes,
            model.out_features,
        ).to(device)
        eval_model.load_state_dict(best_model_state, strict=True)
        print(
            f"Loaded best model from epoch {best_epoch} "
            f"(hidden_sizes={best_hidden_sizes}) for final testing."
        )
    else:
        eval_model = model

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f'Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}')

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else num_epochs,
        hidden_sizes=[layer.mu_w.shape[0] for layer in eval_model.layers],
    )

    final_hidden_sizes = [layer.mu_w.shape[0] for layer in eval_model.layers]
    metrics.update({
        'test_acc': test_acc,
        'test_loss_total': test_loss_total,
        'test_loss_nll': test_loss_nll,
        'test_loss_kl': test_loss_kl,
        'test_brier': test_brier,
        'param_count': count_params(eval_model),
        'trainable_param_count': eval_model.get_param_stats()['trainable_params'],
        'hidden_sizes': final_hidden_sizes,
    })

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        hidden_sizes=metrics["hidden_sizes"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=lambda_penalty,
        selected_checkpoint_metric=checkpoint_metric_label,
        junctures_mode=junctures_mode,
    )


    structural_df = pd.DataFrame({
        'epoch': metrics['structural_epochs'],
        'action': metrics['structural_actions'],
        'delta_grow': metrics['structural_delta_grow'],
        'delta_prune': metrics['structural_delta_prune'],
        'L_before': metrics['structural_L_before'],
        'L_after_none': metrics['structural_L_after_none'],
        'hidden_sizes': [str(hs) for hs in metrics['structural_hidden_sizes']],
        'junctures_mode': metrics['junctures_mode'],
        'prune_mode': metrics['structural_prune_mode'],
        'prune_target_remove': metrics['structural_prune_target_remove'],
        'prune_params_removed': metrics['structural_prune_params_removed'],
        'prune_neurons_pruned': metrics['structural_prune_neurons_pruned'],
        'global_prune_budget': metrics['structural_global_prune_budget'],
        'global_prune_normalize': metrics['structural_global_prune_normalize'],
        'grow_new_only_steps': metrics['structural_grow_new_only_steps'],
    })
    structural_df.to_csv(os.path.join(output_dir, 'structural_decisions.csv'), index=False)

    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(metrics['val_acc']):.2f}%")
    print(f"Best validation total: {min(metrics['val_loss_total']):.4f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final hidden sizes: {final_hidden_sizes}")
    print(f"Structural decisions: {metrics['structural_actions']}")

    return metrics, model, num_epochs

def main(
    save_path,
    hidden_sizes,
    lambda_penalty,
    junctures_mode="both",
    checkpoint_metric="val_loss_total",
    growth_layer_score="mad",
    growth_mad_percentile=90.0,
    decision_cooldown_epochs=50,
    phase1_epochs=10,
    phase2_epochs=10,
    phase3_epochs=10,
    three_phase_growth_layer_idx=1,
    three_phase_growth_gamma=1,
    resume_from_plasticity_dir=None,
    dataset="fashion_mnist",
    run_mode="plasticity",
):
    """
    dataset options:
      - "fashion_mnist": Fashion-MNIST (default)
      - "kmnist": Kuzushiji-MNIST

    run_mode options:
      - "baseline": run only fixed-width baseline
      - "plasticity": run only plasticity (no baseline, no hybrid refinement)
      - "three_phase": run only three-phase baseline
      - "plasticity_and_hybrid": run plasticity and the hybrid refinement, but skip three-phase baseline
      - "hybrid": run only hybrid refinement (requires resume_from_plasticity_dir)
      - "static_replay": train a static FNN using Hidden Sizes from an existing
        plasticity experiment_summary.csv (requires resume_from_plasticity_dir)
    """
    allowed_run_modes = {
        "baseline",
        "plasticity",
        "three_phase",
        "plasticity_and_hybrid",
        "hybrid",
        "static_replay",
    }
    if run_mode not in allowed_run_modes:
        raise ValueError(
            f"run_mode must be one of {sorted(allowed_run_modes)}, got {run_mode!r}"
        )
    # Hyperparameters
    num_epochs = 50 if run_mode == "plasticity" else 30
    batch_size = 256
    learning_rate = 0.005
    beta = 0.002
    decision_interval_min = 1
    decision_interval_max = 1
    decision_interval_power = 1
    decision_warmup_epochs = 0
    decision_cooldown_epochs = 0

    gamma = 0.0
    rho = 0.1
    if three_phase_growth_gamma is None:
        three_phase_growth_gamma = gamma
    # gamma = 0.10
    # rho = 0.10
    prune_mode = "global_param"
    global_prune_budget = "neurons"  # "params" or "neurons"
    global_prune_normalize = "mad"  # "percentile", "zscore", "mad", or "raw"

    warm_start_steps = 32
    warm_start_lr = 0.002
    grow_new_only_steps = 8  
    
    # Create results directory
    os.makedirs(f'{save_path}', exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, batch_size=batch_size, seed=SEED
    )
    print(
        f"Dataset: {dataset} "
        f"(train={len(train_loader.dataset):,}, "
        f"val={len(val_loader.dataset):,}, "
        f"test={len(test_loader.dataset):,})"
    )

    if run_mode in ("baseline"):
        print("\n\n" + "="*50)
        print("Training Baseline Model")
        print("="*50)
        baseline_model = BayesianFNN(784, hidden_sizes, 10).to(device)
        initial_state_dict = copy.deepcopy(baseline_model.state_dict())
        baseline_output_dir = os.path.join(save_path, f'baseline_{hidden_sizes[0]}')
        baseline_metrics, _, _= run_experiment(
            'baseline', 
            baseline_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs, 
            learning_rate,
            start_epoch=1,
            beta=beta,
            output_dir=baseline_output_dir,
        )


    if run_mode in ("static_replay",):
        if resume_from_plasticity_dir is None:
            raise ValueError(
                "run_mode='static_replay' requires resume_from_plasticity_dir to be set"
            )
        plasticity_dir, summary_path = _resolve_plasticity_dir_for_static_replay(
            resume_from_plasticity_dir
        )
        replay_hidden_sizes = _parse_hidden_sizes_from_summary(summary_path)
        static_output_dir = _static_replay_output_dir(plasticity_dir)
        ensure_output_dir(static_output_dir)

        source_meta = _parse_experiment_dir_name(os.path.basename(plasticity_dir))
        source_lambda = source_meta.get("lambda_penalty")
        if source_lambda is None:
            source_lambda = float(lambda_penalty)

        print("\n\n" + "=" * 50)
        print("Training Static Replay of Plasticity Architecture")
        print("=" * 50)
        print(f"Source plasticity dir: {plasticity_dir}")
        print(f"Replayed hidden sizes: {replay_hidden_sizes}")
        print(f"Output dir: {static_output_dir}")

        write_static_replay_provenance(
            static_output_dir,
            {
                "source_plasticity_dir": plasticity_dir,
                "source_summary_csv": summary_path,
                "replayed_hidden_sizes": replay_hidden_sizes,
                "source_lambda_penalty": source_lambda,
                "training_lambda_penalty": 0.0,
            },
        )

        static_model = BayesianFNN(784, replay_hidden_sizes, 10).to(device)
        run_experiment(
            "static_replay",
            static_model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            learning_rate,
            start_epoch=1,
            beta=beta,
            lambda_penalty=0,
            output_dir=static_output_dir,
            checkpoint_metric=checkpoint_metric,
            junctures_mode="static_replay",
            summary_lambda_penalty=source_lambda,
        )


    if run_mode in ("three_phase"):
        # ========== Experiment 1b: Three-phase Baseline Model ==========
        three_phase_model = BayesianFNN(784, hidden_sizes, 10).to(device)
        three_phase_output_dir = os.path.join(
            save_path,
            f'three_phase_baseline_{hidden_sizes[0]}',
        )
        run_three_phase_baseline(
            "three_phase_baseline",
            three_phase_model,
            hidden_sizes,
            train_loader,
            val_loader,
            test_loader,
            phase1_epochs=phase1_epochs,
            phase2_epochs=phase2_epochs,
            phase3_epochs=phase3_epochs,
            growth_layer_idx=three_phase_growth_layer_idx,
            growth_gamma=three_phase_growth_gamma,
            learning_rate=learning_rate,
            beta=beta,
            lambda_penalty=0,
            output_dir=three_phase_output_dir,
            checkpoint_metric=checkpoint_metric,
        )

    if run_mode in ("plasticity"):
        # Plasticity only (no refinement): call the adaptive runner directly.
        suffix = _experiment_dir_suffix(junctures_mode)
        lam_tag = _format_lambda_dir(lambda_penalty)
        plasticity_output_dir = os.path.join(
            save_path, f"plasticity_{hidden_sizes[0]}_{lam_tag}{suffix}"
        )
        base_model = BayesianFNN(784, hidden_sizes, 10).to(device)
        run_adaptive_experiment(
            "plasticity",
            base_model,
            hidden_sizes,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            learning_rate,
            beta,
            lambda_penalty=lambda_penalty,
            gamma=gamma,
            rho=rho,
            warm_start_steps=warm_start_steps,
            warm_start_lr=warm_start_lr,
            decision_interval=None,
            decision_interval_min=decision_interval_min,
            decision_interval_max=decision_interval_max,
            decision_interval_power=decision_interval_power,
            decision_warmup_epochs=decision_warmup_epochs,
            decision_cooldown_epochs=decision_cooldown_epochs,
            output_dir=plasticity_output_dir,
            uncertainty_combine="mean",
            growth_layer_score=growth_layer_score,
            growth_mad_percentile=growth_mad_percentile,
            junctures_mode=junctures_mode,
            prune_mode=prune_mode,
            global_prune_budget=global_prune_budget,
            global_prune_normalize=global_prune_normalize,
            checkpoint_metric=checkpoint_metric,
            grow_new_only_steps=grow_new_only_steps,
        )

    if run_mode in ("plasticity_and_hybrid", "hybrid"):
        if run_mode == "hybrid" and resume_from_plasticity_dir is None:
            raise ValueError(
                "run_mode='hybrid' requires resume_from_plasticity_dir to be set"
            )
        # ========== Experiment 2: Plasticity + hybrid grow-prune refinement ==========
        run_plasticity_then_three_phase(
            save_path=save_path,
            hidden_sizes=hidden_sizes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lambda_penalty=lambda_penalty,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta=beta,
            gamma=gamma,
            rho=rho,
            warm_start_steps=warm_start_steps,
            warm_start_lr=warm_start_lr,
            decision_interval_min=decision_interval_min,
            decision_interval_max=decision_interval_max,
            decision_interval_power=decision_interval_power,
            decision_warmup_epochs=decision_warmup_epochs,
            decision_cooldown_epochs=decision_cooldown_epochs,
            growth_layer_score=growth_layer_score,
            growth_mad_percentile=growth_mad_percentile,
            junctures_mode=junctures_mode,
            prune_mode=prune_mode,
            global_prune_budget=global_prune_budget,
            global_prune_normalize=global_prune_normalize,
            checkpoint_metric=checkpoint_metric,
            grow_epochs=phase2_epochs,
            prune_epochs=phase3_epochs,
            growth_layer_idx=three_phase_growth_layer_idx,
            growth_gamma=three_phase_growth_gamma,
            resume_from_plasticity_dir=resume_from_plasticity_dir,
            grow_new_only_steps=grow_new_only_steps,
        )
    
    # # ========== Compare Results ==========
    # all_metrics = {
    #      'baseline': baseline_metrics,
    #     'plasticity': plasticity_metrics,

    # }
    # plot_metrics(all_metrics, save_path=f'./{save_path}/model_comparison.png')
    
    # Create summary table
    # summary = pd.DataFrame([
        # {
        #     'Model': 'Baseline',
        #     'Parameters': baseline_metrics['param_count'],
        #     'Trainable Params': baseline_metrics['trainable_param_count'],
        #     'Best Val Acc': max(baseline_metrics['val_acc']),
        #     'Best Val Brier': min(baseline_metrics['val_brier']),
        #     'Test Acc': baseline_metrics['test_acc'],
        #     'Test Brier': baseline_metrics['test_brier'],
        #     'Hidden Sizes': baseline_metrics['hidden_sizes'],
        # },
    #     {
    #         'Model': 'Plasticity',
    #         'Parameters': plasticity_metrics['param_count'],
    #         'Trainable Params': plasticity_metrics['trainable_param_count'],
    #         'Best Val Acc': max(plasticity_metrics['val_acc']),
    #         'Best Val Brier': min(plasticity_metrics['val_brier']),
    #         'Test Acc': plasticity_metrics['test_acc'],
    #         'Test Brier': plasticity_metrics['test_brier'],
    #         'Hidden Sizes': plasticity_metrics['hidden_sizes'],
    #     }
    # ])
    
    # summary.to_csv(f'./{save_path}/experiment_summary.csv', index=False)
    # print("\nExperiment Summary:")
    # print(summary)
    # return summary


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
    Returns dict with keys: kind, init_width, lambda_penalty, junctures_mode, label, dir_tags
    (plus nest_p / nest_ref_acc / nest_floor_acc for Nest folders).
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
            "other": {"color": "gray", "marker": "x"},
        }
    lambda_colors = _lambda_penalty_color_map(df, palette=lambda_palette)
    nest_floor_colors = _nest_floor_color_map(df, palette=lambda_palette)
    plasticity_modes = df.loc[df["kind"] == "plasticity", "junctures_mode"].dropna().unique()
    style_by_junctures = (
        style_by == "junctures_mode"
        and len(plasticity_modes) > 1
    )
    fig, ax = plt.subplots(figsize=figsize)
    grouped = None
    if not aggregate_runs:
        for _, row in df.iterrows():
            style = _point_style(
                row, style_map, lambda_colors, style_by_junctures, nest_floor_colors
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
                n_runs=("run", "nunique"),
            )
            .reset_index()
        )
        for _, row in grouped.iterrows():
            style = _point_style(
                row, style_map, lambda_colors, style_by_junctures, nest_floor_colors
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
                row, style_map, lambda_colors, style_by_junctures, nest_floor_colors
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

if __name__ == "__main__":
    # df1=get_statistics(save_path="results", experiments=["plasticity_16_1e-08","plasticity_32_1e-08","plasticity_64_1e-08","plasticity_128_1e-08", "plasticity_256_1e-08"], num_runs=5)
    # print(df1["plasticity_16_1e-08"]["summary"])
    # print(df1["plasticity_32_1e-08"]["summary"])
    # print(df1["plasticity_64_1e-08"]["summary"])
    # print(df1["plasticity_128_1e-08"]["summary"])
    # print(df1["plasticity_256_1e-08"]["summary"])





    # hidden_sizes = [500, 500]
    # for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #     for junctures_mode in ["both"]:
    #         for i in range(1, 6):
    #             set_seed(SEED+i)
    #             print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
    #             main(
    #                 f"results_FashionMnist_FNN_grow1_wstrain64_lr0.002_grownew64_wsval32/run_{i}",
    #                 hidden_sizes,
    #                 lambda_penalty,
    #                 junctures_mode=junctures_mode,
    #                 dataset="fashion_mnist",
    #                 run_mode="plasticity",
    #             )
    # print("All FashionMnist experiments completed.")

    # hidden_sizes = [500, 500]
    # for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #     for junctures_mode in ["prune"]:
    #         for i in range(1, 6):
    #             set_seed(SEED+i)
    #             print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
    #             main(
    #                 f"results_FashionMnist_FNN_grow1_ws32_lr0.002_grownew8/run_{i}",
    #                 hidden_sizes,
    #                 lambda_penalty,
    #                 junctures_mode=junctures_mode,
    #                 dataset="fashion_mnist",
    #                 run_mode="plasticity",
    #             )
    # print("All FashionMnist experiments completed.")



    # for hidden_size in [[75,75],[125,125],[175,175]]:
    #     for lambda_penalty in [0]:
    #         for junctures_mode in ["both"]:
    #             for i in range(1,6):
    #                 set_seed(SEED+i)
    #                 print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
    #                 main(
    #                     f"results_FashionMnist_FNN_grow1/run_{i}",
    #                     hidden_size,
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                     dataset="fashion_mnist",
    #                     run_mode="baseline",
    #                 )
    # print("All experiments completed.")



    # for hidden_size in [[75,75],[125,125],[175,175]]:
    #     for lambda_penalty in [1e-06]:
    #         for junctures_mode in ["both"]:
    #             for i in range(1,6):
    #                 set_seed(SEED+i)
    #                 print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
    #                 main(
    #                     f"results_FashionMnist_FNN_grow1/run_{i}",
    #                     hidden_size,
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                     dataset="fashion_mnist",
    #                     run_mode="three_phase",
    #                     three_phase_growth_layer_idx=1,
    #                     three_phase_growth_gamma=1,
    #                 )
    # print("All experiments completed.")


    # # Static replay: train a fixed FNN using Hidden Sizes from each plasticity run's summary.
    # for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #     for i in range(1, 6):
    #         set_seed(SEED+i)
    #         plasticity_dir = (
    #             f"results_FashionMnist_FNN_grow1_ws32_lr0.002_grownew8/run_{i}/plasticity_500_{_format_lambda_dir(lambda_penalty)}"
    #         )
    #         print(
    #             "Running static replay for run",
    #             i,
    #             f"(source={plasticity_dir})",
    #         )
    #         main(
    #             f"results_FashionMnist_FNN_grow1_ws32_lr0.002_grownew8/run_{i}",
    #             [500, 500],  # unused for architecture; taken from plasticity summary
    #             lambda_penalty=0,
    #             dataset="fashion_mnist",
    #             run_mode="static_replay",
    #             resume_from_plasticity_dir=plasticity_dir,
    #         )
    # print("All static replay experiments completed.")

    # hidden_sizes = [500, 500]
    # for lambda_penalty in [5e-06, 1e-06, 5e-07, 5e-08, 0]:
    #     for junctures_mode in ["prune"]:
    #         for i in range(1, 6):
    #             set_seed(SEED+i)
    #             print("Running FashionMnist experiment for run", i, f"(junctures_mode={junctures_mode})")
    #             main(
    #                 f"results_FashionMnist_FNN2/run_{i}",
    #                 hidden_sizes,
    #                 lambda_penalty,
    #                 junctures_mode=junctures_mode,
    #                 dataset="fashion_mnist",
    #                 run_mode="plasticity",
    #             )
    # print("All FashionMnist experiments completed.")



    # hidden_sizes = [200,200]
    # for r in [1]:
    #     for lambda_penalty in [5e-06]:
    #         for junctures_mode in ["both"]:
    #             set_seed(SEED)
    #             for i in range(1,6):
    #                 print("Running experiment for run", i)
    #                 main(
    #                     f"results_g01_r01_global_param_200_neuron_total_nocool_mad_300epochs/run_{i}",
    #                     [int(hidden_size * r) for hidden_size in hidden_sizes],
    #                     lambda_penalty,
    #                     phase1_epochs=100,
    #                     phase2_epochs=100,
    #                     phase3_epochs=100,
    #                     three_phase_growth_layer_idx=1,
    #                     three_phase_growth_gamma=1,
    #                     run_mode="three_phase",
    #                 )
    # print("All experiments completed.")


    # hidden_sizes = [200,200]
    # for r in [0.7]:
    #     for lambda_penalty in [1e-08]:
    #         for junctures_mode in ["both"]:
    #             set_seed(SEED)
    #             for i in range(1,6):
    #                 print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
    #                 main(
    #                     f"results_g02_r02_global_param_200_neuron_total_nocool_mad_300epochs_15j/run_{i}",
    #                     [int(hidden_size * r) for hidden_size in hidden_sizes],
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                 )
#     plot_param_count(
#     save_path="results_temp_maske_mode_none_bidirectional_unc",
#     experiments=[
#         "plasticity_32_5e-07","plasticity_128_5e-07","plasticity_256_5e-07",
#     ],
# )

    # plot_param_count_vs_test_acc(
    #     save_path="results_Kmnist_FNN",
    #     experiments=[
    #        "baseline_20","baseline_50","baseline_100","baseline_150","baseline_200",
    #         "plasticity_500_5e-06","plasticity_500_1e-06","plasticity_500_5e-07","plasticity_500_5e-08","plasticity_500_0",
    #         # "static_replay_400_5e-06","static_replay_400_1e-06","static_replay_400_5e-07","static_replay_400_1e-07","static_replay_400_5e-08","static_replay_400_5e-09",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_frontier_kinds=("baseline", "plasticity", "static_replay"),
    #     save_path_out="results_FashionMnist_FNN_static_replay.png",
    #     title="Test Acc vs Parameter Count",
    #     y_col="Test Acc",
    #     ylabel="Test Acc",
    # )

    plot_param_count_vs_test_acc(
        save_path="results_FashionMnist_FNN_grow1_ws32_lr0.002_grownew8",
        experiments=[
           "baseline_20","baseline_50","baseline_100","baseline_150","baseline_200",
            # "three_phase_baseline_20","three_phase_baseline_50","three_phase_baseline_100","three_phase_baseline_150","three_phase_baseline_200",
            "plasticity_500_5e-06","plasticity_500_1e-06","plasticity_500_5e-07","plasticity_500_1e-07","plasticity_500_0",
            # "plasticity_500_5e-06_prune_only","plasticity_500_1e-06_prune_only","plasticity_500_5e-07_prune_only","plasticity_500_1e-07_prune_only","plasticity_500_0_prune_only",
            # "static_replay_500_5e-06","static_replay_500_1e-06","static_replay_500_5e-07","static_replay_500_5e-08","static_replay_500_0",
            "nest_300_100_p0.1_refacc89_flooracc87",
            "nest_300_100_p0.1_refacc89_flooracc87.5",
            "nest_300_100_p0.1_refacc89_flooracc88",
            "nest_300_100_p0.1_refacc89_flooracc88.5",
            "nest_300_100_p0.1_refacc89_flooracc89",
        ],
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        pareto_scope="per_junctures_mode",
        pareto_frontier_kinds=("baseline", "three_phase", "plasticity", "plasticity_three_phase","static_replay", "nest"),
        save_path_out="results_FashionMnist_FNN_acc.png",
        title="Test Acc vs Parameter Count",
        y_col="Test Acc",
        ylabel="Test Acc",
    )


    plot_param_count_vs_test_acc(
        save_path="results_FashionMnist_FNN_grow1_ws32_lr0.002_grownew8",
        experiments=[
           "baseline_20","baseline_50","baseline_100","baseline_150","baseline_200",
            # "three_phase_baseline_20","three_phase_baseline_50","three_phase_baseline_100","three_phase_baseline_150","three_phase_baseline_200",
            "plasticity_500_5e-06","plasticity_500_1e-06","plasticity_500_5e-07","plasticity_500_1e-07","plasticity_500_0",
            # "plasticity_500_5e-06_prune_only","plasticity_500_1e-06_prune_only","plasticity_500_5e-07_prune_only","plasticity_500_1e-07_prune_only","plasticity_500_0_prune_only",
            # "static_replay_500_5e-06","static_replay_500_1e-06","static_replay_500_5e-07","static_replay_500_5e-08","static_replay_500_0",
            "nest_300_100_p0.1_refacc89_flooracc87",
            "nest_300_100_p0.1_refacc89_flooracc87.5",
            "nest_300_100_p0.1_refacc89_flooracc88",
            "nest_300_100_p0.1_refacc89_flooracc88.5",
            "nest_300_100_p0.1_refacc89_flooracc89",
        ],
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        pareto_scope="per_junctures_mode",
        pareto_frontier_kinds=("baseline", "three_phase", "plasticity", "plasticity_three_phase","static_replay", "nest"),
        save_path_out="results_FashionMnist_FNN_brier.png",
        title="Test Brier vs Parameter Count",
        y_col="Test Brier",
        ylabel="Test Brier",
    )

    # plot_param_count_vs_test_acc(
    #     save_path="results_FashionMnist_FNN3",
    #     experiments=[
    #         # "baseline_20", "baseline_50", "baseline_100", "baseline_150", "baseline_200",
    #         # small init
    #         "plasticity_20_5e-06", "plasticity_20_1e-06", "plasticity_20_5e-07", "plasticity_20_5e-08", "plasticity_20_0",
    #         # large init
    #         "plasticity_500_5e-06", "plasticity_500_1e-06", "plasticity_500_5e-07", "plasticity_500_5e-08", "plasticity_500_0",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     # annotate_points=True,  # helps tell 20 vs 500 apart
    #     show_pareto_frontier=True,
    #     pareto_scope="per_init_width",
    #     pareto_frontier_kinds=("baseline", "plasticity"),
    #     save_path_out="results_FashionMnist_FNN.png",
    #     title="Test Brier vs Parameter Count",
    #     y_col="Test Brier",
    #     ylabel="Test Brier",
    # )


#make pruning and growing rates dynamic