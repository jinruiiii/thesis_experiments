from BayesianFNN import BayesianFNN
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import copy
import pandas as pd
import matplotlib.pyplot as plt

# Set all random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Random seed set to: {SEED} for full reproducibility")

def seed_worker(worker_id):
    """Function to ensure DataLoader workers use different seeds derived from the base seed"""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
    nll, _, kl_scaled = loss_function(outputs, labels, kl_loss, beta)
    penalty = lambda_penalty * param_count
    total = nll + kl_scaled + penalty
    return total, nll, kl_scaled, penalty


def penalised_elbo_on_batch(model, inputs, labels, beta, lambda_penalty, dataset_size):
    """Penalised ELBO L_pen on a single batch (higher is better)."""
    model.eval()
    param_count = count_params(model)
    with torch.no_grad():
        outputs = model(inputs)
        loss, nll, kl_scaled, penalty = penalised_loss_function(
            outputs, labels, model.kl_loss(), beta, lambda_penalty, param_count
        )
    return -(loss.item())


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


def train(model, train_dataloader, optimizer, epoch, device, beta, lambda_penalty=0):
    model.train()
    running_loss_total = 0.0
    running_loss_nll = 0.0
    running_loss_kl = 0.0
    running_loss_penalty = 0.0
    running_brier = 0.0
    correct = 0
    total = 0
    param_count = count_params(model)
    beta_scaled = (1 / len(train_dataloader.dataset)) * beta

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

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


def validate(model, val_dataloader, device, beta, lambda_penalty=0):
    model.eval()
    val_loss_total = 0.0
    val_loss_nll = 0.0
    val_loss_kl = 0.0
    val_loss_penalty = 0.0
    running_brier = 0.0
    correct = 0
    total = 0
    param_count = count_params(model)
    beta_scaled = (1 / len(val_dataloader.dataset)) * beta

    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)

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


def neurogenesis(plasticity_original, hidden_sizes, exclude=None, gamma=0.1):
    """Growth candidate: expand layer l* with highest mean posterior variance U^(l)."""
    if exclude is None:
        exclude = []
    uncertainty = plasticity_original.get_average_uncertainty_per_layer()
    print("\n Average Uncertainty per Hidden Layer:")
    for i, val in enumerate(uncertainty):
        print(f"  Layer {i+1}: {val.item()/(hidden_sizes[i]**0.5):.6f}")
    layer_to_expand = max(
        (i for i in range(len(uncertainty)) if i not in exclude),
        key=lambda i: uncertainty[i]/(hidden_sizes[i]**0.5)
    )
    neurons_to_add = max(1, math.ceil(gamma * hidden_sizes[layer_to_expand]))
    old_width = hidden_sizes[layer_to_expand]
    print(f"Expanding Layer {layer_to_expand+1} "
            f"(Highest Normalised Uncertainty: {uncertainty[layer_to_expand].item()/(hidden_sizes[layer_to_expand]**0.5):.6f}) "
            f"by {neurons_to_add} neurons")
    )

    expanded_hidden_sizes = hidden_sizes.copy()
    expanded_hidden_sizes[layer_to_expand] += neurons_to_add
    model_device = next(plasticity_original.parameters()).device
    plasticity_neurogenesis = BayesianFNN(
        plasticity_original.in_features, expanded_hidden_sizes, plasticity_original.out_features
    ).to(model_device)
    return plasticity_neurogenesis, expanded_hidden_sizes

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

def _neuron_snr_incoming(layer, eps=1e-8):
    """SNR_j^(l) = mean_i |mu_ij / sigma_ij| over incoming weights only."""
    sigma = F.softplus(layer.rho_w)
    snr = torch.abs(layer.mu_w) / (sigma + eps)
    return snr.mean(dim=1)


def neuroapoptosis(plasticity_model, prune_rate, exclude=None):
    """
    Prune the bottom rho fraction of neurons globally by SNR.
    Returns keep_dict or None if pruning would empty any layer.
    """
    if exclude is None:
        exclude = []
    neuron_entries = []
    for i, layer in enumerate(plasticity_model.layers):
        if i in exclude:
            continue
        snr_per_neuron = _neuron_snr_incoming(layer)
        for j in range(snr_per_neuron.shape[0]):
            neuron_entries.append((i, j, snr_per_neuron[j].item()))

    if not neuron_entries:
        return None

    total_neurons = len(neuron_entries)
    num_prune = max(1, int(prune_rate * total_neurons))
    neuron_entries.sort(key=lambda x: x[2])
    to_prune = set((layer_idx, neuron_idx) for layer_idx, neuron_idx, _ in neuron_entries[:num_prune])

    keep_dict = {}
    print("\n Neurons Pruned from Each Hidden Layer:")
    for i, layer in enumerate(plasticity_model.layers):
        n_neurons = layer.mu_w.shape[0]
        if i in exclude:
            keep_dict[i] = list(range(n_neurons))
        else:
            keep_dict[i] = [j for j in range(n_neurons) if (i, j) not in to_prune]
        print(f"Hidden Layer {i+1}: {n_neurons - len(keep_dict[i])}")

    if any(len(keep_dict[i]) == 0 for i in keep_dict):
        print("Pruning would empty a layer; skipping prune candidate.")
        return None
    return keep_dict


def _mask_warm_start_grads(model, layer_idx, old_width):
    """Zero gradients on phi_old; only new-neuron parameters may update."""
    for i, layer in enumerate(model.layers):
        if i == layer_idx:
            if layer.mu_w.grad is not None:
                layer.mu_w.grad[:old_width].zero_()
                layer.rho_w.grad[:old_width].zero_()
                layer.mu_b.grad[:old_width].zero_()
                layer.rho_b.grad[:old_width].zero_()
        elif i == layer_idx + 1:
            if layer.mu_w.grad is not None:
                layer.mu_w.grad[:, :old_width].zero_()
                layer.rho_w.grad[:, :old_width].zero_()
        else:
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    for i, ln in enumerate(model.ln_layers):
        if i == layer_idx:
            if ln.weight.grad is not None:
                ln.weight.grad[:old_width].zero_()
                ln.bias.grad[:old_width].zero_()
        else:
            for p in ln.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    if layer_idx + 1 < len(model.layers):
        for p in model.out.parameters():
            if p.grad is not None:
                p.grad.zero_()
    else:
        if model.out.mu_w.grad is not None:
            model.out.mu_w.grad[:, :old_width].zero_()
            model.out.rho_w.grad[:, :old_width].zero_()
        if model.out.mu_b.grad is not None:
            model.out.mu_b.grad.zero_()
        if model.out.rho_b.grad is not None:
            model.out.rho_b.grad.zero_()


def warm_start_new_neurons(model, layer_idx, old_width, batch_ws, K, eta_ws, beta, lambda_penalty, dataset_size):
    """K gradient steps on phi_new only using warm-start batch B_ws."""
    inputs, labels = batch_ws
    beta_scaled = (1 / dataset_size) * beta

    optimizer = optim.Adam(model.parameters(), lr=eta_ws)
    model.train()
    param_count = count_params(model)

    for _ in range(K):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, _, _, _ = penalised_loss_function(
            outputs, labels, model.kl_loss(), beta_scaled, lambda_penalty, param_count
        )
        loss.backward()
        _mask_warm_start_grads(model, layer_idx, old_width)
        optimizer.step()


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
):
    """
    Evaluate growth and prune candidates via delta penalised ELBO on B_val.
    Returns (action, model, hidden_sizes, info_dict).
    """
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)

    batch_ws = sample_batch(train_loader, device)
    batch_val = sample_batch(val_loader, device)
    val_inputs, val_labels = batch_val

    L_before = penalised_elbo_on_batch(model, val_inputs, val_labels, beta, lambda_penalty, val_dataset_size)
    info = {
        'L_before': L_before,
        'delta_grow': None,
        'delta_prune': None,
        'param_count_before': count_params(model),
    }

    # Growth candidate
    grow_model, hidden_sizes_g = neurogenesis(model, hidden_sizes, gamma=gamma)
    expand_and_load_encoder_layer(model.state_dict(), grow_model)
    warm_start_new_neurons(
        grow_model, layer_idx, old_width, batch_ws, K, eta_ws, beta, lambda_penalty, train_dataset_size
    )
    L_after_grow = penalised_elbo_on_batch(
        grow_model, val_inputs, val_labels, beta, lambda_penalty, val_dataset_size
    )
    delta_grow = L_after_grow - L_before
    info['delta_grow'] = delta_grow

    # Prune candidate
    delta_prune = float('-inf')
    prune_model = None
    hidden_sizes_p = None
    keep_dict = neuroapoptosis(model, rho)
    if keep_dict is not None:
        prune_model, hidden_sizes_p = build_pruned_model(model, keep_dict)
        L_after_prune = penalised_elbo_on_batch(
            prune_model, val_inputs, val_labels, beta, lambda_penalty, val_dataset_size
        )
        delta_prune = L_after_prune - L_before
    info['delta_prune'] = delta_prune if keep_dict is not None else None

    best_action = 'none'
    best_model = model
    best_hidden_sizes = hidden_sizes

    delta_prune_val = delta_prune if keep_dict is not None else float('-inf')
    if delta_grow > 0 or delta_prune_val > 0:
        if delta_grow >= delta_prune_val:
            best_action = 'grow'
            best_model = grow_model
            best_hidden_sizes = hidden_sizes_g
        else:
            best_action = 'prune'
            best_model = prune_model
            best_hidden_sizes = hidden_sizes_p

    info['action'] = best_action
    info['param_count_after'] = count_params(best_model) if best_action != 'none' else info['param_count_before']
    print(
        f"\nStructural decision: L_before={L_before:.4f}, "
        f"delta_grow={delta_grow:.4f}, delta_prune={info['delta_prune']}, "
        f"action={best_action}"
    )
    return best_action, best_model, best_hidden_sizes, info

def truncate_and_load_encoder_layer(old_sd, keep_dict, new_layer):
    num_layers = len(keep_dict)
    new_sd = {}
    for i in range(num_layers):
        keep_i = keep_dict.get(i, None)
        keep_prev = keep_dict.get(i - 1, None)
        for p in ["mu_w", "rho_w", "mu_b", "rho_b"]:
            key = f"layers.{i}.{p}"
            if key not in old_sd:
                raise ValueError(f"{key} is missing in the plasticity model")
            w = old_sd[key]
            if w.ndim == 2:
                if keep_i is not None:
                    w = w[keep_i, :]
                if keep_prev is not None:
                    w = w[:, keep_prev]
            # bias (1D)
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

    for p in ["mu_w", "rho_w", "mu_b", "rho_b"]:
        key = f"out.{p}"
        if key not in old_sd:
            raise ValueError(f"{key} is missing in the plasticity model")
        w = old_sd[key]
        keep_last = keep_dict.get(num_layers - 1, None)
        if w.ndim == 2 and keep_last is not None:
            w = w[:, keep_last]
        new_sd[key] = w


    new_layer.load_state_dict(new_sd, strict=True)


def run_experiment(experiment_name, model, train_loader, val_loader, test_loader, num_epochs,
                   learning_rate=0.001, start_epoch=1, metrics=None, run_test=False, beta=0.1,
                   lambda_penalty=0, output_dir=None):
    """Run a complete training experiment and return metrics"""
    if output_dir is None:
        output_dir = os.path.join('./results', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

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

    loss_label = 'Penalised ELBO' if lambda_penalty > 0 else 'ELBO'
    best_nll = float("inf")
    best_model_state = None
    
    # Training loop
    last_epoch = best_epoch = start_epoch - 1
    for epoch in range(start_epoch, start_epoch + num_epochs):
        last_epoch = epoch
        
        # Train
        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty = train(
            model, train_loader, optimizer, epoch, device, beta, lambda_penalty
        )
        metrics['train_loss_total'].append(train_loss_total)
        metrics['train_loss_nll'].append(train_loss_nll)
        metrics['train_loss_kl'].append(train_loss_kl)
        metrics['train_acc'].append(train_acc)
        metrics['train_brier'].append(train_brier)
        metrics['train_loss_penalty'].append(train_loss_penalty)

        # Validate
        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta, lambda_penalty
        )
        metrics['val_loss_total'].append(val_loss_total)
        metrics['val_loss_nll'].append(val_loss_nll)
        metrics['val_loss_kl'].append(val_loss_kl)
        metrics['val_acc'].append(val_acc)
        metrics['val_brier'].append(val_brier)
        metrics['val_loss_penalty'].append(val_loss_penalty)

        penalty_str = f', Penalty={train_loss_penalty:.4f}' if lambda_penalty > 0 else ''
        print(
            f'Epoch {epoch}: Train Loss({loss_label})={train_loss_total:.4f}, Train Loss(NLL)={train_loss_nll:.4f}, '
            f'Train Loss(KL)={train_loss_kl:.4f}{penalty_str}, Train Acc={train_acc:.2f}%, Train Brier={train_brier:.3f}, '
            f'Val Loss({loss_label})={val_loss_total:.4f}, Val Loss(NLL)={val_loss_nll:.4f}, '
            f'Val Loss(KL)={val_loss_kl:.4f}, Val Acc={val_acc:.2f}%, Val Brier={val_brier:.3f}'
        )
        
        if val_loss_nll < best_nll:
            best_nll = val_loss_nll
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join(output_dir, 'best_model.pth'))
        
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

    if run_test:
        # Load best model for test
        best_model_for_eval = None
        if best_model_state is not None:
            best_model_for_eval = copy.deepcopy(model)
            best_model_for_eval.load_state_dict(best_model_state)
            print(f"Loaded best model from Epoch {best_epoch} based on validation loss for final testing.")
    
        eval_model = best_model_for_eval if best_model_for_eval is not None else model
        
        test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
            eval_model, test_loader, device, beta, lambda_penalty
        )
        print(f'Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}')
        
        # Save best checkpoint (same weights used for test eval)
        checkpoint = best_model_state if best_model_state is not None else model.state_dict()
        torch.save(checkpoint, os.path.join(output_dir, 'model.pth'))

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
            'param_count': param_stats['total_params'],
            'trainable_param_count': param_stats.get('trainable_params', param_stats['total_params']),
            'hidden_sizes': hidden_sizes
        })
        
        # Create a metrics DataFrame
        metrics_df = pd.DataFrame({
            'epoch': range(1, 1 + len(metrics['train_loss_total'])),
            'train_loss_total': metrics['train_loss_total'],
            'train_loss_nll': metrics['train_loss_nll'],
            'train_loss_kl': metrics['train_loss_kl'],
            'train_loss_penalty': metrics.get('train_loss_penalty', []),
            'train_acc': metrics['train_acc'],
            'train_brier': metrics['train_brier'],
            'val_loss_total': metrics['val_loss_total'],
            'val_loss_nll': metrics['val_loss_nll'],
            'val_loss_kl': metrics['val_loss_kl'],
            'val_loss_penalty': metrics.get('val_loss_penalty', []),
            'val_acc': metrics['val_acc'],
            'val_brier': metrics['val_brier'],
        })
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
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
    decision_interval,
    gamma,
    rho,
    warm_start_steps,
    warm_start_lr,
    output_dir=None,
):
    """
    Dynamic structural adaptation via penalised ELBO (Algorithm 1).
    Structural decisions every decision_interval epochs.
    """
    if output_dir is None:
        output_dir = os.path.join('./results', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Training {experiment_name.upper()} (Penalised ELBO, lambda={lambda_penalty})")
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
        'param_count_history': [],
    }

    best_nll = float('inf')
    best_model_state = None
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty = train(
            model, train_loader, optimizer, epoch, device, beta, lambda_penalty
        )
        metrics['train_loss_total'].append(train_loss_total)
        metrics['train_loss_nll'].append(train_loss_nll)
        metrics['train_loss_kl'].append(train_loss_kl)
        metrics['train_loss_penalty'].append(train_loss_penalty)
        metrics['train_acc'].append(train_acc)
        metrics['train_brier'].append(train_brier)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta, lambda_penalty
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

        if val_loss_nll < best_nll:
            best_nll = val_loss_nll
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join(output_dir, 'best_model.pth'))

        if epoch % decision_interval == 0:
            print("-" * 20 + f" Decision juncture (epoch {epoch}) " + "-" * 20)
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
            )
            metrics['structural_epochs'].append(epoch)
            metrics['structural_actions'].append(action)
            metrics['structural_delta_grow'].append(info.get('delta_grow'))
            metrics['structural_delta_prune'].append(info.get('delta_prune'))
            metrics['structural_L_before'].append(info.get('L_before'))

            if action != 'none':
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
                )
                torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch{epoch}_{action}.pth'))

    # Final test evaluation with best checkpoint
    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, 'metrics.png'),
    )

    if best_model_state is not None:
        eval_model = copy.deepcopy(model)
        eval_model.load_state_dict(best_model_state)
        print(f"Loaded best model from Epoch {best_epoch} based on validation NLL for final testing.")
    else:
        eval_model = model

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta, lambda_penalty
    )
    print(f'Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}')

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    torch.save(checkpoint, os.path.join(output_dir, 'model.pth'))

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

    structural_df = pd.DataFrame({
        'epoch': metrics['structural_epochs'],
        'action': metrics['structural_actions'],
        'delta_grow': metrics['structural_delta_grow'],
        'delta_prune': metrics['structural_delta_prune'],
        'L_before': metrics['structural_L_before'],
    })
    structural_df.to_csv(os.path.join(output_dir, 'structural_decisions.csv'), index=False)

    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(metrics['val_acc']):.2f}%")
    print(f"Best validation NLL: {min(metrics['val_loss_nll']):.4f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final hidden sizes: {final_hidden_sizes}")
    print(f"Structural decisions: {metrics['structural_actions']}")

    return metrics, model, num_epochs

def main(save_path):
    # Hyperparameters
    num_epochs = 200
    batch_size = 1024
    learning_rate = 0.001
    hidden_sizes = [512, 256, 128, 64]
    # hidden_sizes = [16, 16, 16, 16]

    beta = 0.1
    lambda_penalty = 1e-4
    decision_interval = 10
    gamma = 0.1
    rho = 0.05
    warm_start_steps = 5
    warm_start_lr = 0.001
    
    # Create results directory
    os.makedirs(f'{save_path}', exist_ok=True)
    

    # Create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Lambda(lambda x: x.view(-1)) 
    ])
    
    training_data = datasets.FashionMNIST(
        root="../../Datasets",
        train=True,
        download=True,
        transform=transform
    )
    
    train_size = int(0.8 * len(training_data))
    val_size = len(training_data) - train_size 
    
    train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

    test_dataset = datasets.FashionMNIST(
        root="../../Datasets",
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders with fixed seeds for workers
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # ========== Experiment 1: Baseline Model ==========
    print("\n\n" + "="*50)
    print("Training Baseline Model")
    print("="*50)
    baseline_model = BayesianFNN(784, hidden_sizes, 10).to(device)
    initial_state_dict = copy.deepcopy(baseline_model.state_dict())
    baseline_output_dir = os.path.join(save_path, 'baseline')
    baseline_metrics, _, _= run_experiment(
        'baseline', 
        baseline_model, 
        train_loader, 
        val_loader, 
        test_loader, 
        num_epochs, 
        learning_rate,
        start_epoch=1,
        run_test=True,
        beta=beta,
        output_dir=baseline_output_dir,
    )


    # ========== Experiment 2: Adaptive Model (Penalised ELBO) ==========

    print("\n\n" + "=" * 50)
    print("Training Adaptive Model")
    print("=" * 50)

    base_model = BayesianFNN(784, hidden_sizes, 10).to(device)
    base_model.load_state_dict(initial_state_dict)
    plasticity_output_dir = os.path.join(save_path, 'plasticity')
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
        decision_interval=decision_interval,
        gamma=gamma,
        rho=rho,
        warm_start_steps=warm_start_steps,
        warm_start_lr=warm_start_lr,
        output_dir=plasticity_output_dir,
    )
    
    # # ========== Compare Results ==========
    all_metrics = {
         'baseline': baseline_metrics,
        'plasticity': plasticity_metrics,

    }
    plot_metrics(all_metrics, save_path=f'./{save_path}/model_comparison.png')
    
    # Create summary table
    summary = pd.DataFrame([
        {
            'Model': 'Baseline',
            'Parameters': baseline_metrics['param_count'],
            'Trainable Params': baseline_metrics['trainable_param_count'],
            'Best Val Acc': max(baseline_metrics['val_acc']),
            'Best Val Brier': min(baseline_metrics['val_brier']),
            'Test Acc': baseline_metrics['test_acc'],
            'Test Brier': baseline_metrics['test_brier'],
            'Hidden Sizes': baseline_metrics['hidden_sizes'],
        },
        {
            'Model': 'Plasticity',
            'Parameters': plasticity_metrics['param_count'],
            'Trainable Params': plasticity_metrics['trainable_param_count'],
            'Best Val Acc': max(plasticity_metrics['val_acc']),
            'Best Val Brier': min(plasticity_metrics['val_brier']),
            'Test Acc': plasticity_metrics['test_acc'],
            'Test Brier': plasticity_metrics['test_brier'],
            'Hidden Sizes': plasticity_metrics['hidden_sizes'],
        }
    ])
    
    summary.to_csv(f'./{save_path}/experiment_summary.csv', index=False)
    print("\nExperiment Summary:")
    print(summary)
    return summary


def get_statistics(save_path="results/", configs=["underparametrized"], model_names=["Baseline", "Plasticity"], metrics = ["Parameters", "Best Val Acc", "Best Val Brier", "Test Acc", "Test Brier"], num_runs=5): 
    datasets={}
    for config in configs:
        for i in range(1,num_runs+1): 
            datasets[(config,i)] = pd.read_csv(f"{save_path}/{config}/run_{i}/experiment_summary.csv")

    results={}
    # mean and standard deviation
    for config in configs:
        result = []
        for model_name in model_names:
            row = {}
            row["Model"] = model_name
            for metric in metrics:
                metric_values = []
                for i in range(1, num_runs+1):
                    metric_values.append(datasets[(config, i)].loc[datasets[(config, i)]["Model"] == model_name, metric].iloc[0])
                row[f"{metric} (Mean)"] = np.mean(metric_values)
                row[f"{metric} (Std)"] = np.std(metric_values, ddof=1)
            result.append(row)
        results[config] = pd.DataFrame(result)
    return results

if __name__ == "__main__":
    # df1=get_statistics(configs=["overparametrized"], model_names=["Baseline"])
    # print(df1["overparametrized"])
    for i in range(1,6):
        print("Running experiment for run", i)
        main(f"results/overparametrized/run_{i}")
    print("All experiments completed.")
