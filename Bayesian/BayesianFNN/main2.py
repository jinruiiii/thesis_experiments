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
        "Selected checkpoint metric": str(selected_checkpoint_metric),
        "Selected epoch": int(best_epoch),
        "Selected val_total": float(best_val_total),
    }])
    summary_df.to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)

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
    """Growth candidate: expand layer l* with highest mean normalized posterior variance."""
    if exclude is None:
        exclude = []
    uncertainty = plasticity_original.get_average_uncertainty_per_layer()
    print("\n Average Normalised Uncertainty per Hidden Layer:")
    for i, val in enumerate(uncertainty):
        print(f"  Layer {i+1}: {val.item()/(hidden_sizes[i]**0.5):.6f}")
    eligible = [i for i in range(len(uncertainty)) if i not in exclude]
    if not eligible:
        print("[neurogenesis] All layers excluded; ignoring exclude list for this step.")
        eligible = list(range(len(uncertainty)))
    layer_to_expand = max(
        eligible,
        key=lambda i: uncertainty[i] / (hidden_sizes[i] ** 0.5),
    )
    neurons_to_add = max(1, math.ceil(gamma * hidden_sizes[layer_to_expand]))
    old_width = hidden_sizes[layer_to_expand]
    print(f"Expanding Layer {layer_to_expand+1} "
            f"(Highest Normalised Uncertainty: {uncertainty[layer_to_expand].item()/(hidden_sizes[layer_to_expand]**0.5):.6f}) "
            f"by {neurons_to_add} neurons")

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


def neuroapoptosis(
    plasticity_model,
    prune_rate,
    exclude=None,
    snr_combine="geometric",
    min_neurons_per_layer=2,
):
    """
    Per-layer structured pruning: prune bottom prune_rate fraction *within each layer*
    by bidirectional SNR.

    Returns keep_dict[layer_idx] = list of kept neuron indices for each hidden layer,
    or None if pruning would violate min_neurons_per_layer in any layer.
    """
    if exclude is None:
        exclude = []

    keep_dict = {}

    for i, layer in enumerate(plasticity_model.layers):
        n_neurons = layer.mu_w.shape[0]

        # Always keep excluded layers unchanged
        if i in exclude:
            keep_dict[i] = list(range(n_neurons))
            continue

        # If already too small to prune safely, keep as-is
        if n_neurons <= min_neurons_per_layer:
            keep_dict[i] = list(range(n_neurons))
            continue

        # Compute bidirectional SNR score for each neuron j in this layer
        scores = []
        for j in range(n_neurons):
            s = _neuron_snr_bidirectional(plasticity_model, i, j, combine=snr_combine)
            scores.append(float(s.item()))

        # Decide how many to prune in this layer
        num_prune = int(prune_rate * n_neurons)
        num_prune = max(1, num_prune)  # prune at least 1 if pruning is enabled
        # Ensure we keep at least min_neurons_per_layer
        max_prune_allowed = n_neurons - min_neurons_per_layer
        if max_prune_allowed <= 0:
            keep_dict[i] = list(range(n_neurons))
            continue
        num_prune = min(num_prune, max_prune_allowed)

        # Pick lowest-scoring neurons to prune
        neuron_indices = list(range(n_neurons))
        neuron_indices.sort(key=lambda j: scores[j])
        pruned = set(neuron_indices[:num_prune])

        kept = [j for j in range(n_neurons) if j not in pruned]
        if len(kept) < min_neurons_per_layer:
            return None  # should not happen given logic, but keeps it safe

        keep_dict[i] = kept

    # Final safety check: don't allow any layer to become empty
    if any(len(v) == 0 for v in keep_dict.values()):
        return None

    return keep_dict

# def _mask_warm_start_grads(model, layer_idx, old_width):
#     """Zero gradients on phi_old; only new-neuron parameters may update."""
#     for i, layer in enumerate(model.layers):
#         if i == layer_idx:
#             if layer.mu_w.grad is not None:
#                 layer.mu_w.grad[:old_width].zero_()
#                 layer.rho_w.grad[:old_width].zero_()
#                 layer.mu_b.grad[:old_width].zero_()
#                 layer.rho_b.grad[:old_width].zero_()
#         elif i == layer_idx + 1:
#             if layer.mu_w.grad is not None:
#                 layer.mu_w.grad[:, :old_width].zero_()
#                 layer.rho_w.grad[:, :old_width].zero_()
#         else:
#             for p in layer.parameters():
#                 if p.grad is not None:
#                     p.grad.zero_()

#     for i, ln in enumerate(model.ln_layers):
#         if i == layer_idx:
#             if ln.weight.grad is not None:
#                 ln.weight.grad[:old_width].zero_()
#                 ln.bias.grad[:old_width].zero_()
#         else:
#             for p in ln.parameters():
#                 if p.grad is not None:
#                     p.grad.zero_()

#     if layer_idx + 1 < len(model.layers):
#         for p in model.out.parameters():
#             if p.grad is not None:
#                 p.grad.zero_()
#     else:
#         if model.out.mu_w.grad is not None:
#             model.out.mu_w.grad[:, :old_width].zero_()
#             model.out.rho_w.grad[:, :old_width].zero_()
#         if model.out.mu_b.grad is not None:
#             model.out.mu_b.grad.zero_()
#         if model.out.rho_b.grad is not None:
#             model.out.rho_b.grad.zero_()

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
):
    """
    K gradient steps cycling through a list of warm-start batches.
    Trains all params (as in your current warm_start_model), but on multiple batches.
    """
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
        if mask_mode == "grow_new_only":
            _mask_warm_start_grads_grow_new_only(model, layer_idx=layer_idx, old_width=old_width)
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
    grow_exclude_layers=None
):
    """
    Evaluate growth and prune candidates via delta penalised ELBO on B_val.
    Returns (action, model, hidden_sizes, info_dict).
    """
    train_dataset_size = len(train_loader.dataset)
    beta_scaled = (1 / train_dataset_size) * beta

    # choose how many batches
    M_ws = 40     # warm-start batches
    M_val = 10    # evaluation batches

    batches_ws = sample_batches(train_loader, device, M_ws)
    batches_val = sample_batches(val_loader, device, M_val)

    L_before = penalised_elbo_on_batches(model, batches_val, beta_scaled, lambda_penalty)
    info = {
        'L_before': L_before,
        'delta_grow': None,
        'delta_prune': None,
        'param_count_before': count_params(model),
    }
    if grow_exclude_layers is None:
        grow_exclude_layers = []

    # Growth candidate
    grow_model, hidden_sizes_g, layer_idx, old_width = neurogenesis(
        model,
        hidden_sizes,
        exclude=list(grow_exclude_layers),
        gamma=gamma,
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
    )
    L_after_grow = penalised_elbo_on_batches(
        grow_model, batches_val, beta_scaled, lambda_penalty
    )
    delta_grow = L_after_grow - L_before
    info['delta_grow'] = delta_grow

    # Prune candidate
    delta_prune = float('-inf')
    prune_model = None
    hidden_sizes_p = None
    keep_dict = neuroapoptosis(model, rho, snr_combine="geometric")
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

    info['grow_layer_idx'] = layer_idx
    info['grow_exclude_layers'] = list(grow_exclude_layers)
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
                   learning_rate=0.001, start_epoch=1, metrics=None, beta=0.1,
                   lambda_penalty=0, output_dir=None):
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
    best_total = float("inf")
    best_model_state = None
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    
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
        
        if val_loss_total < best_total:
            best_total = val_loss_total
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                hidden_sizes=[layer.mu_w.shape[0] for layer in model.layers],
                selection_metric="val_total",
                selection_value=best_total,
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
        best_val_total=best_total,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=lambda_penalty,
        selected_checkpoint_metric="val_total",
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
    output_dir=None,
    growth_cooldown_junctures=1
):
    """
    Dynamic structural adaptation via penalised ELBO.
    Structural decisions happen at annealed intervals (or fixed if decision_interval is set).
    """
    if output_dir is None:
        output_dir = os.path.join('./results', experiment_name)
    ensure_output_dir(output_dir)

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
        'structural_hidden_sizes': [],
        'param_count_history': [],
    }

    growth_cooldown = {}

    best_total = float('inf')
    best_model_state = None
    best_epoch = 0
    best_hidden_sizes = list(hidden_sizes)
    beta_scaled = (1 / len(train_loader.dataset)) * beta

    def _annealed_decision_interval(epoch: int) -> int:
        # Option B: I(e)=floor(Imin + (Imax-Imin)*(e/E)^p), clamped.
        E = max(1, num_epochs)
        frac = min(1.0, max(0.0, epoch / E))
        raw = decision_interval_min + (decision_interval_max - decision_interval_min) * (frac ** decision_interval_power)
        interval = int(math.floor(raw))
        return interval

    next_decision_epoch = None
    if decision_interval is not None:
        next_decision_epoch = decision_interval
    else:
        next_decision_epoch = decision_interval_min

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

        if val_loss_total < best_total:
            best_total = val_loss_total
            best_epoch = epoch
            best_hidden_sizes = list(hidden_sizes)
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                hidden_sizes=best_hidden_sizes,
                selection_metric="val_total",
                selection_value=best_total,
            )

        if epoch == next_decision_epoch and epoch != num_epochs:
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
            )
            metrics['structural_epochs'].append(epoch)
            metrics['structural_actions'].append(action)
            metrics['structural_delta_grow'].append(info.get('delta_grow'))
            metrics['structural_delta_prune'].append(info.get('delta_prune'))
            metrics['structural_L_before'].append(info.get('L_before'))
            metrics['structural_hidden_sizes'].append(list(hidden_sizes))

            if action != 'none':
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
                )
        

            if action == "grow" and growth_cooldown_junctures > 0:
                grown = info["grow_layer_idx"]
                growth_cooldown[grown] = growth_cooldown_junctures

                for i in list(growth_cooldown.keys()):
                    if i != grown: 
                        growth_cooldown[i] -= 1
                        if growth_cooldown[i] <= 0:
                            del growth_cooldown[i]

            # Schedule next decision juncture.
            if decision_interval is not None:
                next_decision_epoch = min(num_epochs, epoch + int(decision_interval))
            else:
                next_decision_epoch = min(num_epochs, epoch + _annealed_decision_interval(epoch))

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
        best_val_total=best_total,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=lambda_penalty,
        selected_checkpoint_metric="val_total",
    )


    structural_df = pd.DataFrame({
        'epoch': metrics['structural_epochs'],
        'action': metrics['structural_actions'],
        'delta_grow': metrics['structural_delta_grow'],
        'delta_prune': metrics['structural_delta_prune'],
        'L_before': metrics['structural_L_before'],
        'hidden_sizes': [str(hs) for hs in metrics['structural_hidden_sizes']],
    })
    structural_df.to_csv(os.path.join(output_dir, 'structural_decisions.csv'), index=False)

    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(metrics['val_acc']):.2f}%")
    print(f"Best validation total: {min(metrics['val_loss_total']):.4f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final hidden sizes: {final_hidden_sizes}")
    print(f"Structural decisions: {metrics['structural_actions']}")

    return metrics, model, num_epochs

def main(save_path):
    # Hyperparameters
    num_epochs = 200
    batch_size = 1024
    learning_rate = 0.001
    hidden_sizes = [256, 256, 256, 256]

    beta = 0.01
    lambda_penalty = 1e-8
    decision_interval_min = 2
    decision_interval_max = 20
    decision_interval_power = 2
    gamma = 0.2
    rho = 0.05
    warm_start_steps = 80
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
    # print("\n\n" + "="*50)
    # print("Training Baseline Model")
    # print("="*50)
    baseline_model = BayesianFNN(784, hidden_sizes, 10).to(device)
    initial_state_dict = copy.deepcopy(baseline_model.state_dict())
    # baseline_output_dir = os.path.join(save_path, 'baseline')
    # baseline_metrics, _, _= run_experiment(
    #     'baseline', 
    #     baseline_model, 
    #     train_loader, 
    #     val_loader, 
    #     test_loader, 
    #     num_epochs, 
    #     learning_rate,
    #     start_epoch=1,
    #     beta=beta,
    #     output_dir=baseline_output_dir,
    # )


    # ========== Experiment 2: Adaptive Model (Penalised ELBO) ==========

    print("\n\n" + "=" * 50)
    print("Training Adaptive Model")
    print("=" * 50)

    base_model = BayesianFNN(784, hidden_sizes, 10).to(device)
    base_model.load_state_dict(initial_state_dict)
    plasticity_output_dir = os.path.join(save_path, f'plasticity_{hidden_sizes[0]}_{lambda_penalty}')
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
        output_dir=plasticity_output_dir,
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

if __name__ == "__main__":
    # df1=get_statistics(save_path="results", experiments=["plasticity_256_1e-08"], num_runs=5)
    # print(df1["plasticity_256_1e-08"]["summary"])
    for i in range(1,6):
        print("Running experiment for run", i)
        main(f"results/run_{i}")
    print("All experiments completed.")
