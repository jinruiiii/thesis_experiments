from BayesianFNN import BayesianFNN
import random
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import copy
import pandas as pd
import importlib
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
        'strong_baseline': 'yellow',
        'plasticity_multi_growth': 'red',
        'plasticity_single_growth': 'green'
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

class EarlyStopping:
    def __init__(self, patience=3, delta=0.025, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

def loss_function(outputs, labels, kl_loss, beta):
    criterion = nn.CrossEntropyLoss()
    nll = criterion(outputs, labels)
    # normalise to per sample
    return nll + kl_loss*beta, nll, kl_loss*beta

def train(model, train_dataloader, optimizer, epoch, device, beta):
    model.train()
    running_loss_total = 0.0
    running_loss_nll= 0.0
    running_loss_kl = 0.0
    running_brier = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss, nll, kl = loss_function(outputs, labels, model.kl_loss(), beta = (1/len(train_dataloader.dataset)) * beta)
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss_total += loss.item()
        running_loss_nll += nll.item()
        running_loss_kl += kl.item()

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
    train_brier = running_brier / total
    
    return train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier

def validate(model, val_dataloader, device, beta):
    model.eval()
    val_loss_total = 0.0
    val_loss_nll = 0.0
    val_loss_kl = 0.0
    running_brier = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss, nll, kl = loss_function(outputs, labels, model.kl_loss(), beta = (1/len(val_dataloader.dataset)) * beta)

            # Track Statistics
            val_loss_total += loss.item()
            val_loss_nll += nll.item()
            val_loss_kl += kl.item()
            
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
    val_brier = running_brier / total
    
    return val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier


def neurogenesis(plasticity_original, hidden_sizes, exclude=[0], method="uncertainty", growth_rate=0.1):
    layer_to_expand = None
    neurons_to_add  = None
    if method == "uncertainty":
        uncertainty = plasticity_original.get_average_uncertainty_per_layer()
        print("\n Average Normalised Uncertainty per Hidden Layer:")
        for i, val in enumerate(uncertainty):
            print(f"  Layer {i+1}: {val.item()/(hidden_sizes[i]**0.5):.6f}")
        layer_to_expand = max(
            (i for i in range(len(uncertainty)) if i not in exclude),
            key=lambda i: uncertainty[i]/(hidden_sizes[i]**0.5)
        )
        neurons_to_add = max(1, int(hidden_sizes[layer_to_expand] * growth_rate))
        print(f"Expanding Layer {layer_to_expand+1} "
              f"(Highest Normalised Uncertainty: {uncertainty[layer_to_expand].item()/(hidden_sizes[layer_to_expand]**0.5):.6f}) "
              f"by {neurons_to_add} neurons")
    elif method == "snr":
        snr = plasticity_original.get_average_snr_per_layer()
        print("\n Average Signal-to-Noise Ratio per Hidden Layer:")
        for i, val in enumerate(snr):
            print(f"  Layer {i+1}: {val.item()/(hidden_sizes[i]**0.5):.6f}")
        layer_to_expand = min(
            (i for i in range(len(snr)) if i not in exclude),
            key=lambda i: snr[i]/(hidden_sizes[i]**0.5)
        )
        neurons_to_add = max(1, int(hidden_sizes[layer_to_expand] * growth_rate))
        print(f"Expanding Layer {layer_to_expand+1} "
              f"(Lowest SNR: {snr[layer_to_expand].item()/(hidden_sizes[layer_to_expand]**0.5):.6f}) "
              f"by {neurons_to_add} neurons")
    else:
        raise ValueError(f"{method} not a defined method for neurogenesis.")
        
    expanded_hidden_sizes = hidden_sizes.copy()
    expanded_hidden_sizes[layer_to_expand] += neurons_to_add
    plasticity_neurogenesis = BayesianFNN(plasticity_original.in_features, expanded_hidden_sizes, plasticity_original.out_features).to(device)
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

def neuroapoptosis(plasticity_model, threshold=0.04, exclude=[0], method="uncertainty"):
    keep_dict = {}
    print("\n Neurons Pruned from Each Hidden Layer:")
    for i, layer in enumerate(plasticity_model.layers):
        if method == "uncertainty":
            metric = layer.get_uncertainty()
            metric_per_neuron = torch.mean(metric, dim=1)
            if i in exclude:
                keep_dict[i] = list(range(len(metric_per_neuron)))
            else:
                mask = metric_per_neuron <= threshold
                keep_dict[i] = mask.nonzero(as_tuple=True)[0].tolist()
        elif method == "snr":
            metric = layer.get_snr()
            metric_per_neuron = torch.mean(metric, dim=1)
            if i in exclude:
                keep_dict[i] = list(range(len(metric_per_neuron)))
            else:
                mask = metric_per_neuron >= threshold
                keep_dict[i] = mask.nonzero(as_tuple=True)[0].tolist()
        else:
            raise ValueError(f"{method} not a defined method for neuroapoptosis.")

        print(f"Hidden Layer {i+1}: {len(metric_per_neuron) - len(keep_dict[i])}")


    # just to print
    for i, layer in enumerate(plasticity_model.layers):
        if method == "uncertainty":
            metric = layer.get_uncertainty()
            metric_per_neuron = torch.mean(metric, dim=1)
            print(metric_per_neuron)
        elif method == "snr":
            metric = layer.get_snr()
            metric_per_neuron = torch.mean(metric, dim=1)
            print(metric_per_neuron)

    return keep_dict

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

def naive_truncate_and_load_encoder_layer(old_sd, new_layer):
    new_sd = new_layer.state_dict()

    new_trunc_sd = {}

    for k in new_sd.keys():
        if k not in old_sd:
            print(f"[skip] {k} not found in origin_layer")
            continue

        old_param = old_sd[k]
        new_param = new_sd[k]

        if old_param.shape == new_param.shape:
            new_trunc_sd[k] = old_param
        elif len(old_param.shape) == 2:
            # Linear weights
            new_trunc_sd[k] = old_param[:new_param.shape[0], :new_param.shape[1]]
        elif len(old_param.shape) == 1:
            # Biases / LayerNorm
            new_trunc_sd[k] = old_param[:new_param.shape[0]]
        else:
            print(f"[warn] {k} shape mismatch: old {old_param.shape}, new {new_param.shape}")
            continue

    new_layer.load_state_dict(new_trunc_sd, strict=True)

def run_experiment(experiment_name, model, train_loader, val_loader, test_loader, num_epochs, 
                   learning_rate=0.001, start_epoch=1, early_stopper=None, metrics=None, run_test=False, beta=0.1):
    """Run a complete training experiment and return metrics"""
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

    best_nll = float("inf")
    best_model_state = None
    
    # Training loop
    last_epoch = best_epoch = start_epoch - 1
    for epoch in range(start_epoch, start_epoch + num_epochs):
        last_epoch = epoch
        
        # Train
        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier = train(model, train_loader, optimizer, epoch, device, beta)
        metrics['train_loss_total'].append(train_loss_total)
        metrics['train_loss_nll'].append(train_loss_nll)
        metrics['train_loss_kl'].append(train_loss_kl)
        metrics['train_acc'].append(train_acc)
        metrics['train_brier'].append(train_brier)
        
        # Validate
        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier = validate(model, val_loader, device, beta)
        metrics['val_loss_total'].append(val_loss_total)
        metrics['val_loss_nll'].append(val_loss_nll)
        metrics['val_loss_kl'].append(val_loss_kl)
        metrics['val_acc'].append(val_acc)
        metrics['val_brier'].append(val_brier)
        
        print(f'Epoch {epoch}: Train Loss(ELBO)={train_loss_total:.4f}, Train Loss(NLL)={train_loss_nll:.4f}, Train Loss(KL)={train_loss_kl:.4f}, Train Acc={train_acc:.2f}%, Train Brier={train_brier:.3f}, '
              f'Val Loss(ELBO)={val_loss_total:.4f}, Val Loss(NLL)={val_loss_nll:.4f}, Val Loss(KL)={val_loss_kl:.4f}, Val Acc={val_acc:.2f}%, Val Brier={val_brier:.3f}')
        
        if val_loss_nll < best_nll:
            best_nll = val_loss_nll
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, f'./results/{experiment_name}/best_model.pth')
            
        if early_stopper:
            early_stopper.check_early_stop(val_loss_nll)
            if early_stopper.stop_training:
                break

        
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
        save_path=f'./results/{experiment_name}/metrics.png'
    )

    if run_test:
        # Load best model for test
        best_model_for_eval = None
        if best_model_state is not None:
            best_model_for_eval = copy.deepcopy(model)
            best_model_for_eval.load_state_dict(best_model_state)
            print(f"Loaded best model from Epoch {best_epoch} based on validation loss for final testing.")
    
        eval_model = best_model_for_eval if best_model_for_eval is not None else model
        
        test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier = validate(
            eval_model, test_loader, device, beta
        )
        print(f'Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}')
        
        # Save model
        torch.save(model.state_dict(), f'./results/{experiment_name}/model.pth')

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
            'train_acc': metrics['train_acc'],
            'train_brier': metrics['train_brier'],
            'val_loss_total': metrics['val_loss_total'],
            'val_loss_nll': metrics['val_loss_nll'],
            'val_loss_kl': metrics['val_loss_kl'],
            'val_acc': metrics['val_acc'],
            'val_brier':metrics['val_brier']
        })
        metrics_df.to_csv(f'./results/{experiment_name}/metrics.csv', index=False)
        
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


def run_naive_plasticity_experiment(
    experiment_name,
    plasticity_model,
    hidden_sizes,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    learning_rate,
    beta,
    growth_epochs,
    growth_rate
):
    print("\n\n" + "="*50)
    print(f"Training {experiment_name.upper()}")
    print("="*50)

    metrics = None
    num_epochs_used = 0

    print("+"*20 + " Growing Phase " + "+"*20)

    base_epochs = min(growth_epochs, num_epochs)

    # -------------------------
    # Stage 1: Train model
    # -------------------------
    metrics, plasticity_model, epochs_run= run_experiment(
        experiment_name,
        plasticity_model,
        train_loader,
        val_loader,
        test_loader,
        base_epochs,
        learning_rate,
        start_epoch=1,
        metrics=metrics,
        run_test=False,
        beta=beta
    )
    num_epochs_used += epochs_run

    # -------------------------
    # Stage 2: Neurogenesis Growth
    # -------------------------
    old_model = plasticity_model
    new_model, hidden_sizes = neurogenesis(
        old_model,
        hidden_sizes,
        exclude=[],
        method="uncertainty",
        growth_rate=growth_rate
    )
    expand_and_load_encoder_layer(old_model.state_dict(), new_model)

    # -------------------------
    # Stage 3: Train model
    # -------------------------
    plasticity_model = new_model
    remaining_epochs = max(0, num_epochs - num_epochs_used)
    grow_train_epochs = min(growth_epochs, remaining_epochs)
    metrics, plasticity_model, epochs_run = run_experiment(
        experiment_name,
        plasticity_model,
        train_loader,
        val_loader,
        test_loader,
        grow_train_epochs,
        learning_rate,
        start_epoch=1 + num_epochs_used,
        metrics=metrics,
        run_test=False,
        beta=beta
    )
    num_epochs_used += epochs_run

    # -------------------------
    # Stage 4: Neuroapoptosis Pruning
    # -------------------------
    new_model = old_model
    print("-"*20 + " Pruning Phase " + "-"*20)
    naive_truncate_and_load_encoder_layer(plasticity_model.state_dict(), new_model)
    plasticity_model = new_model
    remaining_epochs = max(0, num_epochs - num_epochs_used)
        

    # -------------------------
    # Stage 5: Train model
    # -------------------------
    if remaining_epochs > 0:
        metrics, plasticity_model, epochs_run = run_experiment(
            experiment_name,
            plasticity_model,
            train_loader,
            val_loader,
            test_loader,
            remaining_epochs,
            learning_rate,
            start_epoch=1 + num_epochs_used,
            metrics=metrics,
            run_test=True,
            beta=beta
        )

    return metrics, plasticity_model


def run_plasticity_experiment(
    experiment_name,
    plasticity_model,
    hidden_sizes,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    cycles,
    learning_rate,
    beta,
    growth_steps_per_cycle=2,     
    adapt_epochs_per_growth=10,  
    growth_rate=1.0,
    prune_thresholds=None,
    search_epochs=5,
    warmup_epochs=10
):
    print("\n" + "=" * 50)
    print(f"Training {experiment_name.upper()}")
    print("=" * 50)

    train_metrics = None
    epochs_completed = 0

    epochs_per_cycle = num_epochs // cycles

    for cycle in range(1, cycles + 1):

        print("-" * 20 + f" Cycle = {cycle} " + "-" * 20)

        remaining_cycle_budget = epochs_per_cycle

        if cycle == 1:

            train_metrics, plasticity_model, epochs_run = run_experiment(
                experiment_name,
                plasticity_model,
                train_loader,
                val_loader,
                test_loader,
                warmup_epochs,
                learning_rate,
                start_epoch=1 + epochs_completed,
                metrics=train_metrics,
                run_test=False,
                beta=beta,
            )

            epochs_completed += epochs_run
            remaining_cycle_budget -= epochs_run

        # -------------------------
        # Growth → Train loop
        # -------------------------
        for g in range(growth_steps_per_cycle):

            old_model = plasticity_model

            plasticity_model, hidden_sizes = neurogenesis(
                old_model,
                hidden_sizes,
                exclude=[],
                method="uncertainty",
                growth_rate=growth_rate,
            )

            expand_and_load_encoder_layer(
                old_model.state_dict(),
                plasticity_model
            )

            # adaptive training after growth
            epochs_to_train = adapt_epochs_per_growth

            print(f"[Cycle {cycle} | Growth Step {g+1}] Training for {epochs_to_train} epochs")

            train_metrics, plasticity_model, epochs_run = run_experiment(
                experiment_name,
                plasticity_model,
                train_loader,
                val_loader,
                test_loader,
                epochs_to_train,
                learning_rate,
                start_epoch=1 + epochs_completed,
                metrics=train_metrics,
                run_test=False,
                beta=beta,
            )

            epochs_completed += epochs_run
            remaining_cycle_budget -= epochs_run

        # -------------------------
        # Pruning search phase
        # -------------------------
        print("\n" + "-" * 20 + " Pruning Search Phase " + "-" * 20)

        prune_thresholds = sorted(prune_thresholds)

        best_threshold = None
        best_nll = float("inf")
        prev_keep_dict = None

        for threshold in prune_thresholds:
            print("\n" + "-" * 20 + f" Pruning Search Phase (Threshold = {threshold})" + "-" * 20)
            keep_dict = neuroapoptosis(
                plasticity_model,
                threshold=threshold,
                exclude=[],
                method="snr",
            )

            if [] in keep_dict.values():
                print(
                    f"Prune threshold {threshold} too high; empty layers detected. "
                    f"Skipping this and higher thresholds."
                )
                break

            if keep_dict == prev_keep_dict:
                print(
                    f"Prune threshold {threshold} results in same architecture as previous threshold. "
                    f"Skipping to avoid redundant evaluation."
                )
                continue
            prev_keep_dict = keep_dict

            hidden_sizes = [len(v) for v in keep_dict.values()]

            device = next(plasticity_model.parameters()).device

            candidate_model = BayesianFNN(
                plasticity_model.in_features,
                hidden_sizes,
                plasticity_model.out_features,
            ).to(device)

            truncate_and_load_encoder_layer(
                plasticity_model.state_dict(),
                keep_dict,
                candidate_model,
            )

            prune_metrics, _, _ = run_experiment(
                experiment_name,
                candidate_model,
                train_loader,
                val_loader,
                test_loader,
                search_epochs,
                learning_rate,
                start_epoch=1,
                metrics=None,
                run_test=False,
                beta=beta,
            )

            val_nll = min(prune_metrics["val_loss_nll"])

            if val_nll <= best_nll:
                best_nll = val_nll
                best_threshold = threshold

        print(f"Best prune threshold: {best_threshold} | Val NLL: {best_nll:.4f}")

        # -------------------------
        # Final pruning
        # -------------------------
        keep_dict = neuroapoptosis(
            plasticity_model,
            threshold=best_threshold,
            exclude=[],
            method="snr",
        )

        hidden_sizes = [len(v) for v in keep_dict.values()]

        device = next(plasticity_model.parameters()).device

        old_model = plasticity_model
        plasticity_model = BayesianFNN(
            old_model.in_features,
            hidden_sizes,
            old_model.out_features,
        ).to(device)

        truncate_and_load_encoder_layer(
            old_model.state_dict(),
            keep_dict,
            plasticity_model,
        )

        # -------------------------
        # Post-prune training
        # -------------------------
        train_metrics, plasticity_model, epochs_run = run_experiment(
            experiment_name,
            plasticity_model,
            train_loader,
            val_loader,
            test_loader,
            remaining_cycle_budget,
            learning_rate,
            start_epoch=1 + epochs_completed,
            metrics=train_metrics,
            run_test=True,
            beta=beta,
        )

        epochs_completed += epochs_run

    return train_metrics, plasticity_model

def main(save_path):
    # Hyperparameters
    num_epochs = 200
    cycles=4
    batch_size = 1024
    learning_rate = 0.001
    hidden_sizes = [512,256,128,64]
    #hidden_sizes = [16,16,16,16]
    
    beta=0.1
    prune_threshold=[0,0.25,0.5,0.75,1]
    #prune_threshold=[1.5,1.75,2,2.5,3]
    
    growth_epochs = num_epochs//3
    growth_rate = 2
    patience=3
    delta=0.005

    search_epochs = 15
    
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
    baseline_metrics, baseline_model, _= run_experiment(
        'baseline', 
        baseline_model, 
        train_loader, 
        val_loader, 
        test_loader, 
        num_epochs, 
        learning_rate,
        start_epoch=1,
        run_test=True,
        beta=beta
    )
    
    # # # ========== Experiment 2: Strong Baseline Model ==========
    # base_model = BayesianFNN(784, hidden_sizes, 10).to(device)
    # base_model.load_state_dict(initial_state_dict)
    # strong_baseline_metrics, _ = run_naive_plasticity_experiment(
    #     "strong_baseline",
    #     base_model,
    #     hidden_sizes,
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     num_epochs,
    #     learning_rate,
    #     beta,
    #     66,
    #     growth_rate,
    # )

    # # ========== Experiment 3: Multi-Growth  ==========
    
    # base_model = BayesianFNN(784, hidden_sizes, 10).to(device)
    # base_model.load_state_dict(initial_state_dict)
    # plasticity_multi_growth_metrics, _ = run_plasticity_experiment(
    #     "plasticity_multi_growth",
    #     base_model,
    #     hidden_sizes,
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     num_epochs,
    #     cycles,
    #     learning_rate,
    #     beta,
    #     growth_steps_per_cycle=3,
    #     adapt_epochs_per_growth=10,
    #     growth_rate=1,
    #     prune_thresholds=prune_threshold,
    #     search_epochs=search_epochs,
    #     warmup_epochs=5
    # )

    # # # ========== Experiment 4: Single Growth ==========

    # base_model = BayesianFNN(784, hidden_sizes, 10).to(device)
    # base_model.load_state_dict(initial_state_dict)
    # plasticity_single_growth_metrics, _ = run_plasticity_experiment(
    #     "plasticity_single_growth",
    #     base_model,
    #     hidden_sizes,
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     num_epochs,
    #     cycles,
    #     learning_rate,
    #     beta,
    #     growth_steps_per_cycle=1,
    #     adapt_epochs_per_growth=40,
    #     growth_rate=1.0,
    #     prune_thresholds=prune_threshold,
    #     search_epochs=search_epochs,
    #     warmup_epochs=10
    # )
    
    # # ========== Compare Results ==========
    # Combine all metrics
    all_metrics = {
         'baseline': baseline_metrics,
        # 'strong_baseline': strong_baseline_metrics,
        #'plasticity_multi_growth': plasticity_multi_growth_metrics,
        # 'plasticity_single_growth': plasticity_single_growth_metrics
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
        # {
        #     'Model': 'Strong Baseline',
        #     'Parameters': strong_baseline_metrics['param_count'],
        #     'Trainable Params': strong_baseline_metrics['trainable_param_count'],
        #     'Best Val Acc': max(strong_baseline_metrics['val_acc']),
        #     'Best Val Brier': min(strong_baseline_metrics['val_brier']),
        #     'Test Acc': strong_baseline_metrics['test_acc'],
        #     'Test Brier': strong_baseline_metrics['test_brier'],
        #     'Hidden Sizes': strong_baseline_metrics['hidden_sizes'],
        # },
        # {
        #     'Model': 'Plasticity Multi Growth',
        #     'Parameters': plasticity_multi_growth_metrics['param_count'],
        #     'Trainable Params': plasticity_multi_growth_metrics['trainable_param_count'],
        #     'Best Val Acc': max(plasticity_multi_growth_metrics['val_acc']),
        #     'Best Val Brier': min(plasticity_multi_growth_metrics['val_brier']),
        #     'Test Acc': plasticity_multi_growth_metrics['test_acc'],
        #     'Test Brier': plasticity_multi_growth_metrics['test_brier'],
        #     'Hidden Sizes': plasticity_multi_growth_metrics['hidden_sizes'],
        # },
        # {
        #     'Model': 'Plasticity Single Growth',
        #     'Parameters': plasticity_single_growth_metrics['param_count'],
        #     'Trainable Params': plasticity_single_growth_metrics['trainable_param_count'],
        #     'Best Val Acc': max(plasticity_single_growth_metrics['val_acc']),
        #     'Best Val Brier': min(plasticity_single_growth_metrics['val_brier']),
        #     'Test Acc': plasticity_single_growth_metrics['test_acc'],
        #     'Test Brier': plasticity_single_growth_metrics['test_brier'],
        #     'Hidden Sizes': plasticity_single_growth_metrics['hidden_sizes'],
        # }
    ])
    
    summary.to_csv(f'./{save_path}/experiment_summary.csv', index=False)
    print("\nExperiment Summary:")
    print(summary)
    return summary


def get_statisitcs(save_path="results/", configs=["underparametrized"], model_names=["Baseline", "Strong Baseline", "Plasticity Multi Growth", "Plasticity Single Growth"], metrics = ["Parameters", "Best Val Acc", "Best Val Brier", "Test Acc", "Test Brier"], num_runs=5): 
    num_runs = 5
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
                row[f"{metric} (Std)"] = np.std(metric_values)
            result.append(row)
        results[config] = pd.DataFrame(result)
    return results


df1=get_statisitcs(configs=["overparametrized"], model_names=["Baseline"])
print(df1["overparametrized"])
# for i in range(1,6):
#     print("Running experiment for run", i)
#     main(f"results/overparametrized/run_{i}")
# print("All experiments completed.")
