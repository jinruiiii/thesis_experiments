"""
Bayesian CNN plasticity experiments on CIFAR-10.

Ports the structural adaptation pipeline from main2.py to filter-based
growth/pruning on BayesianCNN.
"""

import copy
import math
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

from BayesianCNN import BayesianCNN
from main2 import (
    SEED,
    _annealed_decision_interval_length,
    _checkpoint_metric_label,
    _initial_decision_epoch,
    _is_better_checkpoint_score,
    _last_structural_epoch,
    _normalize_checkpoint_metric,
    _val_score_for_checkpoint,
    count_params,
    device,
    ensure_output_dir,
    penalised_elbo_on_batches,
    penalised_loss_function,
    plot_metrics,
    plot_param_count_vs_test_acc,
    sample_batches,
    set_seed,
    train,
    validate,
    write_hybrid_provenance,
    write_metrics_csv,
)

DATASET_ROOT = "../../Datasets"
DATASET_CONFIGS = {
    "cifar10": {
        "cls": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "in_channels": 3,
        "num_classes": 10,
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "in_channels": 1,
        "num_classes": 10,
    },
}


def _dataset_to_tensors(raw_dataset, mean, std):
    """Vectorized ToTensor + Normalize over the whole dataset (done once)."""
    data = raw_dataset.data
    if isinstance(data, torch.Tensor):
        # FashionMNIST: uint8 tensor [N, H, W]
        x = data.unsqueeze(1).float().div_(255.0)
    else:
        # CIFAR-10: uint8 numpy array [N, H, W, C]
        x = torch.from_numpy(data).permute(0, 3, 1, 2).float().div_(255.0)
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    x = x.sub_(mean_t).div_(std_t)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def build_dataloaders(dataset_name, batch_size, train_frac=0.8, seed=SEED):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"dataset must be one of {sorted(DATASET_CONFIGS)}, got {dataset_name!r}"
        )
    config = DATASET_CONFIGS[dataset_name]
    dataset_cls = config["cls"]

    training_data_raw = dataset_cls(root=DATASET_ROOT, train=True, download=True)
    test_data_raw = dataset_cls(root=DATASET_ROOT, train=False, download=True)

    x_all, y_all = _dataset_to_tensors(training_data_raw, config["mean"], config["std"])
    x_test, y_test = _dataset_to_tensors(test_data_raw, config["mean"], config["std"])

    train_size = int(train_frac * len(training_data_raw))
    val_size = len(training_data_raw) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(training_data_raw)),
        [train_size, val_size],
        generator=generator,
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


def _model_arch_metadata(model):
    return {
        "conv_channels": list(model.conv_channels),
        "in_channels": model.in_channels,
        "num_classes": model.num_classes,
        "kernel_size": model.kernel_size,
        "padding": model.padding,
        "stride": model.stride,
        "fc_hidden": int(getattr(model, "fc_hidden", model.fc.out_features)),
    }


def _bayesian_cnn_from_arch(
    in_channels,
    conv_channels,
    num_classes,
    kernel_size=3,
    padding=1,
    stride=1,
    fc_hidden=128,
    device=None,
):
    model = BayesianCNN(
        in_channels,
        list(conv_channels),
        num_classes,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        fc_hidden=fc_hidden,
    )
    if device is not None:
        model = model.to(device)
    return model


def _bayesian_cnn_like(model, conv_channels=None, device=None):
    if conv_channels is None:
        conv_channels = _conv_channels_from_model(model)
    if device is None:
        device = next(model.parameters()).device
    return _bayesian_cnn_from_arch(
        model.in_channels,
        conv_channels,
        model.num_classes,
        kernel_size=model.kernel_size,
        padding=model.padding,
        stride=model.stride,
        fc_hidden=getattr(model, "fc_hidden", model.fc.out_features),
        device=device,
    )


def save_checkpoint_cnn(
    path,
    state_dict,
    epoch,
    model=None,
    conv_channels=None,
    selection_metric=None,
    selection_value=None,
):
    payload = {
        "state_dict": state_dict,
        "epoch": int(epoch),
    }
    if model is not None:
        payload.update(_model_arch_metadata(model))
    if conv_channels is not None:
        payload["conv_channels"] = list(conv_channels)
    if selection_metric is not None:
        payload["selection_metric"] = str(selection_metric)
    if selection_value is not None:
        payload["selection_value"] = float(selection_value)
    torch.save(payload, path)


def load_checkpoint_cnn(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device)
    conv_channels = payload.get("conv_channels")
    if conv_channels is None:
        raise ValueError(f"checkpoint at {path!r} is missing conv_channels")
    model = _bayesian_cnn_from_arch(
        in_channels=payload.get("in_channels", 3),
        conv_channels=list(conv_channels),
        num_classes=payload.get("num_classes", 10),
        kernel_size=payload.get("kernel_size", 3),
        padding=payload.get("padding", 1),
        stride=payload.get("stride", 1),
        fc_hidden=payload.get("fc_hidden", 128),
        device=device,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    metadata = {
        "epoch": payload.get("epoch"),
        "selection_metric": payload.get("selection_metric"),
        "selection_value": payload.get("selection_value"),
    }
    return model, list(conv_channels), metadata


def write_experiment_summary_csv_cnn(
    output_dir,
    model_label,
    params,
    trainable_params,
    conv_channels,
    best_epoch,
    best_val_total,
    best_val_acc,
    best_val_brier,
    test_acc,
    test_brier,
    lambda_penalty,
    selected_checkpoint_metric="val_total",
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
        "Conv Channels": str(list(conv_channels)),
        "Lambda Penalty": float(lambda_penalty),
        "Junctures Mode": str(junctures_mode),
        "Selected checkpoint metric": str(selected_checkpoint_metric),
        "Selected epoch": int(best_epoch),
        "Selected val_total": float(best_val_total),
    }])
    summary_df.to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)


def _conv_channels_from_model(model):
    return [layer.out_channels for layer in model.conv_layers]


def _weight_snr(mu, rho, eps=1e-8):
    sigma = F.softplus(rho)
    return torch.abs(mu) / (sigma + eps)


def _filter_snr_incoming(conv_layer, eps=1e-8):
    snr_w = _weight_snr(conv_layer.mu_w, conv_layer.rho_w, eps)
    snr_w_mean = snr_w.mean(dim=(1, 2, 3))
    # if conv_layer.bias_flag:
    #     snr_b = torch.abs(conv_layer.mu_b) / (F.softplus(conv_layer.rho_b) + eps)
    #     return 0.5 * (snr_w_mean + snr_b)
    return snr_w_mean


def _filter_snr_outgoing(next_conv, filter_idx, eps=1e-8):
    snr = _weight_snr(next_conv.mu_w, next_conv.rho_w, eps)
    return snr[:, filter_idx, :, :].mean()


def _filter_snr_outgoing_linear(linear, filter_idx, eps=1e-8):
    snr = _weight_snr(linear.mu_w, linear.rho_w, eps)
    return snr[:, filter_idx].mean()


def _filter_snr_bidirectional(model, layer_idx, filter_idx, eps=1e-8, combine="geometric"):
    conv_layer = model.conv_layers[layer_idx]
    snr_in = _filter_snr_incoming(conv_layer, eps)[filter_idx]
    if layer_idx + 1 < len(model.conv_layers):
        snr_out = _filter_snr_outgoing(model.conv_layers[layer_idx + 1], filter_idx, eps)
    else:
        # Last conv feeds the hidden FC after adaptive average pooling.
        snr_out = _filter_snr_outgoing_linear(model.fc, filter_idx, eps)
    if combine == "min":
        return torch.min(snr_in, snr_out)
    if combine == "mean":
        return 0.5 * (snr_in + snr_out)
    return torch.sqrt(snr_in * snr_out + eps)


def _conv_filter_param_cost(model, layer_idx):
    conv = model.conv_layers[layer_idx]
    kH, kW = conv.kernel_size
    in_ch = conv.in_channels
    kernel_params = 2 * in_ch * kH * kW
    bias_params = 2
    gn_params = 2
    if layer_idx + 1 < len(model.conv_layers):
        next_conv = model.conv_layers[layer_idx + 1]
        kH2, kW2 = next_conv.kernel_size
        downstream = 2 * next_conv.out_channels * kH2 * kW2
    else:
        # Each filter connects into fc (prev_ch -> fc_hidden; mu + rho).
        downstream = 2 * int(model.fc.out_features)
    return kernel_params + bias_params + gn_params + downstream


def _median(values):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _percentile(values, p):
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
    mad_scale = 1.4826
    by_layer = {}
    for layer_idx, raw_scores in raw_scores_by_layer.items():
        n_filters = len(raw_scores)
        if n_filters == 0:
            by_layer[layer_idx] = []
            continue
        if n_filters == 1:
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


def _filter_snr_percentile_ranks(raw_scores_by_layer):
    ranks = {}
    for layer_idx, raw_scores in raw_scores_by_layer.items():
        n_filters = len(raw_scores)
        if n_filters == 0:
            continue
        if n_filters == 1:
            ranks[(layer_idx, 0)] = 0.0
            continue
        order = sorted(range(n_filters), key=lambda j: (raw_scores[j], j))
        for rank, j in enumerate(order):
            ranks[(layer_idx, j)] = rank / (n_filters - 1)
    return ranks


def _filter_snr_zscores(raw_scores_by_layer, eps=1e-8):
    zscores = {}
    for layer_idx, raw_scores in raw_scores_by_layer.items():
        n_filters = len(raw_scores)
        if n_filters == 0:
            continue
        if n_filters == 1:
            zscores[(layer_idx, 0)] = 0.0
            continue
        mean = sum(raw_scores) / n_filters
        variance = sum((score - mean) ** 2 for score in raw_scores) / n_filters
        std = math.sqrt(variance) + eps
        for j, score in enumerate(raw_scores):
            zscores[(layer_idx, j)] = (score - mean) / std
    return zscores


def _filter_snr_mad_scores(raw_scores_by_layer, eps=1e-8):
    mad_scores = {}
    by_layer = _robust_mad_zscores_by_layer(raw_scores_by_layer, eps=eps)
    for layer_idx, zscores in by_layer.items():
        for j, z in enumerate(zscores):
            mad_scores[(layer_idx, j)] = z
    return mad_scores


def _collect_conv_filter_uncertainty_scores(plasticity_model, uncertainty_combine="geometric"):
    raw_by_layer = {}
    for layer_idx, conv_layer in enumerate(plasticity_model.conv_layers):
        n_filters = conv_layer.out_channels
        raw_by_layer[layer_idx] = [
            float(
                plasticity_model._filter_uncertainty_bidirectional(
                    layer_idx, j, combine=uncertainty_combine
                ).item()
            )
            for j in range(n_filters)
        ]
    return raw_by_layer


def _layer_growth_scores_from_mad(raw_by_layer, percentile=100.0):
    percentile = _validate_growth_mad_percentile(percentile)
    by_layer_z = _robust_mad_zscores_by_layer(raw_by_layer)
    return {
        layer_idx: _percentile(zscores, percentile)
        for layer_idx, zscores in by_layer_z.items()
        if zscores
    }


def _collect_conv_filter_snr_scores(plasticity_model, snr_combine="geometric"):
    raw_by_layer = {}
    for layer_idx, conv_layer in enumerate(plasticity_model.conv_layers):
        n_filters = conv_layer.out_channels
        raw_by_layer[layer_idx] = [
            float(
                _filter_snr_bidirectional(
                    plasticity_model, layer_idx, j, combine=snr_combine
                ).item()
            )
            for j in range(n_filters)
        ]
    return raw_by_layer


def _filter_snr_raw(raw_scores_by_layer):
    scores = {}
    for layer_idx, raw_scores in raw_scores_by_layer.items():
        for j, score in enumerate(raw_scores):
            scores[(layer_idx, j)] = score
    return scores


def _layer_normalized_global_snr_scores(raw_scores_by_layer, global_prune_normalize="percentile"):
    if global_prune_normalize == "percentile":
        return _filter_snr_percentile_ranks(raw_scores_by_layer), "layer_percentile"
    if global_prune_normalize == "zscore":
        return _filter_snr_zscores(raw_scores_by_layer), "layer_zscore"
    if global_prune_normalize == "mad":
        return _filter_snr_mad_scores(raw_scores_by_layer), "layer_mad"
    if global_prune_normalize == "raw":
        return _filter_snr_raw(raw_scores_by_layer), "global_raw_snr"
    raise ValueError(
        f"global_prune_normalize must be 'percentile', 'zscore', 'mad', or 'raw', "
        f"got {global_prune_normalize!r}"
    )


def expand_and_load_conv_stack(old_sd, new_model):
    new_sd = new_model.state_dict()
    for k in new_sd.keys():
        if k not in old_sd:
            continue
        old_param = old_sd[k]
        new_param = new_sd[k]
        if old_param.shape == new_param.shape:
            new_sd[k] = old_param
        elif old_param.ndim == 4:
            new_sd[k][: old_param.shape[0], : old_param.shape[1], :, :] = old_param
        elif old_param.ndim == 2 and new_param.ndim == 2:
            new_sd[k][: old_param.shape[0], : old_param.shape[1]] = old_param
        elif old_param.ndim == 1:
            new_sd[k][: old_param.shape[0]] = old_param
    new_model.load_state_dict(new_sd, strict=True)


def truncate_and_load_conv_stack(old_sd, keep_dict, new_model):
    num_layers = len(keep_dict)
    new_sd = {}
    for i in range(num_layers):
        keep_i = keep_dict.get(i, None)
        keep_prev = keep_dict.get(i - 1, None)
        for p in ["mu_w", "rho_w", "mu_b", "rho_b"]:
            key = f"conv_layers.{i}.{p}"
            if key not in old_sd:
                raise ValueError(f"{key} is missing in the plasticity model")
            w = old_sd[key]
            if w.ndim == 4:
                if keep_i is not None:
                    w = w[keep_i, :, :, :]
                if keep_prev is not None:
                    w = w[:, keep_prev, :, :]
            else:
                if keep_i is not None:
                    w = w[keep_i]
            new_sd[key] = w

        for p in ["weight", "bias"]:
            key = f"gn_layers.{i}.{p}"
            if key not in old_sd:
                raise ValueError(f"{key} is missing in the plasticity model")
            w = old_sd[key]
            if keep_i is not None:
                w = w[keep_i]
            new_sd[key] = w

    keep_last = keep_dict.get(num_layers - 1, None)
    for p in ["mu_w", "rho_w", "mu_b", "rho_b"]:
        key = f"fc.{p}"
        if key not in old_sd:
            raise ValueError(f"{key} is missing in the plasticity model")
        w = old_sd[key]
        if w.ndim == 2 and keep_last is not None:
            w = w[:, keep_last]
        new_sd[key] = w

    for p in ["mu_w", "rho_w", "mu_b", "rho_b"]:
        key = f"classifier.{p}"
        if key not in old_sd:
            raise ValueError(f"{key} is missing in the plasticity model")
        new_sd[key] = old_sd[key]

    new_model.load_state_dict(new_sd, strict=True)


def build_pruned_model(model, keep_dict):
    conv_channels = [len(v) for v in keep_dict.values()]
    pruned_model = _bayesian_cnn_like(model, conv_channels=conv_channels)
    truncate_and_load_conv_stack(model.state_dict(), keep_dict, pruned_model)
    return pruned_model, conv_channels


def _mask_warm_start_grads_grow_new_only_cnn(model, layer_idx, old_width):
    if layer_idx is None or old_width is None:
        raise ValueError("layer_idx and old_width must be provided for grow_new_only masking")
    if old_width <= 0:
        return

    def _zero_grad_full(p):
        if p is not None and getattr(p, "grad", None) is not None:
            p.grad.zero_()

    def _zero_grad_filter_rows(p, n_rows):
        if p is not None and getattr(p, "grad", None) is not None:
            if p.grad.ndim == 4:
                p.grad[:n_rows, :, :, :].zero_()
            elif p.grad.ndim == 1:
                p.grad[:n_rows].zero_()

    def _zero_grad_input_channels(p, n_cols):
        if p is not None and getattr(p, "grad", None) is not None:
            if p.grad.ndim == 4:
                p.grad[:, :n_cols, :, :].zero_()
            elif p.grad.ndim == 2:
                p.grad[:, :n_cols].zero_()

    num_conv = len(model.conv_layers)

    for i, conv in enumerate(model.conv_layers):
        if i == layer_idx:
            _zero_grad_filter_rows(getattr(conv, "mu_w", None), old_width)
            _zero_grad_filter_rows(getattr(conv, "rho_w", None), old_width)
            _zero_grad_filter_rows(getattr(conv, "mu_b", None), old_width)
            _zero_grad_filter_rows(getattr(conv, "rho_b", None), old_width)
        elif i == layer_idx + 1:
            _zero_grad_input_channels(getattr(conv, "mu_w", None), old_width)
            _zero_grad_input_channels(getattr(conv, "rho_w", None), old_width)
            _zero_grad_full(getattr(conv, "mu_b", None))
            _zero_grad_full(getattr(conv, "rho_b", None))
        else:
            _zero_grad_full(getattr(conv, "mu_w", None))
            _zero_grad_full(getattr(conv, "rho_w", None))
            _zero_grad_full(getattr(conv, "mu_b", None))
            _zero_grad_full(getattr(conv, "rho_b", None))

    for i, gn in enumerate(model.gn_layers):
        if i == layer_idx:
            _zero_grad_filter_rows(getattr(gn, "weight", None), old_width)
            _zero_grad_filter_rows(getattr(gn, "bias", None), old_width)
        else:
            _zero_grad_full(getattr(gn, "weight", None))
            _zero_grad_full(getattr(gn, "bias", None))

    if layer_idx == num_conv - 1:
        _zero_grad_input_channels(getattr(model.fc, "mu_w", None), old_width)
        _zero_grad_input_channels(getattr(model.fc, "rho_w", None), old_width)
        _zero_grad_full(getattr(model.fc, "mu_b", None))
        _zero_grad_full(getattr(model.fc, "rho_b", None))
    else:
        _zero_grad_full(getattr(model.fc, "mu_w", None))
        _zero_grad_full(getattr(model.fc, "rho_w", None))
        _zero_grad_full(getattr(model.fc, "mu_b", None))
        _zero_grad_full(getattr(model.fc, "rho_b", None))

    _zero_grad_full(getattr(model.classifier, "mu_w", None))
    _zero_grad_full(getattr(model.classifier, "rho_w", None))
    _zero_grad_full(getattr(model.classifier, "mu_b", None))
    _zero_grad_full(getattr(model.classifier, "rho_b", None))


def warm_start_model_on_batches_cnn(
    model,
    batches_ws,
    K,
    eta_ws,
    beta_scaled,
    lambda_penalty,
    mask_mode=None,
    layer_idx=None,
    old_width=None,
):
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
            _mask_warm_start_grads_grow_new_only_cnn(
                model, layer_idx=layer_idx, old_width=old_width
            )
        optimizer.step()


def filtergenesis(
    plasticity_original,
    conv_channels,
    exclude=None,
    gamma=0.1,
    uncertainty_combine="geometric",
    growth_layer_score="mean",
    growth_mad_percentile=100.0,
):
    growth_layer_score = _normalize_growth_layer_score(growth_layer_score)
    growth_mad_percentile = _validate_growth_mad_percentile(growth_mad_percentile)
    if exclude is None:
        exclude = []
    n_layers = len(conv_channels)
    eligible = [i for i in range(n_layers) if i not in exclude]
    if not eligible:
        print("[filtergenesis] All layers excluded; ignoring exclude list for this step.")
        eligible = list(range(n_layers))

    if growth_layer_score == "mean":
        uncertainty = plasticity_original.get_average_bidirectional_uncertainty_per_layer(
            combine=uncertainty_combine
        )
        layer_scores = {
            i: uncertainty[i].item() / (conv_channels[i] ** 0.5)
            for i in range(n_layers)
        }
        print("\n Average Bidirectional Normalised Uncertainty per Conv Layer:")
        for i, val in enumerate(uncertainty):
            print(f"  Layer {i+1}: {val.item()/(conv_channels[i]**0.5):.6f}")
        score_label = "normalised mean uncertainty"
    else:
        raw = _collect_conv_filter_uncertainty_scores(
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
    filters_to_add = max(1, math.ceil(gamma * conv_channels[layer_to_expand]))
    old_width = conv_channels[layer_to_expand]
    print(
        f"Expanding Conv Layer {layer_to_expand+1} "
        f"(highest {score_label}: {layer_scores[layer_to_expand]:.6f}) "
        f"by {filters_to_add} filters"
    )

    expanded_conv_channels = conv_channels.copy()
    expanded_conv_channels[layer_to_expand] += filters_to_add
    model_device = next(plasticity_original.parameters()).device
    plasticity_filtergenesis = BayesianCNN(
        plasticity_original.in_channels,
        expanded_conv_channels,
        plasticity_original.num_classes,
        kernel_size=plasticity_original.kernel_size,
        padding=plasticity_original.padding,
        stride=plasticity_original.stride,
        fc_hidden=getattr(plasticity_original, "fc_hidden", plasticity_original.fc.out_features),
    ).to(model_device)
    return plasticity_filtergenesis, expanded_conv_channels, layer_to_expand, old_width


def _prune_stats_from_keep_dict(plasticity_model, keep_dict, extra=None):
    filters_removed_per_layer = {}
    total_removed = 0
    for layer_idx, conv_layer in enumerate(plasticity_model.conv_layers):
        old_n = conv_layer.out_channels
        new_n = len(keep_dict[layer_idx])
        removed = old_n - new_n
        filters_removed_per_layer[layer_idx] = removed
        total_removed += removed
    stats = {
        "filters_removed_per_layer": filters_removed_per_layer,
        "filters_pruned": total_removed,
        "neurons_pruned": total_removed,
    }
    if extra:
        stats.update(extra)
    return stats


def _print_prune_candidate_summary(plasticity_model, keep_dict, prune_mode, prune_rate):
    print(f"\nPruning candidate (mode={prune_mode}, rate={prune_rate}):")
    for layer_idx, conv_layer in enumerate(plasticity_model.conv_layers):
        old_n = conv_layer.out_channels
        new_n = len(keep_dict[layer_idx])
        removed = old_n - new_n
        if removed > 0:
            print(f"  Conv Layer {layer_idx + 1}: removing {removed} filters ({old_n} -> {new_n})")
        else:
            print(f"  Conv Layer {layer_idx + 1}: no filters removed ({old_n})")


def _filterapoptosis_per_layer(
    plasticity_model,
    prune_rate,
    exclude=None,
    snr_combine="geometric",
    min_filters_per_layer=2,
):
    if exclude is None:
        exclude = []

    keep_dict = {}
    for i, conv_layer in enumerate(plasticity_model.conv_layers):
        n_filters = conv_layer.out_channels
        if i in exclude:
            keep_dict[i] = list(range(n_filters))
            continue
        if n_filters <= min_filters_per_layer:
            keep_dict[i] = list(range(n_filters))
            continue

        scores = []
        for j in range(n_filters):
            s = _filter_snr_bidirectional(plasticity_model, i, j, combine=snr_combine)
            scores.append(float(s.item()))

        num_prune = int(prune_rate * n_filters)
        if num_prune < 1:
            keep_dict[i] = list(range(n_filters))
            continue
        max_prune_allowed = n_filters - min_filters_per_layer
        if max_prune_allowed <= 0:
            keep_dict[i] = list(range(n_filters))
            continue
        num_prune = min(num_prune, max_prune_allowed)

        filter_indices = list(range(n_filters))
        filter_indices.sort(key=lambda j: scores[j])
        pruned = set(filter_indices[:num_prune])
        kept = [j for j in range(n_filters) if j not in pruned]
        if len(kept) < min_filters_per_layer:
            return None, {}
        keep_dict[i] = kept

    if all(
        len(keep_dict[i]) == plasticity_model.conv_layers[i].out_channels
        for i in range(len(plasticity_model.conv_layers))
    ):
        return None, {}

    if any(len(v) == 0 for v in keep_dict.values()):
        return None, {}

    prune_stats = _prune_stats_from_keep_dict(
        plasticity_model, keep_dict, extra={"snr_normalize": "per_layer"}
    )
    return keep_dict, prune_stats


def _filterapoptosis_global_param(
    plasticity_model,
    prune_rate,
    exclude=None,
    snr_combine="geometric",
    min_filters_per_layer=2,
    global_prune_budget="params",
    global_prune_normalize="percentile",
):
    if global_prune_budget not in ("params", "filters"):
        raise ValueError(
            f"global_prune_budget must be 'params' or 'filters', got {global_prune_budget!r}"
        )
    if exclude is None:
        exclude = []

    if global_prune_budget == "filters":
        budget_total = sum(layer.out_channels for layer in plasticity_model.conv_layers)
    else:
        budget_total = count_params(plasticity_model)
    target_remove = int(round(prune_rate * budget_total))
    if target_remove < 1:
        return None, {}

    raw_scores_by_layer = _collect_conv_filter_snr_scores(plasticity_model, snr_combine)
    norm_scores, snr_normalize = _layer_normalized_global_snr_scores(
        raw_scores_by_layer, global_prune_normalize
    )

    candidates = []
    for layer_idx, conv_layer in enumerate(plasticity_model.conv_layers):
        n_filters = conv_layer.out_channels
        if layer_idx in exclude or n_filters <= min_filters_per_layer:
            continue
        param_cost = _conv_filter_param_cost(plasticity_model, layer_idx)
        for filter_idx in range(n_filters):
            snr = raw_scores_by_layer[layer_idx][filter_idx]
            norm_score = norm_scores[(layer_idx, filter_idx)]
            candidates.append((norm_score, snr, layer_idx, filter_idx, param_cost))

    if not candidates:
        return None, {}

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    pruned_per_layer = {i: set() for i in range(len(plasticity_model.conv_layers))}
    params_removed = 0
    filters_pruned = 0
    budget_consumed = 0
    unit_cost = 1 if global_prune_budget == "filters" else None

    for norm_score, snr, layer_idx, filter_idx, param_cost in candidates:
        step_cost = unit_cost if unit_cost is not None else param_cost
        if budget_consumed + step_cost > target_remove:
            continue
        n_filters = plasticity_model.conv_layers[layer_idx].out_channels
        n_pruned = len(pruned_per_layer[layer_idx])
        if n_filters - n_pruned <= min_filters_per_layer:
            continue
        pruned_per_layer[layer_idx].add(filter_idx)
        params_removed += param_cost
        filters_pruned += 1
        budget_consumed += step_cost

    if filters_pruned == 0:
        return None, {}

    keep_dict = {}
    for layer_idx, conv_layer in enumerate(plasticity_model.conv_layers):
        n_filters = conv_layer.out_channels
        pruned = pruned_per_layer[layer_idx]
        kept = [j for j in range(n_filters) if j not in pruned]
        if len(kept) < min_filters_per_layer:
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


def filterapoptosis(
    plasticity_model,
    prune_rate,
    exclude=None,
    snr_combine="geometric",
    min_filters_per_layer=0,
    prune_mode="per_layer",
    global_prune_budget="params",
    global_prune_normalize="percentile",
):
    if prune_mode == "per_layer":
        keep_dict, prune_stats = _filterapoptosis_per_layer(
            plasticity_model,
            prune_rate,
            exclude=exclude,
            snr_combine=snr_combine,
            min_filters_per_layer=min_filters_per_layer,
        )
    elif prune_mode == "global_param":
        keep_dict, prune_stats = _filterapoptosis_global_param(
            plasticity_model,
            prune_rate,
            exclude=exclude,
            snr_combine=snr_combine,
            min_filters_per_layer=min_filters_per_layer,
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


def structural_decision_juncture_cnn(
    model,
    conv_channels,
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
    decision_epsilon=0.0,
):
    if junctures_mode not in ("both", "grow", "prune"):
        raise ValueError(
            f"junctures_mode must be 'both', 'grow', or 'prune', got {junctures_mode!r}"
        )
    decision_epsilon = float(decision_epsilon)
    if decision_epsilon < 0:
        raise ValueError(
            f"decision_epsilon must be >= 0, got {decision_epsilon!r}"
        )
    train_dataset_size = len(train_loader.dataset)
    beta_scaled = (1 / train_dataset_size) * beta

    M_ws = 40
    M_val = 10
    batches_ws = sample_batches(train_loader, device, M_ws)
    batches_val = sample_batches(val_loader, device, M_val)

    L_before = penalised_elbo_on_batches(model, batches_val, beta_scaled, lambda_penalty)

    none_model = copy.deepcopy(model)
    warm_start_model_on_batches_cnn(
        none_model, batches_ws, K, eta_ws, beta_scaled, lambda_penalty
    )
    L_after_none = penalised_elbo_on_batches(
        none_model, batches_val, beta_scaled, lambda_penalty
    )

    info = {
        "L_before": L_before,
        "L_after_none": L_after_none,
        "delta_grow": None,
        "delta_prune": None,
        "param_count_before": count_params(model),
        "prune_mode": prune_mode,
        "global_prune_budget": global_prune_budget,
        "global_prune_normalize": global_prune_normalize,
        "prune_target_remove": None,
        "prune_params_removed": None,
        "prune_neurons_pruned": None,
        "growth_layer_score": growth_layer_score,
        "growth_mad_percentile": growth_mad_percentile,
        "decision_epsilon": decision_epsilon,
    }
    if grow_exclude_layers is None:
        grow_exclude_layers = []

    grow_model = None
    conv_channels_g = None
    layer_idx = None
    delta_grow = float("-inf")

    if junctures_mode in ("both", "grow"):
        grow_model, conv_channels_g, layer_idx, old_width = filtergenesis(
            model,
            conv_channels,
            exclude=list(grow_exclude_layers),
            gamma=gamma,
            uncertainty_combine=uncertainty_combine,
            growth_layer_score=growth_layer_score,
            growth_mad_percentile=growth_mad_percentile,
        )
        expand_and_load_conv_stack(model.state_dict(), grow_model)
        warm_start_model_on_batches_cnn(
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
        delta_grow = L_after_grow - L_after_none
        info["delta_grow"] = delta_grow

    delta_prune = float("-inf")
    prune_model = None
    conv_channels_p = None
    keep_dict = None

    if junctures_mode in ("both", "prune"):
        keep_dict, prune_stats = filterapoptosis(
            model,
            rho,
            snr_combine=snr_combine,
            prune_mode=prune_mode,
            min_filters_per_layer=2,
            global_prune_budget=global_prune_budget,
            global_prune_normalize=global_prune_normalize,
        )
        if prune_stats:
            info["prune_target_remove"] = prune_stats.get("target_remove")
            info["prune_params_removed"] = prune_stats.get("params_removed")
            info["prune_neurons_pruned"] = prune_stats.get("filters_pruned")
            info["prune_budget_unit"] = prune_stats.get("budget_unit", global_prune_budget)
            info["prune_snr_normalize"] = prune_stats.get("snr_normalize", global_prune_normalize)
        if keep_dict is not None:
            prune_model, conv_channels_p = build_pruned_model(model, keep_dict)
            warm_start_model_on_batches_cnn(
                prune_model, batches_ws, K, eta_ws, beta_scaled, lambda_penalty
            )
            L_after_prune = penalised_elbo_on_batches(
                prune_model, batches_val, beta_scaled, lambda_penalty
            )
            delta_prune = L_after_prune - L_after_none
        info["delta_prune"] = delta_prune if keep_dict is not None else None

    best_action = "none"
    best_model = model
    best_conv_channels = conv_channels

    delta_prune_val = delta_prune if keep_dict is not None else float("-inf")
    if junctures_mode == "both":
        if delta_grow > decision_epsilon or delta_prune_val > decision_epsilon:
            if delta_grow >= delta_prune_val:
                best_action = "grow"
                best_model = grow_model
                best_conv_channels = conv_channels_g
            else:
                best_action = "prune"
                best_model = prune_model
                best_conv_channels = conv_channels_p
    elif junctures_mode == "grow" and delta_grow > decision_epsilon:
        best_action = "grow"
        best_model = grow_model
        best_conv_channels = conv_channels_g
    elif junctures_mode == "prune" and delta_prune_val > decision_epsilon:
        best_action = "prune"
        best_model = prune_model
        best_conv_channels = conv_channels_p

    info["grow_layer_idx"] = layer_idx
    info["grow_exclude_layers"] = list(grow_exclude_layers)
    info["junctures_mode"] = junctures_mode
    info["action"] = best_action
    info["param_count_after"] = (
        count_params(best_model) if best_action != "none" else info["param_count_before"]
    )
    delta_grow_str = f"{delta_grow:.4f}" if info["delta_grow"] is not None else "N/A"
    prune_log = ""
    if info.get("prune_neurons_pruned") is not None:
        budget_unit = info.get("prune_budget_unit", "params")
        if budget_unit == "filters":
            prune_log = (
                f", filter_budget={info['prune_target_remove']}, "
                f"pruned={info['prune_neurons_pruned']} filters "
                f"({info['prune_params_removed']} params)"
            )
        else:
            prune_log = (
                f", param_budget={info['prune_target_remove']}, "
                f"pruned={info['prune_params_removed']} params "
                f"({info['prune_neurons_pruned']} filters)"
            )
    print(
        f"\nStructural decision (mode={junctures_mode}, prune_mode={prune_mode}, "
        f"global_prune_budget={global_prune_budget}, "
        f"global_prune_normalize={global_prune_normalize}, "
        f"decision_epsilon={decision_epsilon}): "
        f"L_before={L_before:.4f}, L_after_none={L_after_none:.4f}, "
        f"delta_grow={delta_grow_str}, delta_prune={info['delta_prune']}, "
        f"action={best_action}{prune_log}"
    )
    return best_action, best_model, best_conv_channels, info


def build_grown_model_at_conv_layer(model, conv_channels, growth_layer_idx, growth_gamma):
    conv_channels = _conv_channels_from_model(model)
    growth_layer_idx = int(growth_layer_idx)
    growth_gamma = float(growth_gamma)
    if growth_layer_idx < 0 or growth_layer_idx >= len(conv_channels):
        raise ValueError(
            f"growth_layer_idx must be in [0, {len(conv_channels) - 1}], got {growth_layer_idx}"
        )
    if growth_gamma <= 0:
        raise ValueError(f"growth_gamma must be positive, got {growth_gamma!r}")

    old_width = conv_channels[growth_layer_idx]
    filters_to_add = max(1, math.ceil(growth_gamma * old_width))
    grown_conv_channels = conv_channels.copy()
    grown_conv_channels[growth_layer_idx] += filters_to_add

    model_device = next(model.parameters()).device
    grown_model = BayesianCNN(
        model.in_channels,
        grown_conv_channels,
        model.num_classes,
        kernel_size=model.kernel_size,
        padding=model.padding,
        stride=model.stride,
        fc_hidden=getattr(model, "fc_hidden", model.fc.out_features),
    ).to(model_device)
    expand_and_load_conv_stack(model.state_dict(), grown_model)
    return grown_model, grown_conv_channels, old_width, filters_to_add


def build_three_phase_pruned_model(model, grown_conv_channels, growth_layer_idx, old_width):
    keep_dict = {}
    for layer_idx, width in enumerate(grown_conv_channels):
        if layer_idx == growth_layer_idx:
            keep_dict[layer_idx] = list(range(old_width))
        else:
            keep_dict[layer_idx] = list(range(width))
    return build_pruned_model(model, keep_dict)


def _initialise_standard_metrics():
    return {
        "train_loss_total": [],
        "train_loss_nll": [],
        "train_loss_kl": [],
        "train_loss_penalty": [],
        "train_acc": [],
        "train_brier": [],
        "val_loss_total": [],
        "val_loss_nll": [],
        "val_loss_kl": [],
        "val_loss_penalty": [],
        "val_acc": [],
        "val_brier": [],
        "param_count_history": [],
    }


def _experiment_dir_suffix(junctures_mode):
    return "" if junctures_mode == "both" else f"_{junctures_mode}_only"


def _format_lambda_dir(lambda_penalty):
    return f"{float(lambda_penalty):g}"


def _conv_channels_tag(conv_channels):
    return "_".join(f"{c}f" for c in conv_channels)


def _resolve_plasticity_checkpoint_dir(resume_from_plasticity_dir):
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
    if "Conv Channels" in row and not pd.isna(row["Conv Channels"]):
        metrics["conv_channels"] = row["Conv Channels"]
    return metrics


def _hybrid_output_dir_for_plasticity(plasticity_output_dir, save_path, conv_tag, lam_tag, suffix):
    plasticity_dir_name = os.path.basename(plasticity_output_dir)
    if plasticity_dir_name.startswith("plasticity_"):
        hybrid_name = "plasticity_three_phase_" + plasticity_dir_name[len("plasticity_"):]
        return os.path.join(os.path.dirname(plasticity_output_dir), hybrid_name)
    return os.path.join(
        save_path,
        f"plasticity_three_phase_{conv_tag}_{lam_tag}{suffix}",
    )


def _make_eval_model(model, conv_channels):
    eval_model = BayesianCNN(
        model.in_channels,
        conv_channels,
        model.num_classes,
        kernel_size=model.kernel_size,
        padding=model.padding,
        stride=model.stride,
        fc_hidden=getattr(model, "fc_hidden", model.fc.out_features),
    ).to(device)
    return eval_model


def run_experiment_cnn(
    experiment_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    learning_rate=0.005,
    start_epoch=1,
    metrics=None,
    beta=0.1,
    lambda_penalty=0,
    output_dir=None,
    checkpoint_metric="val_loss_total",
):
    if output_dir is None:
        output_dir = os.path.join("./results", experiment_name)
    ensure_output_dir(output_dir)

    print(f"\n{'-'*20} Running {experiment_name} experiment {'-'*20}")
    param_stats = model.get_param_stats()
    print(f"Model parameters: {param_stats['total_params']:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    if not metrics:
        metrics = _initialise_standard_metrics()

    loss_label = "Penalised ELBO" if lambda_penalty > 0 else "ELBO"
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    best_checkpoint_score = float("inf")
    best_model_state = None
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    last_epoch = best_epoch = start_epoch - 1

    for epoch in range(start_epoch, start_epoch + num_epochs):
        last_epoch = epoch
        train_loss_total, train_acc, train_loss_nll, train_loss_kl, train_brier, train_loss_penalty = train(
            model, train_loader, optimizer, epoch, device, beta_scaled, lambda_penalty
        )
        metrics["train_loss_total"].append(train_loss_total)
        metrics["train_loss_nll"].append(train_loss_nll)
        metrics["train_loss_kl"].append(train_loss_kl)
        metrics["train_acc"].append(train_acc)
        metrics["train_brier"].append(train_brier)
        metrics["train_loss_penalty"].append(train_loss_penalty)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics["val_loss_total"].append(val_loss_total)
        metrics["val_loss_nll"].append(val_loss_nll)
        metrics["val_loss_kl"].append(val_loss_kl)
        metrics["val_acc"].append(val_acc)
        metrics["val_brier"].append(val_brier)
        metrics["val_loss_penalty"].append(val_loss_penalty)
        metrics["param_count_history"].append(count_params(model))

        penalty_str = f", Penalty={train_loss_penalty:.4f}" if lambda_penalty > 0 else ""
        print(
            f"Epoch {epoch}: Train Loss({loss_label})={train_loss_total:.4f}, "
            f"Train Loss(NLL)={train_loss_nll:.4f}, Train Loss(KL)={train_loss_kl:.4f}"
            f"{penalty_str}, Train Acc={train_acc:.2f}%, "
            f"Val Loss({loss_label})={val_loss_total:.4f}, Val Acc={val_acc:.2f}%"
        )

        val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
        if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
            best_checkpoint_score = val_score
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint_cnn(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                model=model,
                selection_metric=checkpoint_metric_label,
                selection_value=best_checkpoint_score,
            )

    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, "metrics.png"),
    )

    eval_model = model
    if best_model_state is not None:
        eval_model = copy.deepcopy(model)
        eval_model.load_state_dict(best_model_state)
        print(f"Loaded best model from Epoch {best_epoch} for final testing.")

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f"Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}")

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint_cnn(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else last_epoch,
        model=eval_model,
    )

    conv_channels = _conv_channels_from_model(eval_model)
    metrics.update({
        "test_acc": test_acc,
        "test_loss_total": test_loss_total,
        "test_loss_nll": test_loss_nll,
        "test_loss_kl": test_loss_kl,
        "test_brier": test_brier,
        "param_count": count_params(eval_model),
        "trainable_param_count": eval_model.get_param_stats()["trainable_params"],
        "conv_channels": conv_channels,
    })

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv_cnn(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        conv_channels=metrics["conv_channels"],
        best_epoch=best_epoch,
        best_val_total=best_checkpoint_score,
        best_val_acc=max(metrics["val_acc"]),
        best_val_brier=min(metrics["val_brier"]),
        test_acc=metrics["test_acc"],
        test_brier=metrics["test_brier"],
        lambda_penalty=lambda_penalty,
        selected_checkpoint_metric=checkpoint_metric_label,
    )

    epochs_run = last_epoch - start_epoch + 1
    return metrics, model, epochs_run


def run_adaptive_experiment_cnn(
    experiment_name,
    model,
    conv_channels,
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
    decision_epsilon=0.0,
):
    if output_dir is None:
        output_dir = os.path.join("./results", experiment_name)
    ensure_output_dir(output_dir)

    print("\n" + "=" * 50)
    print(
        f"Training {experiment_name.upper()} CNN (Penalised ELBO, lambda={lambda_penalty}, "
        f"junctures_mode={junctures_mode})"
    )
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    print(f"Checkpoint selection metric: {checkpoint_metric_label}")
    print("=" * 50)

    param_stats = model.get_param_stats()
    print(f"Model parameters: {param_stats['total_params']:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    metrics = _initialise_standard_metrics()
    metrics.update({
        "structural_epochs": [],
        "structural_actions": [],
        "structural_delta_grow": [],
        "structural_delta_prune": [],
        "structural_L_before": [],
        "structural_L_after_none": [],
        "structural_conv_channels": [],
        "structural_prune_mode": [],
        "structural_prune_target_remove": [],
        "structural_prune_params_removed": [],
        "structural_prune_filters_pruned": [],
        "structural_global_prune_budget": [],
        "structural_global_prune_normalize": [],
        "junctures_mode": junctures_mode,
        "decision_epsilon": float(decision_epsilon),
    })

    growth_cooldown = {}
    best_checkpoint_score = float("inf")
    best_model_state = None
    best_epoch = 0
    best_conv_channels = list(conv_channels)
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    decision_warmup_epochs = max(0, int(decision_warmup_epochs))
    decision_cooldown_epochs = max(0, int(decision_cooldown_epochs))
    last_structural_epoch = _last_structural_epoch(num_epochs, decision_cooldown_epochs)
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
        metrics["train_loss_total"].append(train_loss_total)
        metrics["train_loss_nll"].append(train_loss_nll)
        metrics["train_loss_kl"].append(train_loss_kl)
        metrics["train_loss_penalty"].append(train_loss_penalty)
        metrics["train_acc"].append(train_acc)
        metrics["train_brier"].append(train_brier)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics["val_loss_total"].append(val_loss_total)
        metrics["val_loss_nll"].append(val_loss_nll)
        metrics["val_loss_kl"].append(val_loss_kl)
        metrics["val_loss_penalty"].append(val_loss_penalty)
        metrics["val_acc"].append(val_acc)
        metrics["val_brier"].append(val_brier)
        metrics["param_count_history"].append(count_params(model))

        print(
            f"Epoch {epoch}: Train Loss(Penalised ELBO)={train_loss_total:.4f}, "
            f"Train Acc={train_acc:.2f}%, Params={count_params(model):,}, "
            f"Val Loss(Penalised ELBO)={val_loss_total:.4f}, Val Acc={val_acc:.2f}%"
        )

        val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
        if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
            best_checkpoint_score = val_score
            best_epoch = epoch
            best_conv_channels = list(conv_channels)
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint_cnn(
                os.path.join(output_dir, "best_checkpoint.pth"),
                state_dict=best_model_state,
                epoch=best_epoch,
                model=model,
                conv_channels=best_conv_channels,
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
            action, model, conv_channels, info = structural_decision_juncture_cnn(
                model,
                conv_channels,
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
                decision_epsilon=decision_epsilon,
            )
            metrics["structural_epochs"].append(epoch)
            metrics["structural_actions"].append(action)
            metrics["structural_delta_grow"].append(info.get("delta_grow"))
            metrics["structural_delta_prune"].append(info.get("delta_prune"))
            metrics["structural_L_before"].append(info.get("L_before"))
            metrics["structural_L_after_none"].append(info.get("L_after_none"))
            metrics["structural_conv_channels"].append(list(conv_channels))
            metrics["structural_prune_mode"].append(info.get("prune_mode"))
            metrics["structural_prune_target_remove"].append(info.get("prune_target_remove"))
            metrics["structural_prune_params_removed"].append(info.get("prune_params_removed"))
            metrics["structural_prune_filters_pruned"].append(info.get("prune_neurons_pruned"))
            metrics["structural_global_prune_budget"].append(info.get("global_prune_budget"))
            metrics["structural_global_prune_normalize"].append(
                info.get("global_prune_normalize")
            )

            if action != "none":
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

    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, "metrics.png"),
    )

    if best_model_state is not None:
        eval_model = _make_eval_model(model, best_conv_channels)
        eval_model.load_state_dict(best_model_state, strict=True)
        print(
            f"Loaded best model from epoch {best_epoch} "
            f"(conv_channels={best_conv_channels}) for final testing."
        )
    else:
        eval_model = model

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f"Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}")

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint_cnn(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else num_epochs,
        model=eval_model,
    )

    final_conv_channels = _conv_channels_from_model(eval_model)
    metrics.update({
        "test_acc": test_acc,
        "test_loss_total": test_loss_total,
        "test_loss_nll": test_loss_nll,
        "test_loss_kl": test_loss_kl,
        "test_brier": test_brier,
        "param_count": count_params(eval_model),
        "trainable_param_count": eval_model.get_param_stats()["trainable_params"],
        "conv_channels": final_conv_channels,
    })

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv_cnn(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        conv_channels=metrics["conv_channels"],
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
        "epoch": metrics["structural_epochs"],
        "action": metrics["structural_actions"],
        "delta_grow": metrics["structural_delta_grow"],
        "delta_prune": metrics["structural_delta_prune"],
        "L_before": metrics["structural_L_before"],
        "L_after_none": metrics["structural_L_after_none"],
        "conv_channels": [str(cc) for cc in metrics["structural_conv_channels"]],
        "junctures_mode": metrics["junctures_mode"],
        "decision_epsilon": metrics["decision_epsilon"],
        "prune_mode": metrics["structural_prune_mode"],
        "prune_target_remove": metrics["structural_prune_target_remove"],
        "prune_params_removed": metrics["structural_prune_params_removed"],
        "filters_pruned": metrics["structural_prune_filters_pruned"],
        "global_prune_budget": metrics["structural_global_prune_budget"],
        "global_prune_normalize": metrics["structural_global_prune_normalize"],
    })
    structural_df.to_csv(os.path.join(output_dir, "structural_decisions.csv"), index=False)

    print(f"\n{experiment_name} Summary:")
    print(f"Best validation accuracy: {max(metrics['val_acc']):.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final conv channels: {final_conv_channels}")
    print(f"Structural decisions: {metrics['structural_actions']}")

    return metrics, model, num_epochs


def run_three_phase_baseline_cnn(
    experiment_name,
    model,
    conv_channels,
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
    if output_dir is None:
        output_dir = os.path.join("./results", experiment_name)
    ensure_output_dir(output_dir)

    phase1_epochs = int(phase1_epochs)
    phase2_epochs = int(phase2_epochs)
    phase3_epochs = int(phase3_epochs)
    total_epochs = phase1_epochs + phase2_epochs + phase3_epochs

    conv_channels = _conv_channels_from_model(model)
    checkpoint_metric = _normalize_checkpoint_metric(checkpoint_metric)
    checkpoint_metric_label = _checkpoint_metric_label(checkpoint_metric)
    beta_scaled = (1 / len(train_loader.dataset)) * beta
    loss_label = "Penalised ELBO" if lambda_penalty > 0 else "ELBO"

    print(f"\n{'-'*20} Running {experiment_name} CNN experiment {'-'*20}")
    print(f"Initial conv channels: {conv_channels}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    metrics = _initialise_standard_metrics()
    metrics["phase_history"] = []
    metrics["conv_channel_history"] = []

    best_checkpoint_score = float("inf")
    best_model_state = None
    best_conv_channels = list(conv_channels)
    best_epoch = 0
    last_epoch = 0
    old_width = None
    filters_to_add = 0
    phase = "phase1_static"

    for epoch in range(1, total_epochs + 1):
        if epoch == phase1_epochs + 1 and phase2_epochs > 0:
            print("-" * 20 + f" Three-phase growth boundary (epoch {epoch}) " + "-" * 20)
            conv_channels = _conv_channels_from_model(model)
            model, conv_channels, old_width, filters_to_add = build_grown_model_at_conv_layer(
                model, conv_channels, growth_layer_idx, growth_gamma
            )
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            print(
                f"Added {filters_to_add} filters to conv layer {growth_layer_idx} "
                f"({old_width} -> {conv_channels[growth_layer_idx]})."
            )
        if epoch == phase1_epochs + phase2_epochs + 1 and phase3_epochs > 0:
            print("-" * 20 + f" Three-phase prune boundary (epoch {epoch}) " + "-" * 20)
            if old_width is None:
                raise RuntimeError("cannot enter phase 3 before phase 2 growth has occurred")
            model, conv_channels = build_three_phase_pruned_model(
                model, conv_channels, growth_layer_idx, old_width
            )
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            print(
                f"Removed the {filters_to_add} appended filters from conv layer "
                f"{growth_layer_idx}; conv channels restored to {conv_channels}."
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
        metrics["train_loss_total"].append(train_loss_total)
        metrics["train_loss_nll"].append(train_loss_nll)
        metrics["train_loss_kl"].append(train_loss_kl)
        metrics["train_loss_penalty"].append(train_loss_penalty)
        metrics["train_acc"].append(train_acc)
        metrics["train_brier"].append(train_brier)

        val_loss_total, val_acc, val_loss_nll, val_loss_kl, val_brier, val_loss_penalty = validate(
            model, val_loader, device, beta_scaled, lambda_penalty
        )
        metrics["val_loss_total"].append(val_loss_total)
        metrics["val_loss_nll"].append(val_loss_nll)
        metrics["val_loss_kl"].append(val_loss_kl)
        metrics["val_loss_penalty"].append(val_loss_penalty)
        metrics["val_acc"].append(val_acc)
        metrics["val_brier"].append(val_brier)
        metrics["param_count_history"].append(count_params(model))
        metrics["phase_history"].append(phase)
        metrics["conv_channel_history"].append(list(conv_channels))

        print(
            f"Epoch {epoch} ({phase}): Train Loss({loss_label})={train_loss_total:.4f}, "
            f"Train Acc={train_acc:.2f}%, Params={count_params(model):,}, "
            f"Val Acc={val_acc:.2f}%"
        )

        if phase == "phase3_pruned":
            val_score = _val_score_for_checkpoint(val_loss_total, val_loss_nll, checkpoint_metric)
            if _is_better_checkpoint_score(val_score, best_checkpoint_score, checkpoint_metric):
                best_checkpoint_score = val_score
                best_epoch = epoch
                best_conv_channels = list(conv_channels)
                best_model_state = copy.deepcopy(model.state_dict())
                save_checkpoint_cnn(
                    os.path.join(output_dir, "best_checkpoint.pth"),
                    state_dict=best_model_state,
                    epoch=best_epoch,
                    conv_channels=best_conv_channels,
                    selection_metric=checkpoint_metric_label,
                    selection_value=best_checkpoint_score,
                )

    plot_metrics(
        {experiment_name: metrics},
        save_path=os.path.join(output_dir, "metrics.png"),
    )

    if best_model_state is not None:
        eval_model = _make_eval_model(model, best_conv_channels)
        eval_model.load_state_dict(best_model_state, strict=True)
    else:
        eval_model = model

    test_loss_total, test_acc, test_loss_nll, test_loss_kl, test_brier, _ = validate(
        eval_model, test_loader, device, beta_scaled, lambda_penalty
    )
    print(f"Test Acc={test_acc:.2f}%, Test Loss={test_loss_total:.4f}, Test Brier={test_brier:.3f}")

    checkpoint = best_model_state if best_model_state is not None else model.state_dict()
    save_checkpoint_cnn(
        os.path.join(output_dir, "final_checkpoint.pth"),
        state_dict=checkpoint,
        epoch=best_epoch if best_model_state is not None else last_epoch,
        model=eval_model,
    )

    final_conv_channels = _conv_channels_from_model(eval_model)
    metrics.update({
        "test_acc": test_acc,
        "test_loss_total": test_loss_total,
        "test_loss_nll": test_loss_nll,
        "test_loss_kl": test_loss_kl,
        "test_brier": test_brier,
        "param_count": count_params(eval_model),
        "trainable_param_count": eval_model.get_param_stats()["trainable_params"],
        "conv_channels": final_conv_channels,
    })

    write_metrics_csv(output_dir, metrics)
    write_experiment_summary_csv_cnn(
        output_dir,
        model_label=experiment_name,
        params=metrics["param_count"],
        trainable_params=metrics["trainable_param_count"],
        conv_channels=metrics["conv_channels"],
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

    return metrics, model, total_epochs


def run_plasticity_then_three_phase_cnn(
    save_path,
    conv_channels,
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
    fc_hidden=128,
    in_channels=3,
    num_classes=10,
    decision_epsilon=0.0,
):
    conv_channels = list(conv_channels)
    suffix = _experiment_dir_suffix(junctures_mode)
    lam_tag = _format_lambda_dir(lambda_penalty)
    conv_tag = _conv_channels_tag(conv_channels)

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
            f"plasticity_{conv_tag}_{lam_tag}{suffix}",
        )
        base_model = BayesianCNN(
            in_channels, conv_channels, num_classes, fc_hidden=fc_hidden
        ).to(device)
        plasticity_metrics, _, _ = run_adaptive_experiment_cnn(
            "plasticity",
            base_model,
            conv_channels,
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
            decision_epsilon=decision_epsilon,
        )
        plasticity_ckpt_path = os.path.join(plasticity_output_dir, "best_checkpoint.pth")
        if not os.path.exists(plasticity_ckpt_path):
            raise FileNotFoundError(
                f"Plasticity best checkpoint not found: {plasticity_ckpt_path}"
            )

    hybrid_output_dir = _hybrid_output_dir_for_plasticity(
        plasticity_output_dir, save_path, conv_tag, lam_tag, suffix
    )

    refinement_model, refinement_conv_channels, ckpt_meta = load_checkpoint_cnn(
        plasticity_ckpt_path, device=device
    )
    growth_layer_idx = int(growth_layer_idx)
    growth_gamma = float(growth_gamma)
    growth_base_conv_channels = _conv_channels_from_model(refinement_model)
    growth_layer_width_at_growth = growth_base_conv_channels[growth_layer_idx]
    expected_filters_to_add = max(1, math.ceil(growth_gamma * growth_layer_width_at_growth))

    hybrid_metrics, _, _ = run_three_phase_baseline_cnn(
        "plasticity_three_phase",
        refinement_model,
        refinement_conv_channels,
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
        "plasticity_conv_channels_before_refinement": refinement_conv_channels,
        "plasticity_test_acc": plasticity_metrics.get("test_acc"),
        "plasticity_param_count": plasticity_metrics.get("param_count"),
        "growth_base_conv_channels": growth_base_conv_channels,
        "growth_layer_idx": growth_layer_idx,
        "growth_layer_width_at_growth": growth_layer_width_at_growth,
        "growth_gamma": growth_gamma,
        "expected_filters_to_add": expected_filters_to_add,
        "grow_epochs": int(grow_epochs),
        "prune_epochs": int(prune_epochs),
        "hybrid_test_acc": hybrid_metrics.get("test_acc"),
        "hybrid_param_count": hybrid_metrics.get("param_count"),
        "hybrid_conv_channels": hybrid_metrics.get("conv_channels"),
        "resumed_from_plasticity_dir": resume_from_plasticity_dir is not None,
    }
    provenance_path = write_hybrid_provenance(hybrid_output_dir, provenance)
    print(f"Hybrid provenance written to {provenance_path}")

    return plasticity_metrics, hybrid_metrics


def plot_param_count_vs_test_acc_cnn(
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
    Same API as main2.plot_param_count_vs_test_acc, for CNN result trees
    (run_*/baseline_10f_10f, run_*/plasticity_20f_20f_5e-06, ...).
    """
    return plot_param_count_vs_test_acc(
        save_path=save_path,
        experiments=experiments,
        experiment_glob=experiment_glob,
        num_runs=num_runs,
        aggregate_runs=aggregate_runs,
        x_col=x_col,
        y_col=y_col,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        save_path_out=save_path_out,
        show=show,
        figsize=figsize,
        dpi=dpi,
        alpha_individual=alpha_individual,
        marker_size=marker_size,
        annotate_points=annotate_points,
        group_by=group_by,
        style_map=style_map,
        lambda_palette=lambda_palette,
        style_by=style_by,
        show_pareto_frontier=show_pareto_frontier,
        pareto_scope=pareto_scope,
        pareto_y_goal=pareto_y_goal,
        pareto_frontier_kinds=pareto_frontier_kinds,
        pareto_linestyle=pareto_linestyle,
        pareto_linewidth=pareto_linewidth,
    )


def test_conv_surgery():
    print("Testing conv surgery...")
    model = BayesianCNN(3, [32, 64], 10, fc_hidden=128).to(device)
    grown_model, grown_channels, layer_idx, old_width = filtergenesis(
        model, [32, 64], gamma=0.1, growth_layer_score="mean"
    )
    expand_and_load_conv_stack(model.state_dict(), grown_model)
    assert grown_model.conv_layers[layer_idx].out_channels == grown_channels[layer_idx]
    if layer_idx + 1 < len(grown_model.conv_layers):
        assert (
            grown_model.conv_layers[layer_idx + 1].in_channels
            == grown_channels[layer_idx]
        )
    else:
        assert grown_model.fc.in_features == grown_channels[layer_idx]
        assert grown_model.classifier.in_features == grown_model.fc_hidden

    keep_dict, _ = filterapoptosis(
        grown_model,
        0.1,
        prune_mode="per_layer",
        min_filters_per_layer=2,
    )
    if keep_dict is not None:
        pruned_model, pruned_channels = build_pruned_model(grown_model, keep_dict)
        assert pruned_model.fc.in_features == pruned_channels[-1]
        assert pruned_model.classifier.in_features == pruned_model.fc_hidden
        x = torch.randn(2, 3, 32, 32, device=device)
        out = pruned_model(x)
        assert out.shape == (2, 10)
        print(f"Pruned channels: {pruned_channels}")
    print("Conv surgery test passed.")


def main(
    save_path,
    conv_channels,
    lambda_penalty,
    dataset="cifar10",
    junctures_mode="both",
    checkpoint_metric="val_loss_total",
    growth_layer_score="mad",
    growth_mad_percentile=90.0,
    decision_cooldown_epochs=0,
    phase1_epochs=100,
    phase2_epochs=100,
    phase3_epochs=100,
    three_phase_growth_layer_idx=1,
    three_phase_growth_gamma=1,
    resume_from_plasticity_dir=None,
    run_mode="plasticity",
    fc_hidden=128,
):
    allowed_run_modes = {
        "baseline",
        "plasticity",
        "three_phase",
        "plasticity_and_hybrid",
        "hybrid",
    }
    if run_mode not in allowed_run_modes:
        raise ValueError(
            f"run_mode must be one of {sorted(allowed_run_modes)}, got {run_mode!r}"
        )
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"dataset must be one of {sorted(DATASET_CONFIGS)}, got {dataset!r}"
        )
    in_channels = DATASET_CONFIGS[dataset]["in_channels"]
    num_classes = DATASET_CONFIGS[dataset]["num_classes"]

    num_epochs = 300
    batch_size = 128
    learning_rate = 0.005
    beta = 0.01
    decision_interval_min = 10
    decision_interval_max = 10
    decision_interval_power = 1
    decision_warmup_epochs = 0
    decision_cooldown_epochs = 0
    decision_epsilon = 0

    gamma = 0.1
    rho = 0.1
    if three_phase_growth_gamma is None:
        three_phase_growth_gamma = gamma
    prune_mode = "global_param"
    global_prune_budget = "filters"
    global_prune_normalize = "mad"
    warm_start_steps = 80
    warm_start_lr = 0.005

    os.makedirs(save_path, exist_ok=True)
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, batch_size=batch_size, seed=SEED
    )
    print(
        f"Dataset: {dataset} "
        f"(train={len(train_loader.dataset):,}, "
        f"val={len(val_loader.dataset):,}, "
        f"test={len(test_loader.dataset):,})"
    )

    conv_channels = list(conv_channels)
    conv_tag = _conv_channels_tag(conv_channels)

    if run_mode in ("baseline",):
        baseline_model = BayesianCNN(
            in_channels, conv_channels, num_classes, fc_hidden=fc_hidden
        ).to(device)
        baseline_output_dir = os.path.join(save_path, f"baseline_{conv_tag}")
        run_experiment_cnn(
            "baseline",
            baseline_model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            learning_rate,
            beta=beta,
            output_dir=baseline_output_dir,
        )

    if run_mode in ("three_phase",):
        three_phase_model = BayesianCNN(
            in_channels, conv_channels, num_classes, fc_hidden=fc_hidden
        ).to(device)
        three_phase_output_dir = os.path.join(save_path, f"three_phase_baseline_{conv_tag}")
        run_three_phase_baseline_cnn(
            "three_phase_baseline",
            three_phase_model,
            conv_channels,
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

    if run_mode in ("plasticity",):
        lam_tag = _format_lambda_dir(lambda_penalty)
        plasticity_output_dir = os.path.join(
            save_path, f"plasticity_{conv_tag}_{lam_tag}"
        )
        base_model = BayesianCNN(
            in_channels, conv_channels, num_classes, fc_hidden=fc_hidden
        ).to(device)
        run_adaptive_experiment_cnn(
            "plasticity",
            base_model,
            conv_channels,
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
            decision_epsilon=decision_epsilon,
        )

    if run_mode in ("plasticity_and_hybrid", "hybrid"):
        if run_mode == "hybrid" and resume_from_plasticity_dir is None:
            raise ValueError(
                "run_mode='hybrid' requires resume_from_plasticity_dir to be set"
            )
        run_plasticity_then_three_phase_cnn(
            save_path=save_path,
            conv_channels=conv_channels,
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
            fc_hidden=fc_hidden,
            in_channels=in_channels,
            num_classes=num_classes,
            decision_epsilon=decision_epsilon,
        )


if __name__ == "__main__":
    # Smoke / unit tests (fast)
    # test_conv_surgery()
    SEED = 42

    # for channels in [[150,150]]:
    #     set_seed(SEED)
    #     for i in range(1,6):
    #         print("Running experiment for run", i)
    #         main(save_path=f"./results_FashionMnist_CNN2_wstrain200_wsval100_0.1_ws0/run_{i}", conv_channels=channels, lambda_penalty=1e-8, run_mode="baseline", fc_hidden=128, dataset="fashion_mnist")

    # for channels in [[20,20],[50,50],[100,100],[150,150]]:
    #     for lambda_penalty in [5e-06,1e-06,5e-07,1e-07,5e-08]:
    #         set_seed(SEED)  
    #         for i in range(1,6):
    #             print("Running experiment for run", i)
    #             main(save_path=f"./results_FashionMnist_CNN2_wstrain200_wsval100_1/run_{i}", conv_channels=channels, lambda_penalty=lambda_penalty, run_mode="plasticity", fc_hidden=128, dataset="fashion_mnist")

    # for channels in [[300 ,300]]:
    #     for lambda_penalty in [1e-05]:
    #         set_seed(SEED)  
    #         for i in range(1,6):
    #             print("Running experiment for run", i)
    #             main(save_path=f"./results_FashionMnist_CNN2_wstrain200_wsval100_0.1_ws0/run_{i}", conv_channels=channels, lambda_penalty=lambda_penalty, run_mode="plasticity", fc_hidden=128, dataset="fashion_mnist")

    # for channels in [[200 ,200]]:
    #     for lambda_penalty in [1e-05]:
    #         set_seed(SEED)  
    #         for i in range(1,6):
    #             print("Running experiment for run", i)
    #             main(save_path=f"./results_FashionMnist_CNN2_wstrain200_wsval100_0.1_ws0/run_{i}", conv_channels=channels, lambda_penalty=lambda_penalty, run_mode="plasticity", fc_hidden=128, dataset="fashion_mnist")
    # plot_param_count_vs_test_acc_cnn(
    #     save_path="results_FashionMnist_CNN2_wstrain200_wsval100_0.1_ws0",
    #     experiments=[
    #         "baseline_10f_10f",
    #         "baseline_20f_20f",
    #         "baseline_50f_50f",
    #         "baseline_100f_100f",
    #         "baseline_150f_150f",
    #         "plasticity_200f_200f_1e-05",
    #         "plasticity_200f_200f_5e-06",
    #         "plasticity_200f_200f_5e-07",
    #         "plasticity_200f_200f_1e-07",
    #         "plasticity_200f_200f_5e-08",


    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_frontier_kinds=("baseline", "plasticity"),
    #     save_path_out="results_FashionMnist_CNN2_wstrain200_wsval100_0.1_ws0_test_brier.png",
    #     title="Fashion-MNIST CNN: Test Brier vs Parameter Count",
    #     y_col="Test Brier",
    #     ylabel="Test Brier",
    #     show=False,
    # )



