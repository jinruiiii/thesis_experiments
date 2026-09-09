import copy
import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from lib.train import (
    count_params,
    penalised_elbo_on_batches,
    penalised_loss_function,
    sample_batches,
)
from models.bayesian_cnn import BayesianCNN


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

        for p in ["prior_mu_w", "prior_sigma_w", "prior_mu_b", "prior_sigma_b"]:
            key = f"conv_layers.{i}.{p}"
            if key not in old_sd:
                continue
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
    for p in ["mu_w", "rho_w", "mu_b", "rho_b", "prior_mu_w", "prior_sigma_w", "prior_mu_b", "prior_sigma_b"]:
        key = f"fc.{p}"
        if key not in old_sd:
            if p.startswith("prior_"):
                continue
            raise ValueError(f"{key} is missing in the plasticity model")
        w = old_sd[key]
        if w.ndim == 2 and keep_last is not None:
            w = w[:, keep_last]
        new_sd[key] = w

    for p in ["mu_w", "rho_w", "mu_b", "rho_b", "prior_mu_w", "prior_sigma_w", "prior_mu_b", "prior_sigma_b"]:
        key = f"classifier.{p}"
        if key not in old_sd:
            if p.startswith("prior_"):
                continue
            raise ValueError(f"{key} is missing in the plasticity model")
        new_sd[key] = old_sd[key]

    new_model.load_state_dict(new_sd, strict=True)


def build_pruned_model(model, keep_dict):
    conv_channels = [len(keep_dict[i]) for i in sorted(keep_dict.keys())]
    pruned_model = _bayesian_cnn_like(model, conv_channels=conv_channels)
    truncate_and_load_conv_stack(model.state_dict(), keep_dict, pruned_model)
    return pruned_model, conv_channels


def _mask_warm_start_grads_grow_new_only(model, layer_idx, old_width):
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


def warm_start_model_on_batches(
    model,
    batches_ws,
    K,
    eta_ws,
    beta_scaled,
    lambda_penalty,
    mask_mode=None,
    layer_idx=None,
    old_width=None,
    grow_new_only_steps=None,
):
    """
    K gradient steps cycling through warm-start batches.

    When mask_mode == "grow_new_only", the new-filter gradient mask is applied only for
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
        if mask_mode == "grow_new_only" and step < new_only_limit:
            _mask_warm_start_grads_grow_new_only(
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
    random_growth=False,
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

    if random_growth:
        layer_to_expand = random.choice(eligible)
        filters_to_add = max(1, math.ceil(gamma * conv_channels[layer_to_expand]))
        old_width = conv_channels[layer_to_expand]
        print(
            f"\n[filtergenesis] random_growth=True; eligible layers={eligible}"
        )
        print(
            f"Expanding Conv Layer {layer_to_expand+1} "
            f"(random choice among {len(eligible)} eligible) "
            f"by {filters_to_add} filters"
        )
    else:
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


def structural_decision_juncture(
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
    grow_new_only_steps=None,
    random_growth=False,
):
    """
    Evaluate growth and/or prune candidates via delta penalised ELBO on B_val.
    junctures_mode: "both", "grow", "prune", or "both_gp" (grow, prune, or grow+prune).
    grow_new_only_steps: first N grow warm-start steps use new-only mask (default all K).
    random_growth: if True, pick grow layer uniformly among eligible instead of uncertainty/MAD.
    """
    if junctures_mode not in ("both", "grow", "prune", "both_gp"):
        raise ValueError(
            f"junctures_mode must be 'both', 'grow', 'prune', or 'both_gp', "
            f"got {junctures_mode!r}"
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

    M_ws = 40
    M_val = 10
    batches_ws = sample_batches(train_loader, device, M_ws)
    batches_val = sample_batches(val_loader, device, M_val)

    L_before = penalised_elbo_on_batches(model, batches_val, beta_scaled, lambda_penalty)

    none_model = copy.deepcopy(model)
    warm_start_model_on_batches(
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
        "delta_grow_prune": None,
        "param_count_before": count_params(model),
        "prune_mode": prune_mode,
        "global_prune_budget": global_prune_budget,
        "global_prune_normalize": global_prune_normalize,
        "prune_target_remove": None,
        "prune_params_removed": None,
        "prune_neurons_pruned": None,
        "growth_layer_score": growth_layer_score,
        "growth_mad_percentile": growth_mad_percentile,
        "grow_new_only_steps": resolved_grow_new_only_steps,
        "random_growth": bool(random_growth),
    }
    if grow_exclude_layers is None:
        grow_exclude_layers = []

    grow_model = None
    conv_channels_g = None
    layer_idx = None
    delta_grow = float("-inf")

    if junctures_mode in ("both", "grow", "both_gp"):
        grow_model, conv_channels_g, layer_idx, old_width = filtergenesis(
            model,
            conv_channels,
            exclude=list(grow_exclude_layers),
            gamma=gamma,
            uncertainty_combine=uncertainty_combine,
            growth_layer_score=growth_layer_score,
            growth_mad_percentile=growth_mad_percentile,
            random_growth=random_growth,
        )
        expand_and_load_conv_stack(model.state_dict(), grow_model)
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
        info["delta_grow"] = delta_grow

    delta_prune = float("-inf")
    prune_model = None
    conv_channels_p = None
    keep_dict = None

    if junctures_mode in ("both", "prune", "both_gp"):
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
            info["prune_snr_normalize"] = prune_stats.get(
                "snr_normalize", global_prune_normalize
            )
        if keep_dict is not None:
            prune_model, conv_channels_p = build_pruned_model(model, keep_dict)
            warm_start_model_on_batches(
                prune_model, batches_ws, K, eta_ws, beta_scaled, lambda_penalty
            )
            L_after_prune = penalised_elbo_on_batches(
                prune_model, batches_val, beta_scaled, lambda_penalty
            )
            delta_prune = L_after_prune - L_after_none
        info["delta_prune"] = delta_prune if keep_dict is not None else None

    delta_grow_prune = float("-inf")
    grow_prune_model = None
    conv_channels_gp = None
    if junctures_mode == "both_gp":
        gp_grow_model, _, layer_idx_gp, _ = filtergenesis(
            model,
            conv_channels,
            exclude=list(grow_exclude_layers),
            gamma=gamma,
            uncertainty_combine=uncertainty_combine,
            growth_layer_score=growth_layer_score,
            growth_mad_percentile=growth_mad_percentile,
            random_growth=random_growth,
        )
        expand_and_load_conv_stack(model.state_dict(), gp_grow_model)
        keep_dict_gp, prune_stats_gp = filterapoptosis(
            gp_grow_model,
            rho,
            snr_combine=snr_combine,
            prune_mode=prune_mode,
            min_filters_per_layer=2,
            global_prune_budget=global_prune_budget,
            global_prune_normalize=global_prune_normalize,
        )
        if prune_stats_gp:
            info["grow_prune_target_remove"] = prune_stats_gp.get("target_remove")
            info["grow_prune_params_removed"] = prune_stats_gp.get("params_removed")
            info["grow_prune_neurons_pruned"] = prune_stats_gp.get("filters_pruned")
        if keep_dict_gp is not None:
            grow_prune_model, conv_channels_gp = build_pruned_model(
                gp_grow_model, keep_dict_gp
            )
            warm_start_model_on_batches(
                grow_prune_model, batches_ws, K, eta_ws, beta_scaled, lambda_penalty
            )
            L_after_grow_prune = penalised_elbo_on_batches(
                grow_prune_model, batches_val, beta_scaled, lambda_penalty
            )
            delta_grow_prune = L_after_grow_prune - L_after_none
            info["delta_grow_prune"] = delta_grow_prune
            if layer_idx is None:
                layer_idx = layer_idx_gp
        else:
            info["delta_grow_prune"] = None

    best_action = "none"
    best_model = model
    best_conv_channels = conv_channels

    delta_prune_val = delta_prune if keep_dict is not None else float("-inf")
    if junctures_mode == "both":
        if delta_grow > 0 or delta_prune_val > 0:
            if delta_grow >= delta_prune_val:
                best_action = "grow"
                best_model = grow_model
                best_conv_channels = conv_channels_g
            else:
                best_action = "prune"
                best_model = prune_model
                best_conv_channels = conv_channels_p
    elif junctures_mode == "both_gp":
        candidates = []
        if delta_grow > 0:
            candidates.append((delta_grow, 0, "grow", grow_model, conv_channels_g))
        if delta_grow_prune > 0 and grow_prune_model is not None:
            candidates.append(
                (delta_grow_prune, 1, "grow_prune", grow_prune_model, conv_channels_gp)
            )
        if delta_prune_val > 0:
            candidates.append(
                (delta_prune_val, 2, "prune", prune_model, conv_channels_p)
            )
        if candidates:
            candidates.sort(key=lambda c: (-c[0], c[1]))
            _, _, best_action, best_model, best_conv_channels = candidates[0]
    elif junctures_mode == "grow" and delta_grow > 0:
        best_action = "grow"
        best_model = grow_model
        best_conv_channels = conv_channels_g
    elif junctures_mode == "prune" and delta_prune_val > 0:
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
    delta_gp_log = ""
    if junctures_mode == "both_gp":
        delta_gp_str = (
            f"{info['delta_grow_prune']:.4f}"
            if info.get("delta_grow_prune") is not None
            else "N/A"
        )
        delta_gp_log = f", delta_grow_prune={delta_gp_str}"
    print(
        f"\nStructural decision (mode={junctures_mode}, prune_mode={prune_mode}, "
        f"global_prune_budget={global_prune_budget}, "
        f"global_prune_normalize={global_prune_normalize}, "
        f"grow_new_only_steps={resolved_grow_new_only_steps}/{K}): "
        f"L_before={L_before:.4f}, L_after_none={L_after_none:.4f}, "
        f"delta_grow={delta_grow_str}, delta_prune={info['delta_prune']}"
        f"{delta_gp_log}, "
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

