import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.train import (
    count_params,
    penalised_elbo_on_batches,
    penalised_loss_function,
    sample_batches,
)
from models.bayesian_fnn import BayesianFNN

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
    junctures_mode: "both" (grow or prune), "grow" (grow only), "prune" (prune only),
    or "both_gp" (grow, prune, or grow+prune combined).
    grow_new_only_steps: for grow warm-start, apply new-only mask for this many initial
    steps (default None = all K steps); remaining steps update all parameters.
    Returns (action, model, hidden_sizes, info_dict).
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
        'delta_grow_prune': None,
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

    if junctures_mode in ("both", "grow", "both_gp"):
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

    if junctures_mode in ("both", "prune", "both_gp"):
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

    # Combined grow+prune candidate (both_gp only): grow then prune, full warm-start
    # (no new-neuron isolation), same warm-start style as the prune branch.
    delta_grow_prune = float('-inf')
    grow_prune_model = None
    hidden_sizes_gp = None
    if junctures_mode == "both_gp":
        gp_grow_model, _, layer_idx_gp, _ = neurogenesis(
            model,
            hidden_sizes,
            exclude=list(grow_exclude_layers),
            gamma=gamma,
            uncertainty_combine=uncertainty_combine,
            growth_layer_score=growth_layer_score,
            growth_mad_percentile=growth_mad_percentile,
        )
        expand_and_load_encoder_layer(model.state_dict(), gp_grow_model)
        keep_dict_gp, prune_stats_gp = neuroapoptosis(
            gp_grow_model,
            rho,
            snr_combine=snr_combine,
            prune_mode=prune_mode,
            min_neurons_per_layer=2,
            global_prune_budget=global_prune_budget,
            global_prune_normalize=global_prune_normalize,
        )
        if prune_stats_gp:
            info['grow_prune_target_remove'] = prune_stats_gp.get('target_remove')
            info['grow_prune_params_removed'] = prune_stats_gp.get('params_removed')
            info['grow_prune_neurons_pruned'] = prune_stats_gp.get('neurons_pruned')
        if keep_dict_gp is not None:
            grow_prune_model, hidden_sizes_gp = build_pruned_model(
                gp_grow_model, keep_dict_gp
            )
            warm_start_model_on_batches(
                grow_prune_model,
                batches_ws,
                K,
                eta_ws,
                beta_scaled,
                lambda_penalty,
            )
            L_after_grow_prune = penalised_elbo_on_batches(
                grow_prune_model, batches_val, beta_scaled, lambda_penalty
            )
            delta_grow_prune = L_after_grow_prune - L_after_none
            info['delta_grow_prune'] = delta_grow_prune
            if layer_idx is None:
                layer_idx = layer_idx_gp
        else:
            info['delta_grow_prune'] = None

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
    elif junctures_mode == "both_gp":
        # Max delta > 0; ties prefer grow, then grow_prune, then prune.
        candidates = []
        if delta_grow > 0:
            candidates.append((delta_grow, 0, 'grow', grow_model, hidden_sizes_g))
        if delta_grow_prune > 0 and grow_prune_model is not None:
            candidates.append(
                (delta_grow_prune, 1, 'grow_prune', grow_prune_model, hidden_sizes_gp)
            )
        if delta_prune_val > 0:
            candidates.append(
                (delta_prune_val, 2, 'prune', prune_model, hidden_sizes_p)
            )
        if candidates:
            candidates.sort(key=lambda c: (-c[0], c[1]))
            _, _, best_action, best_model, best_hidden_sizes = candidates[0]
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
    delta_gp_log = ""
    if junctures_mode == "both_gp":
        delta_gp_str = (
            f"{info['delta_grow_prune']:.4f}"
            if info['delta_grow_prune'] is not None
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

