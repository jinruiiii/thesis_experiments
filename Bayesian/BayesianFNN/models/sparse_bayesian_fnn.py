"""Sparse Bayesian FNN layers and NeST growth/prune operators."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.train import loss_function
from models.bayesian_fnn import BayesianLinear

DEFAULT_RHO = -3.0

class MaskedBayesianLinear(BayesianLinear):
    """BayesianLinear with a binary connectivity mask on weights."""

    def __init__(self, in_features, out_features, bias_flag=True):
        super().__init__(in_features, out_features, bias_flag=bias_flag)
        self.register_buffer("mask_w", torch.ones(out_features, in_features))

    def _sigma(self, rho):
        return F.softplus(rho)

    def sample_weight(self, mean_field=False):
        if mean_field:
            w = self.mu_w
        else:
            sigma_w = self._sigma(self.rho_w)
            w = self.mu_w + sigma_w * torch.randn_like(sigma_w)
        return w * self.mask_w

    def sample_bias(self, mean_field=False):
        if not self.bias_flag:
            return None
        if mean_field:
            return self.mu_b
        sigma_b = self._sigma(self.rho_b)
        return self.mu_b + sigma_b * torch.randn_like(sigma_b)

    def forward(self, x, mean_field=False):
        w = self.sample_weight(mean_field=mean_field)
        b = self.sample_bias(mean_field=mean_field)
        return F.linear(x, w, b)

    def weight_snr(self, eps=1e-8):
        return torch.abs(self.mu_w) / (self._sigma(self.rho_w) + eps)

    def active_weight_count(self):
        return int(self.mask_w.sum().item())

    def active_param_count(self):
        n = self.active_weight_count()
        if self.bias_flag:
            n += self.out_features
        return n

    def kl_loss(self):
        """KL only over active weights (+ all biases)."""
        sigma_w = self._sigma(self.rho_w).clamp_min(1e-6)
        prior_sigma = self.prior_sigma_w.clamp_min(1e-6)
        posterior = torch.distributions.Normal(self.mu_w, sigma_w)
        prior = torch.distributions.Normal(self.prior_mu_w, prior_sigma)
        kl_w = torch.distributions.kl_divergence(posterior, prior)
        kl = (kl_w * self.mask_w).sum()

        if self.bias_flag:
            sigma_b = self._sigma(self.rho_b).clamp_min(1e-6)
            posterior_b = torch.distributions.Normal(self.mu_b, sigma_b)
            prior_b = torch.distributions.Normal(
                self.prior_mu_b, self.prior_sigma_b.clamp_min(1e-6)
            )
            kl = kl + torch.distributions.kl_divergence(posterior_b, prior_b).sum()
        return kl

    def avg_abs_active_mu(self):
        active = self.mask_w > 0.5
        if not active.any():
            return 0.1
        return float(self.mu_w[active].abs().mean().item())


class SparseBayesianFNN(nn.Module):
    """Two-hidden-layer sparse Bayesian FNN (NeST-style connectivity)."""

    def __init__(self, in_features, hidden_sizes, out_features):
        if len(hidden_sizes) != 2:
            raise ValueError(
                f"Bayesian NeST expects exactly 2 hidden layers, got {hidden_sizes}"
            )
        super().__init__()
        self.in_features = in_features
        self.hidden_sizes = list(hidden_sizes)
        self.out_features = out_features

        self.layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        prev = in_features
        for h in self.hidden_sizes:
            self.layers.append(MaskedBayesianLinear(prev, h))
            self.ln_layers.append(nn.LayerNorm(h))
            prev = h
        self.out = MaskedBayesianLinear(prev, out_features)

    def all_masked_layers(self):
        return list(self.layers) + [self.out]

    def forward(self, x, mean_field=False):
        for layer, ln in zip(self.layers, self.ln_layers):
            x = layer(x, mean_field=mean_field)
            x = ln(x)
            x = F.silu(x)
        return self.out(x, mean_field=mean_field)

    def kl_loss(self):
        return sum(layer.kl_loss() for layer in self.all_masked_layers())

    def active_param_count(self):
        """Active connections (weights + biases once each; not μ/ρ variational count)."""
        return sum(layer.active_param_count() for layer in self.all_masked_layers())

    def layernorm_param_count(self):
        return sum(p.numel() for ln in self.ln_layers for p in ln.parameters())

    def sparse_param_count(self):
        """Variational params on active connections: 2×(weights+biases) + LayerNorm."""
        return 2 * self.active_param_count() + self.layernorm_param_count()

    def dense_param_count(self):
        """All stored nn.Parameters (dense μ/ρ tensors + LayerNorm)."""
        return sum(p.numel() for p in self.parameters())

    def get_param_stats(self):
        active = self.active_param_count()
        sparse = self.sparse_param_count()
        dense = self.dense_param_count()
        return {
            "total_params": dense,
            "active_params": active,
            "sparse_params": sparse,
            "dense_params": dense,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen_params": 0,
        }

    def hidden_widths(self):
        return [layer.out_features for layer in self.layers]

    def sparsity(self):
        active = sum(layer.active_weight_count() for layer in self.all_masked_layers())
        total = sum(layer.mask_w.numel() for layer in self.all_masked_layers())
        return 1.0 - (active / max(total, 1))


# ---------------------------------------------------------------------------
# Seed connectivity
# ---------------------------------------------------------------------------


def _ensure_neuron_io(mask, min_in=1, min_out=1):
    """Ensure every output neuron has >=min_in edges and every input col >=min_out."""
    out_f, in_f = mask.shape
    for i in range(out_f):
        if mask[i].sum() < min_in:
            j = int(torch.randint(0, in_f, (1,)).item())
            mask[i, j] = 1.0
    for j in range(in_f):
        if mask[:, j].sum() < min_out:
            i = int(torch.randint(0, out_f, (1,)).item())
            mask[i, j] = 1.0
    return mask


def init_seed_masks(model: SparseBayesianFNN, activate_frac=0.1, generator=None):
    """
    Randomly activate activate_frac of weights per layer and repair connectivity
    so every hidden neuron has incoming and outgoing edges (gradient path).
    """
    g = generator or torch.Generator(device="cpu")
    with torch.no_grad():
        for layer in model.all_masked_layers():
            mask = (torch.rand(layer.mask_w.shape, generator=g) < activate_frac).float()
            # Guarantee at least one connection somewhere.
            if mask.sum() == 0:
                mask[0, 0] = 1.0
            layer.mask_w.copy_(mask.to(layer.mask_w.device))

        # Hidden layer 0: every neuron in + path from inputs.
        _ensure_neuron_io(model.layers[0].mask_w)
        # Hidden layer 1: every neuron in from h0 and out to out layer.
        _ensure_neuron_io(model.layers[1].mask_w)
        # Output: every class has >=1 in; every h1 neuron has >=1 out.
        _ensure_neuron_io(model.out.mask_w)

        # Extra: each h0 neuron must have outgoing into h1.
        h0 = model.layers[0].out_features
        for j in range(h0):
            if model.layers[1].mask_w[:, j].sum() < 1:
                i = int(torch.randint(0, model.layers[1].out_features, (1,), generator=g).item())
                model.layers[1].mask_w[i, j] = 1.0
        # Each h1 neuron must have outgoing into output.
        h1 = model.layers[1].out_features
        for j in range(h1):
            if model.out.mask_w[:, j].sum() < 1:
                i = int(torch.randint(0, model.out.out_features, (1,), generator=g).item())
                model.out.mask_w[i, j] = 1.0


def _collect_growth_signals(model, loader, device, beta_scaled, max_batches=8):
    """
    Mean-field forward + backward to collect:
      activations x before each masked linear, and ∂L/∂preact for each.
    Returns lists aligned with [h0, h1, out].
    """
    model.eval()
    layers = model.all_masked_layers()
    act_sums = [None] * len(layers)
    grad_sums = [None] * len(layers)
    n_seen = 0

    for b_idx, (inputs, labels) in enumerate(loader):
        if b_idx >= max_batches:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        acts = []
        preacts = []

        x = inputs
        for i, (layer, ln) in enumerate(zip(model.layers, model.ln_layers)):
            acts.append(x.detach())
            x = layer(x, mean_field=True)
            x.retain_grad()
            preacts.append(x)
            x = ln(x)
            x = F.silu(x)
        acts.append(x.detach())
        logits = model.out(x, mean_field=True)
        logits.retain_grad()
        preacts.append(logits)

        loss, _, _ = loss_function(logits, labels, model.kl_loss(), beta_scaled)
        model.zero_grad(set_to_none=True)
        loss.backward()

        for i, (act, pre) in enumerate(zip(acts, preacts)):
            g = pre.grad.detach()
            a = act.detach()
            # Average over batch for stable scores.
            a_mean = a.mean(dim=0)
            g_mean = g.mean(dim=0)
            if act_sums[i] is None:
                act_sums[i] = a_mean
                grad_sums[i] = g_mean
            else:
                act_sums[i] = act_sums[i] + a_mean
                grad_sums[i] = grad_sums[i] + g_mean
        n_seen += 1

    if n_seen == 0:
        raise RuntimeError("No batches available for growth signal collection")
    acts_out = [a / n_seen for a in act_sums]
    grads_out = [g / n_seen for g in grad_sums]
    return acts_out, grads_out


# ---------------------------------------------------------------------------
# Connection growth (Policy 1)
# ---------------------------------------------------------------------------


def connection_growth_step(
    model,
    loader,
    device,
    beta_scaled,
    conn_grow_frac=0.01,
    max_batches=8,
):
    """Activate top conn_grow_frac of dormant connections by |∂L/∂w| = |g_i x_j|."""
    acts, grads = _collect_growth_signals(
        model, loader, device, beta_scaled, max_batches=max_batches
    )
    layers = model.all_masked_layers()
    grown_total = 0
    per_layer = []

    with torch.no_grad():
        for layer, x, g in zip(layers, acts, grads):
            # Score matrix (out, in): |g_i| * |x_j| matches |g_i * x_j| magnitude.
            score = torch.outer(g.abs(), x.abs())
            dormant = layer.mask_w < 0.5
            if not dormant.any():
                per_layer.append(0)
                continue
            n_dormant = int(dormant.sum().item())
            k = max(1, int(math.ceil(conn_grow_frac * n_dormant)))
            score_dormant = score.clone()
            score_dormant[~dormant] = -1.0
            flat = score_dormant.view(-1)
            topk = torch.topk(flat, k=min(k, n_dormant), largest=True)
            idx = topk.indices
            grown = 0
            for flat_i in idx.tolist():
                if flat[flat_i] < 0:
                    continue
                oi = flat_i // layer.in_features
                ij = flat_i % layer.in_features
                if layer.mask_w[oi, ij] > 0.5:
                    continue
                layer.mask_w[oi, ij] = 1.0
                # Small He-like init for newly woken μ; keep rho at default.
                fan_in = max(int(layer.mask_w[oi].sum().item()), 1)
                std = math.sqrt(2.0 / fan_in)
                layer.mu_w[oi, ij] = torch.randn((), device=layer.mu_w.device) * std
                layer.rho_w[oi, ij] = DEFAULT_RHO
                grown += 1
            grown_total += grown
            per_layer.append(grown)

    return {"grown_connections": grown_total, "per_layer": per_layer}


# ---------------------------------------------------------------------------
# Neuron growth (Policy 2 + square-root init)
# ---------------------------------------------------------------------------


def _expand_masked_linear_out(layer: MaskedBayesianLinear, new_out: int):
    """Expand out_features by appending rows (new neurons)."""
    old_out, in_f = layer.out_features, layer.in_features
    if new_out < old_out:
        raise ValueError("new_out must be >= old_out")
    if new_out == old_out:
        return layer
    add = new_out - old_out
    device = layer.mu_w.device

    def _cat_rows(param, fill):
        extra = fill((add, in_f), device)
        return nn.Parameter(torch.cat([param.data, extra], dim=0))

    def _cat_rows_buf(buf, fill):
        extra = fill((add, in_f), device)
        return torch.cat([buf, extra], dim=0)

    layer.mu_w = _cat_rows(layer.mu_w, lambda shape, dev: torch.zeros(shape, device=dev))
    layer.rho_w = _cat_rows(
        layer.rho_w, lambda shape, dev: torch.full(shape, DEFAULT_RHO, device=dev)
    )
    layer.mask_w = _cat_rows_buf(
        layer.mask_w, lambda shape, dev: torch.zeros(shape, device=dev)
    )
    layer.prior_mu_w = _cat_rows_buf(
        layer.prior_mu_w, lambda shape, dev: torch.zeros(shape, device=dev)
    )
    layer.prior_sigma_w = _cat_rows_buf(
        layer.prior_sigma_w, lambda shape, dev: torch.ones(shape, device=dev)
    )
    if layer.bias_flag:
        layer.mu_b = nn.Parameter(
            torch.cat([layer.mu_b.data, torch.zeros(add, device=device)], dim=0)
        )
        layer.rho_b = nn.Parameter(
            torch.cat(
                [layer.rho_b.data, torch.full((add,), DEFAULT_RHO, device=device)],
                dim=0,
            )
        )
        layer.prior_mu_b = torch.cat(
            [layer.prior_mu_b, torch.zeros(add, device=device)], dim=0
        )
        layer.prior_sigma_b = torch.cat(
            [layer.prior_sigma_b, torch.ones(add, device=device)], dim=0
        )
    layer.out_features = new_out
    return layer


def _expand_masked_linear_in(layer: MaskedBayesianLinear, new_in: int):
    """Expand in_features by appending columns (inputs from new prev neurons)."""
    out_f, old_in = layer.out_features, layer.in_features
    if new_in < old_in:
        raise ValueError("new_in must be >= old_in")
    if new_in == old_in:
        return layer
    add = new_in - old_in
    device = layer.mu_w.device

    def _cat_cols(param, fill):
        extra = fill((out_f, add), device)
        return nn.Parameter(torch.cat([param.data, extra], dim=1))

    def _cat_cols_buf(buf, fill):
        extra = fill((out_f, add), device)
        return torch.cat([buf, extra], dim=1)

    layer.mu_w = _cat_cols(layer.mu_w, lambda shape, dev: torch.zeros(shape, device=dev))
    layer.rho_w = _cat_cols(
        layer.rho_w, lambda shape, dev: torch.full(shape, DEFAULT_RHO, device=dev)
    )
    layer.mask_w = _cat_cols_buf(
        layer.mask_w, lambda shape, dev: torch.zeros(shape, device=dev)
    )
    layer.prior_mu_w = _cat_cols_buf(
        layer.prior_mu_w, lambda shape, dev: torch.zeros(shape, device=dev)
    )
    layer.prior_sigma_w = _cat_cols_buf(
        layer.prior_sigma_w, lambda shape, dev: torch.ones(shape, device=dev)
    )
    layer.in_features = new_in
    return layer


def _expand_layernorm(ln: nn.LayerNorm, new_dim: int):
    old = ln.normalized_shape[0]
    if new_dim == old:
        return ln
    if new_dim < old:
        raise ValueError("cannot shrink LayerNorm")
    add = new_dim - old
    device = ln.weight.device
    new_ln = nn.LayerNorm(new_dim).to(device)
    with torch.no_grad():
        new_ln.weight[:old] = ln.weight.data
        new_ln.bias[:old] = ln.bias.data
        new_ln.weight[old:] = 1.0
        new_ln.bias[old:] = 0.0
    return new_ln


def neuron_growth_step(
    model: SparseBayesianFNN,
    loader,
    device,
    beta_scaled,
    hidden_layer_idx,
    beta_growth=0.4,
    birth_strength=0.4,
    max_batches=8,
):
    """
    Add one neuron in hidden layer `hidden_layer_idx` (0 or 1) via NeST Algorithm 1
    with square-root weight init and birth-strength rescale.
    """
    if hidden_layer_idx not in (0, 1):
        raise ValueError("hidden_layer_idx must be 0 or 1")

    acts, grads = _collect_growth_signals(
        model, loader, device, beta_scaled, max_batches=max_batches
    )
    # Bridging: x from layer l-1 (acts[hidden_layer_idx]), g from layer l+1
    # grads[hidden_layer_idx + 1] is ∂L/∂preact of the *next* linear layer.
    x_prev = acts[hidden_layer_idx]  # (N,)
    g_next = grads[hidden_layer_idx + 1]  # (M,)
    # G[m, n] = g_m * x_n
    G = torch.outer(g_next, x_prev)  # (M, N)

    M, N = G.shape
    n_select = max(1, int(math.ceil(beta_growth * M * N)))
    abs_G = G.abs().reshape(-1)
    topk = torch.topk(abs_G, k=min(n_select, abs_G.numel()), largest=True)
    thresh = topk.values[-1].item()

    w_out = torch.zeros(M, device=device)
    w_in = torch.zeros(N, device=device)
    selected = 0
    with torch.no_grad():
        for m in range(M):
            for n in range(N):
                gmn = G[m, n].item()
                if abs(gmn) >= thresh - 1e-12:
                    # Square-root rule (eq. 3): |δw_in| = |δw_out| = sqrt(|G|)
                    delta = math.sqrt(max(abs(gmn), 0.0))
                    sign = 1.0 if torch.rand(()) < 0.5 else -1.0
                    w_out[m] = w_out[m] + sign * delta
                    w_in[n] = w_in[n] + sign * delta * (1.0 if gmn >= 0 else -1.0)
                    selected += 1

        in_layer = model.layers[hidden_layer_idx]
        if hidden_layer_idx == 0:
            out_layer = model.layers[1]
        else:
            out_layer = model.out

        avg_in = in_layer.avg_abs_active_mu()
        avg_out = out_layer.avg_abs_active_mu()
        avg_w_in = float(w_in.abs().mean().item()) if w_in.abs().sum() > 0 else 1.0
        avg_w_out = float(w_out.abs().mean().item()) if w_out.abs().sum() > 0 else 1.0
        # Birth strength rescale (eq. 7)
        w_in = w_in * birth_strength * (avg_in / max(avg_w_in, 1e-8))
        w_out = w_out * birth_strength * (avg_out / max(avg_w_out, 1e-8))

        # Expand architecture: +1 neuron in hidden layer.
        old_h = in_layer.out_features
        new_h = old_h + 1
        _expand_masked_linear_out(in_layer, new_h)
        model.ln_layers[hidden_layer_idx] = _expand_layernorm(
            model.ln_layers[hidden_layer_idx], new_h
        ).to(device)
        _expand_masked_linear_in(out_layer, new_h)

        # Place new neuron connections where w_in / w_out nonzero.
        new_row = old_h  # index of new neuron in in_layer
        new_col = old_h  # index of new neuron as input to out_layer
        for n in range(N):
            if w_in[n].abs() > 0:
                in_layer.mask_w[new_row, n] = 1.0
                in_layer.mu_w[new_row, n] = w_in[n]
                in_layer.rho_w[new_row, n] = DEFAULT_RHO
        for m in range(M):
            if w_out[m].abs() > 0:
                out_layer.mask_w[m, new_col] = 1.0
                out_layer.mu_w[m, new_col] = w_out[m]
                out_layer.rho_w[m, new_col] = DEFAULT_RHO
        # Ensure the new neuron is not isolated.
        if in_layer.mask_w[new_row].sum() < 1:
            j = int(torch.argmax(x_prev.abs()).item())
            in_layer.mask_w[new_row, j] = 1.0
            in_layer.mu_w[new_row, j] = 0.01
            in_layer.rho_w[new_row, j] = DEFAULT_RHO
        if out_layer.mask_w[:, new_col].sum() < 1:
            i = int(torch.argmax(g_next.abs()).item())
            out_layer.mask_w[i, new_col] = 1.0
            out_layer.mu_w[i, new_col] = 0.01
            out_layer.rho_w[i, new_col] = DEFAULT_RHO

        model.hidden_sizes[hidden_layer_idx] = new_h

    return {
        "grown_neuron_layer": hidden_layer_idx,
        "new_width": model.hidden_sizes[hidden_layer_idx],
        "selected_pairs": selected,
        "active_in": int(in_layer.mask_w[new_row].sum().item()),
        "active_out": int(out_layer.mask_w[:, new_col].sum().item()),
    }


# ---------------------------------------------------------------------------
# SNR pruning (Policy 4 with SNR)
# ---------------------------------------------------------------------------


def snr_prune_step(model: SparseBayesianFNN, prune_frac=0.01):
    """Prune bottom prune_frac of active weights per layer by SNR."""
    pruned_total = 0
    per_layer = []
    with torch.no_grad():
        for layer in model.all_masked_layers():
            active = layer.mask_w > 0.5
            n_active = int(active.sum().item())
            if n_active == 0:
                per_layer.append(0)
                continue
            k = max(1, int(math.floor(prune_frac * n_active)))
            k = min(k, n_active)
            snr = layer.weight_snr()
            snr_masked = snr.clone()
            snr_masked[~active] = float("inf")
            flat = snr_masked.view(-1)
            bottom = torch.topk(flat, k=k, largest=False)
            pruned = 0
            for flat_i in bottom.indices.tolist():
                oi = flat_i // layer.in_features
                ij = flat_i % layer.in_features
                if layer.mask_w[oi, ij] < 0.5:
                    continue
                layer.mask_w[oi, ij] = 0.0
                pruned += 1
            pruned_total += pruned
            per_layer.append(pruned)
    return {"pruned_connections": pruned_total, "per_layer": per_layer}


def cleanup_dead_neurons(model: SparseBayesianFNN):
    """
    Compact by removing hidden neurons with zero fan-in or zero fan-out.
    Returns number of neurons removed.
    """
    removed = 0
    # Process from last hidden to first so indices stay consistent within a pass;
    # we rebuild keep indices per layer.
    for h_idx in (1, 0):
        in_layer = model.layers[h_idx]
        out_layer = model.layers[1] if h_idx == 0 else model.out
        width = in_layer.out_features
        keep = []
        for j in range(width):
            fan_in = int(in_layer.mask_w[j].sum().item())
            fan_out = int(out_layer.mask_w[:, j].sum().item())
            if fan_in > 0 and fan_out > 0:
                keep.append(j)
            else:
                removed += 1
        if len(keep) == width:
            continue
        if len(keep) == 0:
            # Keep a single dummy neuron to preserve architecture.
            keep = [0]
            removed -= 1
            in_layer.mask_w[0, 0] = 1.0
            out_layer.mask_w[0, 0] = 1.0

        keep_t = torch.tensor(keep, device=in_layer.mu_w.device, dtype=torch.long)
        # Shrink in_layer rows.
        in_layer.mu_w = nn.Parameter(in_layer.mu_w.data[keep_t].clone())
        in_layer.rho_w = nn.Parameter(in_layer.rho_w.data[keep_t].clone())
        in_layer.mask_w = in_layer.mask_w[keep_t].clone()
        in_layer.prior_mu_w = in_layer.prior_mu_w[keep_t].clone()
        in_layer.prior_sigma_w = in_layer.prior_sigma_w[keep_t].clone()
        if in_layer.bias_flag:
            in_layer.mu_b = nn.Parameter(in_layer.mu_b.data[keep_t].clone())
            in_layer.rho_b = nn.Parameter(in_layer.rho_b.data[keep_t].clone())
            in_layer.prior_mu_b = in_layer.prior_mu_b[keep_t].clone()
            in_layer.prior_sigma_b = in_layer.prior_sigma_b[keep_t].clone()
        in_layer.out_features = len(keep)
        old_ln = model.ln_layers[h_idx]
        new_ln = nn.LayerNorm(len(keep)).to(in_layer.mu_w.device)
        with torch.no_grad():
            new_ln.weight.copy_(old_ln.weight.data[keep_t])
            new_ln.bias.copy_(old_ln.bias.data[keep_t])
        model.ln_layers[h_idx] = new_ln

        # Shrink out_layer columns.
        out_layer.mu_w = nn.Parameter(out_layer.mu_w.data[:, keep_t].clone())
        out_layer.rho_w = nn.Parameter(out_layer.rho_w.data[:, keep_t].clone())
        out_layer.mask_w = out_layer.mask_w[:, keep_t].clone()
        out_layer.prior_mu_w = out_layer.prior_mu_w[:, keep_t].clone()
        out_layer.prior_sigma_w = out_layer.prior_sigma_w[:, keep_t].clone()
        out_layer.in_features = len(keep)
        model.hidden_sizes[h_idx] = len(keep)

    return {"neurons_removed": removed, "hidden_sizes": list(model.hidden_sizes)}
