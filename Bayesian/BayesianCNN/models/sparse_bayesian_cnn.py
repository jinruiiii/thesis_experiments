"""Sparse Bayesian CNN layers and NeST growth/prune operators (Policies 1, 3, 4.3.1)."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bayesian_linear import BayesianLinear

DEFAULT_RHO = -3.0


class MaskedBayesianLinear(BayesianLinear):
    """BayesianLinear with a binary connectivity mask on weights."""

    def __init__(self, in_features, out_features, bias_flag=True):
        super().__init__(in_features, out_features, bias_flag=bias_flag)
        self.register_buffer("mask_w", torch.ones(out_features, in_features))

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

    def effective_weight_magnitude(self):
        return self.mu_w.abs()

    def active_weight_count(self):
        return int(self.mask_w.sum().item())

    def active_param_count(self):
        n = self.active_weight_count()
        if self.bias_flag:
            n += self.out_features
        return n

    def kl_loss(self):
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


class MaskedBayesianConv2d(nn.Module):
    """Bayesian conv layer with a binary mask on kernel weights."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias_flag=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.bias_flag = bias_flag

        kH, kW = self.kernel_size
        self.mu_w = nn.Parameter(torch.randn(out_channels, in_channels, kH, kW) * 0.1)
        self.rho_w = nn.Parameter(torch.ones(out_channels, in_channels, kH, kW) * DEFAULT_RHO)
        self.register_buffer("mask_w", torch.ones(out_channels, in_channels, kH, kW))
        self.register_buffer("prior_mu_w", torch.zeros(out_channels, in_channels, kH, kW))
        self.register_buffer("prior_sigma_w", torch.ones(out_channels, in_channels, kH, kW))

        if self.bias_flag:
            self.mu_b = nn.Parameter(torch.randn(out_channels) * 0.1)
            self.rho_b = nn.Parameter(torch.ones(out_channels) * DEFAULT_RHO)
            self.register_buffer("prior_mu_b", torch.zeros(out_channels))
            self.register_buffer("prior_sigma_b", torch.ones(out_channels))

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
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)

    def effective_weight_magnitude(self, gn_layer, eps=1e-8):
        scale = gn_layer.weight.abs().clamp_min(eps).view(-1, 1, 1, 1)
        return self.mu_w.abs() / scale

    def active_weight_count(self):
        return int(self.mask_w.sum().item())

    def active_param_count(self):
        n = self.active_weight_count()
        if self.bias_flag:
            n += self.out_channels
        return n

    def kl_loss(self):
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


class SparseBayesianCNN(nn.Module):
    """Two-conv-layer sparse Bayesian CNN (NeST-style connectivity)."""

    def __init__(
        self,
        in_channels,
        conv_channels,
        num_classes,
        kernel_size=3,
        padding=1,
        stride=1,
        fc_hidden=128,
    ):
        if len(conv_channels) != 2:
            raise ValueError(
                f"Bayesian NeST CNN expects exactly 2 conv layers, got {conv_channels}"
            )
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = list(conv_channels)
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.fc_hidden = int(fc_hidden)

        self.conv_layers = nn.ModuleList()
        self.gn_layers = nn.ModuleList()
        prev_ch = in_channels
        for out_ch in self.conv_channels:
            self.conv_layers.append(
                MaskedBayesianConv2d(
                    prev_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.gn_layers.append(nn.GroupNorm(1, out_ch))
            prev_ch = out_ch

        self.fc = MaskedBayesianLinear(prev_ch, self.fc_hidden)
        self.classifier = MaskedBayesianLinear(self.fc_hidden, num_classes)

    def conv_masked_layers(self):
        return list(self.conv_layers)

    def linear_masked_layers(self):
        return [self.fc, self.classifier]

    def all_masked_layers(self):
        return self.conv_masked_layers() + self.linear_masked_layers()

    def forward(self, x, mean_field=False):
        for conv, gn in zip(self.conv_layers, self.gn_layers):
            x = conv(x, mean_field=mean_field)
            x = gn(x)
            x = F.silu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x, mean_field=mean_field)
        x = F.silu(x)
        return self.classifier(x, mean_field=mean_field)

    def kl_loss(self):
        kl = sum(layer.kl_loss() for layer in self.conv_layers)
        kl += self.fc.kl_loss()
        kl += self.classifier.kl_loss()
        return kl

    def gn_param_count(self):
        return sum(p.numel() for gn in self.gn_layers for p in gn.parameters())

    def active_param_count(self):
        return sum(layer.active_param_count() for layer in self.all_masked_layers())

    def sparse_param_count(self):
        return 2 * self.active_param_count() + self.gn_param_count()

    def dense_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def conv_widths(self):
        return list(self.conv_channels)

    def sparsity(self):
        active = sum(layer.active_weight_count() for layer in self.all_masked_layers())
        total = sum(layer.mask_w.numel() for layer in self.all_masked_layers())
        return 1.0 - (active / max(total, 1))


def _ensure_filter_io(mask, min_in=1, min_out=1):
    if mask.ndim == 4:
        out_f, in_f, _, _ = mask.shape
        for o in range(out_f):
            if mask[o].sum() < min_in:
                i = int(torch.randint(0, in_f, (1,)).item())
                kh = int(torch.randint(0, mask.shape[2], (1,)).item())
                kw = int(torch.randint(0, mask.shape[3], (1,)).item())
                mask[o, i, kh, kw] = 1.0
        for i in range(in_f):
            if mask[:, i, :, :].sum() < min_out:
                o = int(torch.randint(0, out_f, (1,)).item())
                kh = int(torch.randint(0, mask.shape[2], (1,)).item())
                kw = int(torch.randint(0, mask.shape[3], (1,)).item())
                mask[o, i, kh, kw] = 1.0
    elif mask.ndim == 2:
        out_f, in_f = mask.shape
        for o in range(out_f):
            if mask[o].sum() < min_in:
                j = int(torch.randint(0, in_f, (1,)).item())
                mask[o, j] = 1.0
        for j in range(in_f):
            if mask[:, j].sum() < min_out:
                o = int(torch.randint(0, out_f, (1,)).item())
                mask[o, j] = 1.0
    else:
        raise ValueError(f"Expected 2D or 4D mask, got shape {tuple(mask.shape)}")
    return mask


def init_seed_masks(model: SparseBayesianCNN, activate_frac=0.1, generator=None):
    g = generator or torch.Generator(device="cpu")
    with torch.no_grad():
        for layer in model.all_masked_layers():
            mask = (torch.rand(layer.mask_w.shape, generator=g) < activate_frac).float()
            if mask.sum() == 0:
                mask.view(-1)[0] = 1.0
            layer.mask_w.copy_(mask.to(layer.mask_w.device))

        for conv in model.conv_layers:
            _ensure_filter_io(conv.mask_w)

        _ensure_filter_io(model.fc.mask_w)
        _ensure_filter_io(model.classifier.mask_w)

        for j in range(model.conv_channels[0]):
            if model.conv_layers[1].mask_w[:, j, :, :].sum() < 1:
                o = int(torch.randint(0, model.conv_channels[1], (1,), generator=g).item())
                kh = int(torch.randint(0, 3, (1,)).item())
                kw = int(torch.randint(0, 3, (1,)).item())
                model.conv_layers[1].mask_w[o, j, kh, kw] = 1.0

        last_w = model.conv_channels[1]
        for j in range(last_w):
            if model.fc.mask_w[:, j].sum() < 1:
                o = int(torch.randint(0, model.fc_hidden, (1,), generator=g).item())
                model.fc.mask_w[o, j] = 1.0

        for j in range(model.fc_hidden):
            if model.classifier.mask_w[:, j].sum() < 1:
                o = int(torch.randint(0, model.num_classes, (1,), generator=g).item())
                model.classifier.mask_w[o, j] = 1.0


def _channel_means_4d(x):
    return x.abs().mean(dim=(0, 2, 3))


def _eval_loss_on_batches(model, batches, beta_scaled):
    from lib.train import loss_function

    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in batches:
            outputs = model(inputs, mean_field=True)
            loss, _, _ = loss_function(outputs, labels, model.kl_loss(), beta_scaled)
            losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def _collect_growth_batches(loader, device, max_batches):
    batches = []
    for b_idx, (inputs, labels) in enumerate(loader):
        if b_idx >= max_batches:
            break
        batches.append((inputs.to(device), labels.to(device)))
    if not batches:
        raise RuntimeError("No batches available for growth signal collection")
    return batches


def _collect_growth_signals(model, batches, beta_scaled):
    from lib.train import loss_function

    model.eval()
    act_sums = [None] * len(model.conv_layers)
    grad_sums = [None] * len(model.conv_layers)
    fc_act_sum = None
    fc_grad_sum = None
    cls_act_sum = None
    cls_grad_sum = None
    n_seen = 0

    for inputs, labels in batches:
        acts = []
        preacts = []
        x = inputs
        for conv, gn in zip(model.conv_layers, model.gn_layers):
            acts.append(x.detach())
            x = conv(x, mean_field=True)
            x.retain_grad()
            preacts.append(x)
            x = gn(x)
            x = F.silu(x)

        fc_input = F.adaptive_avg_pool2d(x, 1).flatten(1)
        fc_out = model.fc(fc_input, mean_field=True)
        fc_out.retain_grad()
        logits = model.classifier(F.silu(fc_out), mean_field=True)
        logits.retain_grad()

        loss, _, _ = loss_function(logits, labels, model.kl_loss(), beta_scaled)
        model.zero_grad(set_to_none=True)
        loss.backward()

        for i, (act, pre) in enumerate(zip(acts, preacts)):
            a_mean = _channel_means_4d(act)
            g_mean = _channel_means_4d(pre.grad.detach())
            if act_sums[i] is None:
                act_sums[i] = a_mean
                grad_sums[i] = g_mean
            else:
                act_sums[i] = act_sums[i] + a_mean
                grad_sums[i] = grad_sums[i] + g_mean

        fc_a = fc_input.detach().mean(dim=0)
        fc_g = fc_out.grad.detach().mean(dim=0)
        cls_a = F.silu(fc_out).detach().mean(dim=0)
        cls_g = logits.grad.detach().mean(dim=0)
        if fc_act_sum is None:
            fc_act_sum = fc_a
            fc_grad_sum = fc_g
            cls_act_sum = cls_a
            cls_grad_sum = cls_g
        else:
            fc_act_sum = fc_act_sum + fc_a
            fc_grad_sum = fc_grad_sum + fc_g
            cls_act_sum = cls_act_sum + cls_a
            cls_grad_sum = cls_grad_sum + cls_g
        n_seen += 1

    acts_out = [a / n_seen for a in act_sums]
    grads_out = [g / n_seen for g in grad_sums]
    linear_acts = [fc_act_sum / n_seen, cls_act_sum / n_seen]
    linear_grads = [fc_grad_sum / n_seen, cls_grad_sum / n_seen]
    return acts_out, grads_out, linear_acts, linear_grads


def connection_growth_step(
    model,
    loader,
    device,
    beta_scaled,
    conn_grow_frac=0.01,
    max_batches=8,
):
    batches = _collect_growth_batches(loader, device, max_batches)
    acts, grads, linear_acts, linear_grads = _collect_growth_signals(
        model, batches, beta_scaled
    )
    grown_total = 0
    per_layer = []

    with torch.no_grad():
        for layer, x, g in zip(model.conv_layers, acts, grads):
            score = torch.einsum("o,i->oi", g.abs(), x.abs())
            score = score.unsqueeze(-1).unsqueeze(-1).expand_as(layer.mask_w)
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
            grown = 0
            for flat_i in topk.indices.tolist():
                if flat[flat_i] < 0:
                    continue
                idx = torch.unravel_index(torch.tensor(flat_i), layer.mask_w.shape)
                oi, ij, kh, kw = [int(v) for v in idx]
                if layer.mask_w[oi, ij, kh, kw] > 0.5:
                    continue
                layer.mask_w[oi, ij, kh, kw] = 1.0
                fan_in = max(int(layer.mask_w[oi, :, :, :].sum().item()), 1)
                std = math.sqrt(2.0 / fan_in)
                layer.mu_w[oi, ij, kh, kw] = torch.randn((), device=layer.mu_w.device) * std
                layer.rho_w[oi, ij, kh, kw] = DEFAULT_RHO
                grown += 1
            grown_total += grown
            per_layer.append(grown)

        for layer, x, g in zip(model.linear_masked_layers(), linear_acts, linear_grads):
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
            grown = 0
            for flat_i in topk.indices.tolist():
                if flat[flat_i] < 0:
                    continue
                oi = flat_i // layer.in_features
                ij = flat_i % layer.in_features
                if layer.mask_w[oi, ij] > 0.5:
                    continue
                layer.mask_w[oi, ij] = 1.0
                fan_in = max(int(layer.mask_w[oi].sum().item()), 1)
                std = math.sqrt(2.0 / fan_in)
                layer.mu_w[oi, ij] = torch.randn((), device=layer.mu_w.device) * std
                layer.rho_w[oi, ij] = DEFAULT_RHO
                grown += 1
            grown_total += grown
            per_layer.append(grown)

    return {"grown_connections": grown_total, "per_layer": per_layer}


def _expand_conv_out(conv: MaskedBayesianConv2d, new_out: int):
    if new_out <= conv.out_channels:
        return
    add = new_out - conv.out_channels
    device = conv.mu_w.device
    kH, kW = conv.kernel_size

    def cat_param(param, fill):
        extra = fill((add, conv.in_channels, kH, kW), device)
        return nn.Parameter(torch.cat([param.data, extra], dim=0))

    def cat_buf(buf, fill):
        extra = fill((add, conv.in_channels, kH, kW), device)
        return torch.cat([buf, extra], dim=0)

    conv.mu_w = cat_param(conv.mu_w, lambda shape, dev: torch.zeros(shape, device=dev))
    conv.rho_w = cat_param(
        conv.rho_w, lambda shape, dev: torch.full(shape, DEFAULT_RHO, device=dev)
    )
    conv.mask_w = cat_buf(conv.mask_w, lambda shape, dev: torch.zeros(shape, device=dev))
    conv.prior_mu_w = cat_buf(conv.prior_mu_w, lambda shape, dev: torch.zeros(shape, device=dev))
    conv.prior_sigma_w = cat_buf(
        conv.prior_sigma_w, lambda shape, dev: torch.ones(shape, device=dev)
    )
    if conv.bias_flag:
        conv.mu_b = nn.Parameter(
            torch.cat([conv.mu_b.data, torch.zeros(add, device=device)], dim=0)
        )
        conv.rho_b = nn.Parameter(
            torch.cat(
                [conv.rho_b.data, torch.full((add,), DEFAULT_RHO, device=device)],
                dim=0,
            )
        )
        conv.prior_mu_b = torch.cat([conv.prior_mu_b, torch.zeros(add, device=device)], dim=0)
        conv.prior_sigma_b = torch.cat(
            [conv.prior_sigma_b, torch.ones(add, device=device)], dim=0
        )
    conv.out_channels = new_out


def _expand_conv_in(conv: MaskedBayesianConv2d, new_in: int):
    if new_in <= conv.in_channels:
        return
    add = new_in - conv.in_channels
    device = conv.mu_w.device
    kH, kW = conv.kernel_size
    out_f = conv.out_channels

    conv.mu_w = nn.Parameter(
        torch.cat(
            [conv.mu_w.data, torch.zeros(out_f, add, kH, kW, device=device)],
            dim=1,
        )
    )
    conv.rho_w = nn.Parameter(
        torch.cat(
            [
                conv.rho_w.data,
                torch.full((out_f, add, kH, kW), DEFAULT_RHO, device=device),
            ],
            dim=1,
        )
    )
    conv.mask_w = torch.cat(
        [conv.mask_w, torch.zeros(out_f, add, kH, kW, device=device)],
        dim=1,
    )
    conv.prior_mu_w = torch.cat(
        [conv.prior_mu_w, torch.zeros(out_f, add, kH, kW, device=device)],
        dim=1,
    )
    conv.prior_sigma_w = torch.cat(
        [conv.prior_sigma_w, torch.ones(out_f, add, kH, kW, device=device)],
        dim=1,
    )
    conv.in_channels = new_in


def _expand_linear_in(layer: MaskedBayesianLinear, new_in: int):
    if new_in <= layer.in_features:
        return
    add = new_in - layer.in_features
    device = layer.mu_w.device
    out_f = layer.out_features
    layer.mu_w = nn.Parameter(
        torch.cat([layer.mu_w.data, torch.zeros(out_f, add, device=device)], dim=1)
    )
    layer.rho_w = nn.Parameter(
        torch.cat(
            [
                layer.rho_w.data,
                torch.full((out_f, add), DEFAULT_RHO, device=device),
            ],
            dim=1,
        )
    )
    layer.mask_w = torch.cat(
        [layer.mask_w, torch.zeros(out_f, add, device=device)],
        dim=1,
    )
    layer.prior_mu_w = torch.cat(
        [layer.prior_mu_w, torch.zeros(out_f, add, device=device)],
        dim=1,
    )
    layer.prior_sigma_w = torch.cat(
        [layer.prior_sigma_w, torch.ones(out_f, add, device=device)],
        dim=1,
    )
    layer.in_features = new_in


def _expand_gn(gn: nn.GroupNorm, new_channels: int):
    old = gn.num_channels
    if new_channels <= old:
        return gn
    device = gn.weight.device
    new_gn = nn.GroupNorm(1, new_channels).to(device)
    with torch.no_grad():
        new_gn.weight[:old] = gn.weight.data
        new_gn.bias[:old] = gn.bias.data
        new_gn.weight[old:] = 1.0
        new_gn.bias[old:] = 0.0
    return new_gn


def _random_candidate_kernel(in_ch, kH, kW, device, activate_frac=0.3):
    mask = (torch.rand(in_ch, kH, kW, device=device) < activate_frac).float()
    if mask.sum() == 0:
        mask.view(-1)[0] = 1.0
    mu = torch.randn(in_ch, kH, kW, device=device) * 0.1
    return mu * mask, mask


def feature_map_growth_step(
    model: SparseBayesianCNN,
    loader,
    device,
    beta_scaled,
    conv_layer_idx,
    num_candidates=12,
    max_batches=8,
    outgoing_activate_frac=0.3,
):
    if conv_layer_idx not in (0, 1):
        raise ValueError("conv_layer_idx must be 0 or 1")

    batches = _collect_growth_batches(loader, device, max_batches)
    baseline_loss = _eval_loss_on_batches(model, batches, beta_scaled)

    in_layer = model.conv_layers[conv_layer_idx]
    old_out = in_layer.out_channels
    new_out = old_out + 1
    new_row = old_out

    _expand_conv_out(in_layer, new_out)
    model.gn_layers[conv_layer_idx] = _expand_gn(
        model.gn_layers[conv_layer_idx], new_out
    ).to(device)
    model.conv_channels[conv_layer_idx] = new_out

    if conv_layer_idx == 0:
        out_layer = model.conv_layers[1]
        _expand_conv_in(out_layer, new_out)
        new_col = old_out
    else:
        out_layer = model.fc
        _expand_linear_in(out_layer, new_out)
        new_col = old_out

    best_loss = float("inf")
    best_state = None
    candidate_losses = []

    for _ in range(num_candidates):
        with torch.no_grad():
            mu_k, mask_k = _random_candidate_kernel(
                in_layer.in_channels,
                in_layer.kernel_size[0],
                in_layer.kernel_size[1],
                in_layer.mu_w.device,
            )
            in_layer.mask_w[new_row] = mask_k
            in_layer.mu_w[new_row] = mu_k
            in_layer.rho_w[new_row] = DEFAULT_RHO
            in_layer.mu_b[new_row] = 0.0
            in_layer.rho_b[new_row] = DEFAULT_RHO

            if conv_layer_idx == 0:
                out_mask = (
                    torch.rand_like(out_layer.mask_w[:, new_col, :, :])
                    < outgoing_activate_frac
                ).float()
                if out_mask.sum() == 0:
                    out_mask.view(-1)[0] = 1.0
                out_layer.mask_w[:, new_col, :, :] = out_mask
                out_layer.mu_w[:, new_col, :, :] = (
                    torch.randn_like(out_layer.mu_w[:, new_col, :, :]) * 0.05
                )
                out_layer.rho_w[:, new_col, :, :] = DEFAULT_RHO
            else:
                out_mask = (
                    torch.rand(out_layer.out_features, device=device) < outgoing_activate_frac
                ).float()
                if out_mask.sum() == 0:
                    out_mask[0] = 1.0
                out_layer.mask_w[:, new_col] = out_mask
                out_layer.mu_w[:, new_col] = torch.randn(
                    out_layer.out_features, device=device
                ) * 0.05
                out_layer.rho_w[:, new_col] = DEFAULT_RHO

        cand_loss = _eval_loss_on_batches(model, batches, beta_scaled)
        candidate_losses.append(cand_loss)
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    if conv_layer_idx == 0:
        active_out = int(out_layer.mask_w[:, new_col, :, :].sum().item())
    else:
        active_out = int(out_layer.mask_w[:, new_col].sum().item())

    return {
        "conv_layer_idx": conv_layer_idx,
        "new_width": model.conv_channels[conv_layer_idx],
        "baseline_loss": baseline_loss,
        "best_loss": best_loss,
        "delta_loss": best_loss - baseline_loss,
        "num_candidates": num_candidates,
        "candidate_losses": candidate_losses,
        "active_in": int(in_layer.mask_w[new_row].sum().item()),
        "active_out": active_out,
    }


def effective_weight_prune_step(model: SparseBayesianCNN, prune_frac=0.01):
    pruned_total = 0
    per_layer = []
    with torch.no_grad():
        for conv, gn in zip(model.conv_layers, model.gn_layers):
            active = conv.mask_w > 0.5
            n_active = int(active.sum().item())
            if n_active == 0:
                per_layer.append(0)
                continue
            k = max(1, int(math.floor(prune_frac * n_active)))
            k = min(k, n_active)
            eff = conv.effective_weight_magnitude(gn)
            eff_masked = eff.clone()
            eff_masked[~active] = float("inf")
            flat = eff_masked.view(-1)
            bottom = torch.topk(flat, k=k, largest=False)
            pruned = 0
            for flat_i in bottom.indices.tolist():
                idx = torch.unravel_index(torch.tensor(flat_i), conv.mask_w.shape)
                oi, ij, kh, kw = [int(v) for v in idx]
                if conv.mask_w[oi, ij, kh, kw] < 0.5:
                    continue
                conv.mask_w[oi, ij, kh, kw] = 0.0
                pruned += 1
            pruned_total += pruned
            per_layer.append(pruned)

        for layer in model.linear_masked_layers():
            active = layer.mask_w > 0.5
            n_active = int(active.sum().item())
            if n_active == 0:
                per_layer.append(0)
                continue
            k = max(1, int(math.floor(prune_frac * n_active)))
            k = min(k, n_active)
            eff = layer.effective_weight_magnitude()
            eff_masked = eff.clone()
            eff_masked[~active] = float("inf")
            flat = eff_masked.view(-1)
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


def cleanup_dead_filters(model: SparseBayesianCNN):
    removed = 0
    for h_idx in (1, 0):
        in_layer = model.conv_layers[h_idx]
        if h_idx == 0:
            out_layer = model.conv_layers[1]
            out_is_conv = True
        else:
            out_layer = model.fc
            out_is_conv = False

        width = in_layer.out_channels
        keep = []
        for j in range(width):
            fan_in = int(in_layer.mask_w[j].sum().item())
            if out_is_conv:
                fan_out = int(out_layer.mask_w[:, j, :, :].sum().item())
            else:
                fan_out = int(out_layer.mask_w[:, j].sum().item())
            if fan_in > 0 and fan_out > 0:
                keep.append(j)
            else:
                removed += 1

        if len(keep) == width:
            continue
        if len(keep) == 0:
            keep = [0]
            removed -= 1
            in_layer.mask_w[0, 0, 0, 0] = 1.0
            if out_is_conv:
                out_layer.mask_w[0, 0, 0, 0] = 1.0
            else:
                out_layer.mask_w[0, 0] = 1.0

        keep_t = torch.tensor(keep, device=in_layer.mu_w.device, dtype=torch.long)
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
        in_layer.out_channels = len(keep)

        old_gn = model.gn_layers[h_idx]
        new_gn = nn.GroupNorm(1, len(keep)).to(in_layer.mu_w.device)
        with torch.no_grad():
            new_gn.weight.copy_(old_gn.weight.data[keep_t])
            new_gn.bias.copy_(old_gn.bias.data[keep_t])
        model.gn_layers[h_idx] = new_gn

        if out_is_conv:
            out_layer.mu_w = nn.Parameter(out_layer.mu_w.data[:, keep_t].clone())
            out_layer.rho_w = nn.Parameter(out_layer.rho_w.data[:, keep_t].clone())
            out_layer.mask_w = out_layer.mask_w[:, keep_t].clone()
            out_layer.prior_mu_w = out_layer.prior_mu_w[:, keep_t].clone()
            out_layer.prior_sigma_w = out_layer.prior_sigma_w[:, keep_t].clone()
            out_layer.in_channels = len(keep)
        else:
            out_layer.mu_w = nn.Parameter(out_layer.mu_w.data[:, keep_t].clone())
            out_layer.rho_w = nn.Parameter(out_layer.rho_w.data[:, keep_t].clone())
            out_layer.mask_w = out_layer.mask_w[:, keep_t].clone()
            out_layer.prior_mu_w = out_layer.prior_mu_w[:, keep_t].clone()
            out_layer.prior_sigma_w = out_layer.prior_sigma_w[:, keep_t].clone()
            out_layer.in_features = len(keep)

        model.conv_channels[h_idx] = len(keep)

    return {"filters_removed": removed, "conv_channels": list(model.conv_channels)}
