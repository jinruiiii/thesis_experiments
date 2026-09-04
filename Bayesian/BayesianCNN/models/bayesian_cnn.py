import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bayesian_linear import BayesianLinear


class BayesianConv2d(nn.Module):
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
        self.mu_w = nn.Parameter(
            torch.randn(out_channels, in_channels, kH, kW) * 0.1
        )
        self.rho_w = nn.Parameter(torch.ones(out_channels, in_channels, kH, kW) * -3)

        if self.bias_flag:
            self.mu_b = nn.Parameter(torch.randn(out_channels) * 0.1)
            self.rho_b = nn.Parameter(torch.ones(out_channels) * -3)

        self.register_buffer(
            "prior_mu_w", torch.zeros(out_channels, in_channels, kH, kW)
        )
        self.register_buffer(
            "prior_sigma_w", torch.ones(out_channels, in_channels, kH, kW)
        )
        if self.bias_flag:
            self.register_buffer("prior_mu_b", torch.zeros(out_channels))
            self.register_buffer("prior_sigma_b", torch.ones(out_channels))

    def _sigma(self, rho):
        return F.softplus(rho)

    def set_prior_from_posterior(self):
        """Freeze current variational posterior as the KL prior (VCL-style)."""
        with torch.no_grad():
            self.prior_mu_w.copy_(self.mu_w.detach())
            self.prior_sigma_w.copy_(self._sigma(self.rho_w).detach().clamp_min(1e-6))
            if self.bias_flag:
                self.prior_mu_b.copy_(self.mu_b.detach())
                self.prior_sigma_b.copy_(self._sigma(self.rho_b).detach().clamp_min(1e-6))

    def forward(self, x):
        sigma_w = self._sigma(self.rho_w)
        eps_w = torch.randn_like(sigma_w)
        w = self.mu_w + sigma_w * eps_w

        if self.bias_flag:
            sigma_b = self._sigma(self.rho_b)
            eps_b = torch.randn_like(sigma_b)
            b = self.mu_b + sigma_b * eps_b
            return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)

        return F.conv2d(x, w, None, stride=self.stride, padding=self.padding)

    def kl_loss(self):
        posterior_w = torch.distributions.Normal(self.mu_w, self._sigma(self.rho_w))
        prior_w = torch.distributions.Normal(
            self.prior_mu_w, self.prior_sigma_w.clamp_min(1e-6)
        )
        kl = torch.distributions.kl_divergence(posterior_w, prior_w).sum()

        if self.bias_flag:
            posterior_b = torch.distributions.Normal(self.mu_b, self._sigma(self.rho_b))
            prior_b = torch.distributions.Normal(
                self.prior_mu_b, self.prior_sigma_b.clamp_min(1e-6)
            )
            kl += torch.distributions.kl_divergence(posterior_b, prior_b).sum()

        return kl


class BayesianCNN(nn.Module):
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
                BayesianConv2d(
                    prev_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.gn_layers.append(nn.GroupNorm(1, out_ch))
            prev_ch = out_ch

        self.fc = BayesianLinear(prev_ch, self.fc_hidden)
        self.classifier = BayesianLinear(self.fc_hidden, num_classes)

    def forward(self, x):
        for conv, gn in zip(self.conv_layers, self.gn_layers):
            x = conv(x)
            x = gn(x)
            x = F.silu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.silu(x)
        x = self.classifier(x)
        return x

    def kl_loss(self):
        kl = sum(layer.kl_loss() for layer in self.conv_layers)
        kl += self.fc.kl_loss()
        kl += self.classifier.kl_loss()
        return kl

    def set_prior_from_posterior(self):
        """Freeze current variational posterior as the KL prior (VCL-style)."""
        for layer in self.conv_layers:
            layer.set_prior_from_posterior()
        self.fc.set_prior_from_posterior()
        self.classifier.set_prior_from_posterior()

    def get_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
        }

    def _weight_variance(self, rho):
        sigma = F.softplus(rho)
        return sigma ** 2

    def _filter_uncertainty_incoming(self, conv_layer):
        """Mean incoming weight variance per output filter."""
        var_w = self._weight_variance(conv_layer.rho_w)
        unc_w = var_w.mean(dim=(1, 2, 3))
        return unc_w

    def _filter_uncertainty_outgoing(self, next_conv, filter_idx):
        """Mean outgoing weight variance for filter j via input channel j of next conv."""
        var_w = self._weight_variance(next_conv.rho_w)
        return var_w[:, filter_idx, :, :].mean()

    def _filter_uncertainty_bidirectional(
        self, layer_idx, filter_idx, combine="geometric", eps=1e-8
    ):
        conv_layer = self.conv_layers[layer_idx]
        unc_in = self._filter_uncertainty_incoming(conv_layer)[filter_idx]
        if layer_idx + 1 < len(self.conv_layers):
            unc_out = self._filter_uncertainty_outgoing(
                self.conv_layers[layer_idx + 1], filter_idx
            )
        else:
            # Last conv feeds the hidden FC after adaptive average pooling.
            rho = self.fc.rho_w
            var_w = self._weight_variance(rho)
            unc_out = var_w[:, filter_idx].mean()
        if combine == "min":
            return torch.min(unc_in, unc_out)
        if combine == "mean":
            return 0.5 * (unc_in + unc_out)
        return torch.sqrt(unc_in * unc_out + eps)

    def get_average_bidirectional_uncertainty_per_layer(self, combine="geometric", eps=1e-8):
        average_uncertainty_per_layer = []
        for layer_idx, conv_layer in enumerate(self.conv_layers):
            n_filters = conv_layer.out_channels
            scores = [
                self._filter_uncertainty_bidirectional(
                    layer_idx, j, combine=combine, eps=eps
                )
                for j in range(n_filters)
            ]
            average_uncertainty_per_layer.append(torch.stack(scores).mean())
        return average_uncertainty_per_layer


def test_model_shape():
    print("Testing BayesianCNN shape...")
    x = torch.randn(8, 3, 32, 32)
    model = BayesianCNN(3, [32, 64], 10, fc_hidden=48)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    assert model.fc.in_features == 64
    assert model.fc.out_features == 48
    assert model.classifier.in_features == 48
    assert model.classifier.out_features == 10
    stats = model.get_param_stats()
    print("Stats:", stats)
    unc = model.get_average_bidirectional_uncertainty_per_layer(combine="geometric")
    print("Layer uncertainties:", [u.item() for u in unc])


if __name__ == "__main__":
    test_model_shape()
