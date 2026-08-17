import torch 
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, bias_flag=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias_flag
        
        self.mu_w = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.rho_w = nn.Parameter(torch.ones(out_features, in_features) * -3)
        
        if self.bias_flag:
            self.mu_b = nn.Parameter(torch.randn(out_features) * 0.1)
            self.rho_b = nn.Parameter(torch.ones(out_features) * -3)

        # Elementwise Gaussian prior; defaults match N(0, 1).
        self.register_buffer("prior_mu_w", torch.zeros(out_features, in_features))
        self.register_buffer("prior_sigma_w", torch.ones(out_features, in_features))
        if self.bias_flag:
            self.register_buffer("prior_mu_b", torch.zeros(out_features))
            self.register_buffer("prior_sigma_b", torch.ones(out_features))

    def forward(self, x):
        sigma_w = self._sigma(self.rho_w)
        eps_w = torch.randn_like(sigma_w)
        w = self.mu_w + sigma_w * eps_w
        
        if self.bias_flag:
            sigma_b = self._sigma(self.rho_b)
            eps_b = torch.randn_like(sigma_b)
            b = self.mu_b + sigma_b * eps_b
            return F.linear(x, w, b)
            
        return F.linear(x, w)

    def set_prior_from_posterior(self):
        """Freeze current variational posterior as the KL prior (VCL-style)."""
        with torch.no_grad():
            self.prior_mu_w.copy_(self.mu_w.detach())
            self.prior_sigma_w.copy_(self._sigma(self.rho_w).detach().clamp_min(1e-6))
            if self.bias_flag:
                self.prior_mu_b.copy_(self.mu_b.detach())
                self.prior_sigma_b.copy_(self._sigma(self.rho_b).detach().clamp_min(1e-6))

    def kl_loss(self):
        posterior_w = torch.distributions.Normal(self.mu_w, self._sigma(self.rho_w))
        prior_w = torch.distributions.Normal(self.prior_mu_w, self.prior_sigma_w.clamp_min(1e-6))
        kl = torch.distributions.kl_divergence(posterior_w, prior_w).sum()

        if self.bias_flag:
            posterior_b = torch.distributions.Normal(self.mu_b, self._sigma(self.rho_b))
            prior_b = torch.distributions.Normal(self.prior_mu_b, self.prior_sigma_b.clamp_min(1e-6))
            kl += torch.distributions.kl_divergence(posterior_b, prior_b).sum()

        return kl

    def get_snr(self):
        eps = 1e-8 
        sigma_w = self._sigma(self.rho_w)
        snr = torch.abs(self.mu_w) / (sigma_w +eps)
        if self.bias_flag:
            sigma_b = self._sigma(self.rho_b)
            snr_b = torch.abs(self.mu_b) / (sigma_b +eps)
            snr = torch.cat((snr, snr_b.unsqueeze(1)), dim=1)
        return snr

    def get_uncertainty(self):
        var_w = self._sigma(self.rho_w) ** 2
        if self.bias_flag:
            var_b = self._sigma(self.rho_b) ** 2
            return torch.cat((var_w, var_b.unsqueeze(1)), dim=1)
        return var_w

    def _sigma(self, rho):
        return F.softplus(rho)
        

class BayesianFNN(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.out_features = out_features
        self.layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        prev_dim = in_features
        for current_dim in self.hidden_sizes:
            self.layers.append(BayesianLinear(prev_dim, current_dim))
            self.ln_layers.append(nn.LayerNorm(current_dim))
            prev_dim = current_dim
        self.out = BayesianLinear(prev_dim, self.out_features)

    def forward(self, x):
        for layer, ln in zip(self.layers, self.ln_layers):
            x = layer(x)
            x = ln(x)
            x = F.silu(x)
        x = self.out(x)
        return x

    def set_prior_from_posterior(self):
        """Set KL prior of every Bayesian layer to its current posterior."""
        for layer in self.layers:
            layer.set_prior_from_posterior()
        self.out.set_prior_from_posterior()

    def kl_loss(self):
        kl_loss = sum(layer.kl_loss() for layer in self.layers) + self.out.kl_loss()
        return kl_loss

    def get_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params
        }

    def get_average_snr_per_layer(self):
        average_snr_per_layer = []
        for layer in self.layers:
            snr = layer.get_snr()
            average_snr_per_layer.append(torch.sum(snr)/snr.numel())
        return average_snr_per_layer

    def get_average_uncertainty_per_layer(self):
        average_uncertainty_per_layer = []
        for layer in self.layers:
            uncertainty = layer.get_uncertainty()
            average_uncertainty_per_layer.append(torch.sum(uncertainty)/uncertainty.numel())
        return average_uncertainty_per_layer

    def _weight_variance(self, rho):
        sigma = F.softplus(rho)
        return sigma ** 2

    def _neuron_uncertainty_incoming(self, layer):
        """Mean incoming weight variance per output neuron (row j of rho_w)."""
        var_w = self._weight_variance(layer.rho_w)
        return var_w.mean(dim=1)

    def _neuron_uncertainty_outgoing(self, next_layer, neuron_idx):
        """Mean outgoing weight variance for hidden neuron j via column j of the next layer."""
        var_w = self._weight_variance(next_layer.rho_w)
        return var_w[:, neuron_idx].mean()

    def _neuron_uncertainty_bidirectional(
        self, layer_idx, neuron_idx, combine="geometric", eps=1e-8
    ):
        """
        Combined weight variance uncertainty for neuron j in hidden layer layer_idx.
        combine: 'geometric' (sqrt(in*out)), 'min', or 'mean'
        """
        layer = self.layers[layer_idx]
        unc_in = self._neuron_uncertainty_incoming(layer)[neuron_idx]
        if layer_idx + 1 < len(self.layers):
            unc_out = self._neuron_uncertainty_outgoing(
                self.layers[layer_idx + 1], neuron_idx
            )
        else:
            unc_out = self._neuron_uncertainty_outgoing(self.out, neuron_idx)
        if combine == "min":
            return torch.min(unc_in, unc_out)
        if combine == "mean":
            return 0.5 * (unc_in + unc_out)
        return torch.sqrt(unc_in * unc_out + eps)

    def get_average_bidirectional_uncertainty_per_layer(self, combine="geometric", eps=1e-8):
        """Mean bidirectional weight-variance uncertainty per hidden layer."""
        average_uncertainty_per_layer = []
        for layer_idx, layer in enumerate(self.layers):
            n_neurons = layer.mu_w.shape[0]
            scores = [
                self._neuron_uncertainty_bidirectional(
                    layer_idx, j, combine=combine, eps=eps
                )
                for j in range(n_neurons)
            ]
            average_uncertainty_per_layer.append(torch.stack(scores).mean())
        return average_uncertainty_per_layer


def test_model_shape():
    print("Testing model shape...")
    x = torch.randn(16,32)
    model = BayesianFNN(32, [64,16], 10)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    stats = model.get_param_stats()
    print("Stats:", stats)
    kl0 = model.kl_loss().item()
    model.set_prior_from_posterior()
    kl1 = model.kl_loss().item()
    print(f"KL before set_prior={kl0:.4f}, after set_prior={kl1:.4f} (should be ~0)")

if __name__ == "__main__":
    test_model_shape()
