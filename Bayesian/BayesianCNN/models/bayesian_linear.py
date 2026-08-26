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
        snr = torch.abs(self.mu_w) / (sigma_w + eps)
        if self.bias_flag:
            sigma_b = self._sigma(self.rho_b)
            snr_b = torch.abs(self.mu_b) / (sigma_b + eps)
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
