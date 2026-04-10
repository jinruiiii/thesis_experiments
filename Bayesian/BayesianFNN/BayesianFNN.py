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
            
        self.prior = torch.distributions.Normal(0, 1)

    def forward(self, x):
        sigma_w = torch.log1p(torch.exp(self.rho_w))
        eps_w = torch.randn_like(sigma_w)
        w = self.mu_w + sigma_w * eps_w
        
        if self.bias_flag:
            sigma_b = torch.log1p(torch.exp(self.rho_b))
            eps_b = torch.randn_like(sigma_b)
            b = self.mu_b + sigma_b * eps_b
            return F.linear(x, w, b)
            
        return F.linear(x, w)

    def kl_loss(self):
        posterior_w = torch.distributions.Normal(self.mu_w, torch.log1p(torch.exp(self.rho_w)))
        kl = torch.distributions.kl_divergence(posterior_w, self.prior).sum()

        if self.bias_flag:
            posterior_b = torch.distributions.Normal(self.mu_b, torch.log1p(torch.exp(self.rho_b)))
            kl += torch.distributions.kl_divergence(posterior_b, self.prior).sum()

        return kl

    def snr(self):
        """
        Computes the Signal-to-Noise Ratio (SNR) for the layer.
    
        SNR is defined as |mu| / sigma for each weight and bias.
    
        Returns:
            Tensor: 
                - If bias is True: concatenated tensor of shape [out_features, in_features + 1]  containing SNRs for weights and biases.
                - If bias is False: tensor of shape [out_features, in_features] containing SNRs for weights only.
        """    
        # numerical stability
        eps = 1e-8 
        sigma_w = F.softplus(self.rho_w)
        snr = torch.abs(self.mu_w) / (sigma_w +eps)
        if self.bias_flag:
            sigma_b = F.softplus(self.rho_b)
            snr_b = torch.abs(self.mu_b) / (sigma_b +eps)
            snr = torch.cat((snr, snr_b.unsqueeze(1)), dim=1)
        return snr

class BayesianFNN(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.out_features = out_features

        self.layers = nn.ModuleList()
        prev_dim = in_features
        for current_dim in self.hidden_sizes:
            self.layers.append(BayesianLinear(prev_dim, current_dim))
            prev_dim = current_dim
        self.out = BayesianLinear(prev_dim, self.out_features)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.out(x)
        return x

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
            snr = layer.snr()
            average_snr_per_layer.append(torch.sum(snr)/snr.numel())
        return average_snr_per_layer


def test_model_shape():
    print("Testing model shape...")
    x = torch.randn(16,32)
    model = BayesianFNN(32, [64,16], 10)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    stats = model.get_param_stats()
    print("Stats:", stats)

if __name__ == "__main__":
    test_model_shape()