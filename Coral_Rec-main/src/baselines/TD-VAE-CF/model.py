import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class TD_VAE_CF(nn.Module):
    def __init__(self, opt, num_node):
        super(TD_VAE_CF, self).__init__()
        self.num_node = num_node
        self.hidden_dim = opt.hiddenSize
        self.latent_dim = opt.hiddenSize // 2  # 通常潜空间维度较小
        self.dropout_p = getattr(opt, 'dropout', 0.5)
        
        # Encoder
        self.encoder_dims = [self.num_node, self.hidden_dim, self.latent_dim * 2]
        self.encoder_modules = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.encoder_dims[:-1], self.encoder_dims[1:])):
            if i == len(self.encoder_dims) - 2:
                self.encoder_modules.append(nn.Linear(d_in, d_out))
            else:
                self.encoder_modules.append(nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.Tanh()
                ))

        # Decoder
        self.decoder_dims = [self.latent_dim, self.hidden_dim, self.num_node]
        self.decoder_modules = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.decoder_dims[:-1], self.decoder_dims[1:])):
            if i == len(self.decoder_dims) - 2:
                self.decoder_modules.append(nn.Linear(d_in, d_out))
            else:
                self.decoder_modules.append(nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.Tanh()
                ))
        
        self.dropout = nn.Dropout(self.dropout_p)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def encode(self, x):
        h = F.normalize(x)
        h = self.dropout(h)
        for layer in self.encoder_modules:
            h = layer(h)
        
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        h = z
        for layer in self.decoder_modules:
            h = layer(h)
        return h  # Logits

    def forward(self, x, intervention_params=None):
        """
        x: User interaction vector (Batch_size, Num_items)
        intervention_params: dict, contains 'target_direction', 'lambda', etc.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if not self.training and intervention_params is not None:
            z = self.apply_intervention(z, intervention_params)
        # ------------------------------------

        logits = self.decode(z)
        return logits, mu, logvar

    def apply_intervention(self, z, params):
        
        if 'direction' in params and 'lambda' in params:
            direction = params['direction'].to(z.device)
            strength = params['lambda']

            direction = F.normalize(direction, dim=0)
            dot_prod = torch.matmul(z, direction) # (Batch,)
            proj = dot_prod.unsqueeze(1) * direction.unsqueeze(0) # (Batch, Latent)
            z_new = z - strength * proj
            return z_new
        return z

    def loss_function(self, recon_x, x, mu, logvar, anneal=1.0):
        # Multinomial Likelihood (Standard for Multi-VAE)
        log_softmax_var = F.log_softmax(recon_x, dim=1)
        neg_ll = - torch.sum(log_softmax_var * x, dim=1)
        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(neg_ll + anneal * kl_div)