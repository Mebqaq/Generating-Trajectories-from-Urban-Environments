"""Multi-head autoencoder for static environmental features."""
import numpy as np
import torch
from torch import nn


class MultiHeadGridAutoencoder(nn.Module):
    def __init__(self, prop_dim, cont_dim, lat_dim=8):
        super().__init__()
        in_dim = prop_dim + cont_dim

        # Encoder: input to latent.
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, lat_dim)
        )

        # Shared decoder trunk.
        self.decoder_shared = nn.Sequential(
            nn.Linear(lat_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )

        # Proportion head: logits trained with softmax + KL divergence.
        self.head_prop = nn.Linear(128, prop_dim)

        # Continuous head: linear output trained with MSE.
        self.head_cont = nn.Linear(128, cont_dim)

    def forward(self, x):
        z = self.encoder(x)
        hidden = self.decoder_shared(z)

        prop_logits = self.head_prop(hidden)
        cont_pred = self.head_cont(hidden)

        return prop_logits, cont_pred, z


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
