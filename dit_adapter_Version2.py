import torch
import torch.nn as nn

class LatentToTokensAdapter(nn.Module):
    def __init__(self, latent_dim=256, token_dim=512, num_tokens=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.proj = nn.Linear(latent_dim, num_tokens * token_dim)

    def forward(self, z):
        # z: (B, latent_dim) -> tokens: (B, num_tokens, token_dim)
        x = self.proj(z)
        x = x.view(z.shape[0], self.num_tokens, self.token_dim)
        return x