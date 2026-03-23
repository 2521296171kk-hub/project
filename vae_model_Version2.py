import torch
import torch.nn as nn

class Conv3dVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # compute flatten size for 64^3 input: after 3 halving -> 8^3; channels 128 => 128*8*8*8
        self._flatten_dim = 128 * 8 * 8 * 8
        self.fc_mu = nn.Linear(self._flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flatten_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self._flatten_dim)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (128,8,8,8)),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, in_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        x = self.dec(h)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar