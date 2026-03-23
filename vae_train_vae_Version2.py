import argparse
import os
import torch
from torch.utils.data import DataLoader
from vae.model import Conv3dVAE
from craftsman_dataset.neudf_dataset import NeUDFGridDataset
import torch.optim as optim
import numpy as np

def loss_fn(recon, x, mu, logvar):
    bce = torch.nn.functional.mse_loss(recon, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + 1e-4 * kld, bce, kld

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--out_dir', type=str, default='vae/ckpts')
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = NeUDFGridDataset(args.data_dir, max_samples=args.num_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = Conv3dVAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for i, x in enumerate(dl):
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, bce, kld = loss_fn(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{args.epochs} loss={total_loss/len(dl):.6f}')
        torch.save(model.state_dict(), os.path.join(args.out_dir, f'vae_epoch{epoch+1}.pth'))

    # export some latents
    model.eval()
    latents_dir = os.path.join(args.out_dir, 'latents')
    os.makedirs(latents_dir, exist_ok=True)
    idx = 0
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            zs = z.cpu().numpy()
            for j in range(zs.shape[0]):
                np.save(os.path.join(latents_dir, f'latent_{idx}.npy'), zs[j])
                idx += 1
            if idx >= args.num_samples:
                break
    print('Saved latents to', latents_dir)