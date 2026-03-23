# Placeholder sampling pipeline: loads adapter (as stand-in for DiT) and VAE decoder to produce UDF grid reconstructions
import argparse
import os
import numpy as np
import torch
from vae.model import Conv3dVAE
from dit.adapter import LatentToTokensAdapter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dit_ckpt', type=str)
    parser.add_argument('--vae_ckpt', type=str)
    parser.add_argument('--out_dir', type=str, default='outputs')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = Conv3dVAE()
    if args.vae_ckpt:
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'))
    vae.to(device).eval()
    adapter = LatentToTokensAdapter()
    if args.dit_ckpt:
        adapter.load_state_dict(torch.load(args.dit_ckpt, map_location='cpu'))
    adapter.to(device).eval()
    # sample random latent
    z = torch.randn(1, 256).to(device)
    # pretend adapter/DiT transforms z (placeholder)
    tokens = adapter(z)
    # reconstruct from z via VAE decoder
    with torch.no_grad():
        recon = vae.decode(z).cpu().numpy()
    np.save(os.path.join(args.out_dir, 'recon_example.npy'), recon)
    print('Saved recon to', args.out_dir)