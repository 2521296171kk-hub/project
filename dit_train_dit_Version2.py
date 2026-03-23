# Placeholder script to illustrate training DiT on latents. Real DiT implementation should be used.
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dit.adapter import LatentToTokensAdapter

class LatentDataset(Dataset):
    def __init__(self, latents_dir, max_samples=None):
        files = [os.path.join(latents_dir,f) for f in os.listdir(latents_dir) if f.endswith('.npy')]
        self.files = files[:max_samples] if max_samples else files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        z = np.load(self.files[idx]).astype(np.float32)
        return torch.from_numpy(z)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents_dir', required=True)
    parser.add_argument('--out_dir', default='dit/ckpts')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = LatentDataset(args.latents_dir)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adapter = LatentToTokensAdapter().to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    for epoch in range(args.epochs):
        adapter.train()
        for z in dl:
            z = z.to(device)
            tokens = adapter(z)
            loss = tokens.square().mean()  # placeholder
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(adapter.state_dict(), os.path.join(args.out_dir, f'adapter_epoch{epoch+1}.pth'))
        print('Epoch', epoch+1)