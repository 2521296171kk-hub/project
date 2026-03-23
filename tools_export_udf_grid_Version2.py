# Placeholder script to export UDF from NeUDF model if available.
# Here we provide a simple interface: load model, sample points in bbox, and save grid.
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

# This script assumes NeUDF's SDFNetwork is importable as models.fields.SDFNetwork
try:
    from models.fields import SDFNetwork
except Exception:
    SDFNetwork = None

def sample_udf_from_model(model, bbox_min, bbox_max, resolution=64, device='cuda'):
    xs = np.linspace(bbox_min, bbox_max, resolution)
    grid = np.stack(np.meshgrid(xs, xs, xs, indexing='xy'), axis=-1).reshape(-1,3).astype(np.float32)
    udfs = np.zeros((grid.shape[0],), dtype=np.float32)
    batch = 1 << 20
    model.to(device).eval()
    with torch.no_grad():
        for i in tqdm(range(0, grid.shape[0], batch)):
            pts = torch.from_numpy(grid[i:i+batch]).to(device)
            sdf = model.sdf(pts).squeeze(-1).cpu().numpy()
            udfs[i:i+batch] = np.abs(sdf)
    return udfs.reshape((resolution, resolution, resolution))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, help='Path to NeUDF checkpoint (optional)')
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--bbox', type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    if SDFNetwork is None or args.model_ckpt is None:
        print('NeUDF SDFNetwork not available or checkpoint not provided. Exiting.')
        exit(1)
    # NOTE: user must adapt model init args
    model = SDFNetwork(d_in=3, d_out=257, d_hidden=256, n_layers=8)
    ckpt = torch.load(args.model_ckpt, map_location='cpu')
    model.load_state_dict(ckpt)
    grid = sample_udf_from_model(model, args.bbox[0], args.bbox[1], resolution=args.resolution, device=args.device)
    np.save(args.out_path, grid.astype(np.float32))
    print('Saved', args.out_path)