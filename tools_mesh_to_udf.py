# Simple mesh -> unsigned distance field exporter
import os
import argparse
import numpy as np
import trimesh
from tqdm import tqdm

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

def mesh_to_udf(mesh_path, resolution=64, bbox_scale=1.05):
    mesh = trimesh.load(mesh_path, force='mesh')
    bbox = mesh.bounds  # (min, max)
    center = (bbox[0] + bbox[1]) / 2.0
    extent = (bbox[1] - bbox[0]).max() * bbox_scale
    mins = center - extent/2.0
    maxs = center + extent/2.0
    xs = np.linspace(mins[0], maxs[0], resolution)
    ys = np.linspace(mins[1], maxs[1], resolution)
    zs = np.linspace(mins[2], maxs[2], resolution)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='xy'), axis=-1).reshape(-1,3)

    if KDTree is None:
        # fallback: use trimesh.proximity which is slower
        distances = mesh.nearest.signed_distance(grid)
        udf = np.abs(distances)
    else:
        nearest, distance = mesh.nearest.on_surface(grid)
        udf = np.linalg.norm(grid - nearest, axis=-1)

    udf_grid = udf.reshape((resolution, resolution, resolution))
    meta = {
        'resolution': resolution,
        'mins': mins.tolist(),
        'maxs': maxs.tolist(),
    }
    return udf_grid, meta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--resolution', type=int, default=64)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mesh_files = [os.path.join(args.mesh_dir, f) for f in os.listdir(args.mesh_dir) if f.lower().endswith(('.obj','.ply','.stl','.glb','.gltf'))]
    for mf in mesh_files:
        name = os.path.splitext(os.path.basename(mf))[0]
        out_npy = os.path.join(args.out_dir, name + f'_udf_{args.resolution}.npy')
        out_meta = os.path.join(args.out_dir, name + f'_udf_{args.resolution}.npz')
        print('Processing', mf)
        udf, meta = mesh_to_udf(mf, resolution=args.resolution)
        np.save(out_npy, udf.astype(np.float32))
        np.savez_compressed(out_meta, **meta)
        print('Saved', out_npy)