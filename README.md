# NeUDF + CraftsMan Integration (A-path)

This repository contains initial scaffolding to implement Path A: generate UDF training data from meshes, train a 3D-Conv VAE on UDF grids, and prepare latent tokens for a DiT-style diffusion model. It is intended as a starting point; you will collect data later and run training on your A800 GPU.

Quick start (prototype):
- Generate UDF grids from meshes: python tools/mesh_to_udf.py --mesh_dir data/meshes --out_dir data/udf --resolution 64
- Train VAE (64^3): python vae/train_vae.py --data_dir data/udf --epochs 50 --batch_size 8 --num_samples 1000
- Prepare latent tokens for DiT: python dit/train_dit.py --latents_dir vae/latents --out_dir dit/data
- Sample (placeholder): python scripts/sample_from_dit.py --dit_ckpt path/to/dit.ckpt --vae_ckpt path/to/vae.ckpt

See each script for details and options. Adjust resolutions and batch sizes to your hardware.