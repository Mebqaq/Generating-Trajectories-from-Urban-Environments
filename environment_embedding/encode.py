"""Encode a new city's static features into the frozen source latent space."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .model import MultiHeadGridAutoencoder


def main():
    parser = argparse.ArgumentParser(description='Encode a city with the frozen source autoencoder.')
    parser.add_argument('--features', type=Path, required=True, help='(n_locations, prop_dim + cont_dim) .npy array')
    parser.add_argument('--bundle', type=Path, required=True, help='Directory produced by train_autoencoder.py')
    parser.add_argument('--output', type=Path, required=True, help='Destination .npy for the (n_locations, latent_dim) embedding')
    args = parser.parse_args()

    scaler = np.load(args.bundle / 'scaler.npz')
    prop_dim = int(scaler['prop_dim']); cont_dim = int(scaler['cont_dim'])
    cont_mean = scaler['cont_mean'].astype(np.float64); cont_scale = scaler['cont_scale'].astype(np.float64)
    stats = np.load(args.bundle / 'sem_stats.npz')
    sem_mean, sem_std = stats['sem_mean'], stats['sem_std']
    latent_dim = int(sem_mean.shape[0])

    features = np.load(args.features)
    if features.ndim != 2 or features.shape[1] != prop_dim + cont_dim:
        parser.error(f'--features must be (n_locations, {prop_dim + cont_dim}); got {features.shape}')

    prop_data = features[:, :prop_dim].astype(np.float32)
    prop_data = prop_data / (prop_data.sum(axis=1, keepdims=True) + 1e-8)
    cont_data = np.log1p(features[:, prop_dim:].astype(np.float64))
    cont_data = (cont_data - cont_mean[None, :]) / cont_scale[None, :]

    x = np.concatenate([prop_data, cont_data.astype(np.float32)], axis=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiHeadGridAutoencoder(prop_dim, cont_dim, lat_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(args.bundle / 'ae.pth', map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        z = model.encoder(torch.from_numpy(x).float().to(device)).cpu().numpy()
    sem_norm = ((z - sem_mean[None, :]) / (sem_std[None, :] + 1e-8)).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, sem_norm)
    print(f'wrote {args.output} shape={sem_norm.shape} '
          f'(mean {sem_norm.mean(0).round(4).tolist()}, std {sem_norm.std(0).round(4).tolist()})')


if __name__ == '__main__':
    main()
