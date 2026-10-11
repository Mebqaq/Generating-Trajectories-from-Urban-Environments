"""Train the static-environment autoencoder on one source city."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from .model import MultiHeadGridAutoencoder, set_seed


def preprocess(features: np.ndarray, prop_dim: int):
    """Row-normalise proportions and log1p+standardise continuous columns."""
    prop_data = np.asarray(features[:, :prop_dim], dtype=np.float32)
    prop_data = prop_data / (prop_data.sum(axis=1, keepdims=True) + 1e-8)

    cont_data = np.asarray(features[:, prop_dim:], dtype=np.float32)
    cont_data = np.log1p(cont_data)
    scaler = StandardScaler()
    cont_data = scaler.fit_transform(cont_data)

    x = torch.FloatTensor(np.concatenate([prop_data, cont_data], axis=1))
    return x, scaler


def main():
    parser = argparse.ArgumentParser(description='Train the static-feature autoencoder.')
    parser.add_argument('--features', type=Path, required=True, help='(n_locations, prop_dim + cont_dim) .npy array')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prop-dim', type=int, default=11)
    parser.add_argument('--latent-dim', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    for name in ('ae.pth', 'scaler.npz', 'sem_stats.npz', 'raw_embeddings.npy'):
        if (args.output_dir / name).exists():
            parser.error(f'Output already exists: {args.output_dir / name}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = np.load(args.features)
    if features.ndim != 2:
        parser.error(f'--features must be a 2-D array; got {features.shape}')
    prop_dim = args.prop_dim
    cont_dim = features.shape[1] - prop_dim
    if prop_dim < 1 or cont_dim < 1:
        parser.error(f'--prop-dim={prop_dim} leaves {cont_dim} continuous columns; both must be positive')
    print(f'input: {features.shape[0]} locations, {features.shape[1]} columns (prop {prop_dim}, cont {cont_dim})')

    x, scaler = preprocess(features, prop_dim)
    dataset = TensorDataset(x)
    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = MultiHeadGridAutoencoder(prop_dim, cont_dim, lat_dim=args.latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            bx = batch[0].to(device)
            target_prop, target_cont = bx[:, :prop_dim], bx[:, prop_dim:]
            optimizer.zero_grad()
            prop_logits, cont_pred, _ = model(bx)
            loss = F.kl_div(F.log_softmax(prop_logits, dim=-1), target_prop, reduction='batchmean') \
                + F.mse_loss(cont_pred, target_cont)
            loss.backward()
            optimizer.step()

        model.eval()
        validation = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bx = batch[0].to(device)
                target_prop, target_cont = bx[:, :prop_dim], bx[:, prop_dim:]
                prop_logits, cont_pred, _ = model(bx)
                validation += (F.kl_div(F.log_softmax(prop_logits, dim=-1), target_prop, reduction='batchmean').item()
                               + F.mse_loss(cont_pred, target_cont).item())
        validation /= len(val_loader)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  epoch [{epoch+1:3d}/{args.epochs}] val={validation:.4f}')
        if validation < best_val:
            best_val, bad = validation, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f'early stop at epoch {epoch+1}, best_val={best_val:.4f}')
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.encoder(x.to(device)).cpu().numpy()
    sem_mean, sem_std = z.mean(axis=0), z.std(axis=0)
    sem_min, sem_max = z.min(axis=0), z.max(axis=0)

    torch.save(model.state_dict(), args.output_dir / 'ae.pth')
    np.savez(args.output_dir / 'scaler.npz',
             cont_mean=scaler.mean_.astype(np.float64), cont_scale=scaler.scale_.astype(np.float64),
             prop_dim=prop_dim, cont_dim=cont_dim)
    np.savez(args.output_dir / 'sem_stats.npz',
             sem_mean=sem_mean, sem_std=sem_std, sem_min=sem_min, sem_max=sem_max)
    np.save(args.output_dir / 'raw_embeddings.npy', z.astype(np.float32))
    print(f'raw latent {z.shape}, best_val={best_val:.4f}; saved ae.pth / scaler.npz / sem_stats.npz / raw_embeddings.npy')


if __name__ == '__main__':
    main()
