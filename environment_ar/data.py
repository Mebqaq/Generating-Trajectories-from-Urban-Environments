"""Read a training or evaluation population from a NumPy data root."""
from pathlib import Path

import numpy as np
import torch


class GridData:
    def __init__(self, root, seq_len=24):
        self.root = Path(root)
        self.seq_len = int(seq_len)
        self.grids = np.load(self.root / 'grids.npy', mmap_mode='r')
        self.homes = np.load(self.root / 'homes.npy')
        self.offsets = np.load(self.root / 'offsets.npy')
        split_path = self.root / 'split.npy'
        self.split = np.load(split_path) if split_path.exists() else None
        with np.load(self.root / 'grid.npz') as grid:
            self.coords = np.asarray(grid['coords_km'], dtype=np.float32)
            self.semantics = np.asarray(grid['sem_norm'], dtype=np.float32)
        self._valid_days = None
        self._check()

    def _check(self):
        if self.grids.ndim != 2 or self.grids.shape[1] != self.seq_len:
            raise ValueError(f'grids.npy must be (n_days, {self.seq_len}); got {self.grids.shape}')
        if self.grids.shape[0] == 0:
            raise ValueError('grids.npy is empty')
        if not np.issubdtype(self.grids.dtype, np.integer):
            raise ValueError(f'grids.npy must be an integer array; got {self.grids.dtype}')
        if self.homes.ndim != 1 or self.homes.shape[0] == 0 or not np.issubdtype(self.homes.dtype, np.integer):
            raise ValueError('homes.npy must be a non-empty 1-D integer array')
        if self.offsets.ndim != 1 or not np.issubdtype(self.offsets.dtype, np.integer):
            raise ValueError('offsets.npy must be a 1-D integer array')
        if self.offsets.shape[0] != self.homes.shape[0] + 1:
            raise ValueError('offsets.npy must have length n_users + 1')
        if self.offsets[0] != 0 or self.offsets[-1] != self.grids.shape[0]:
            raise ValueError('offsets.npy must span [0, n_days]')
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError('offsets.npy must be non-decreasing')
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(f'coords_km must be (n_grids, 2); got {self.coords.shape}')
        if self.semantics.ndim != 2 or self.semantics.shape[0] != self.coords.shape[0]:
            raise ValueError('sem_norm must be (n_grids, S) matching coords_km')
        if not np.isfinite(self.coords).all() or not np.isfinite(self.semantics).all():
            raise ValueError('grid.npz contains non-finite values')
        if self.homes.min() < 0 or self.homes.max() >= self.coords.shape[0]:
            raise ValueError('homes.npy contains an out-of-range location id')
        if self.split is not None:
            if self.split.ndim != 1 or self.split.shape[0] != self.homes.shape[0]:
                raise ValueError('split.npy must be one code per resident')
            if not np.issubdtype(self.split.dtype, np.integer):
                raise ValueError(f'split.npy must be an integer array; got {self.split.dtype}')

    @property
    def n_grids(self):
        return int(self.coords.shape[0])

    @property
    def n_users(self):
        return int(self.homes.shape[0])

    @property
    def n_days(self):
        return int(self.grids.shape[0])

    @property
    def semantic_dim(self):
        return int(self.semantics.shape[1])

    def valid_days_per_resident(self):
        """Number of days each resident has with at least one observed hour."""
        if self._valid_days is None:
            day_valid = ((self.grids >= 0) & (self.grids < self.n_grids)).any(axis=1)
            cumulative = np.concatenate([[0], np.cumsum(day_valid)])
            self._valid_days = (cumulative[self.offsets[1:]] - cumulative[self.offsets[:-1]]).astype(np.int64)
        return self._valid_days

    def make_groups(self, fit=0, dev=1, held=2, validation_ratio=0.1, seed=42):
        """Return ``(groups, split_info)`` for training and validation residents.

        Uses ``split.npy`` when present, otherwise draws a random split from the
        non-empty residents; all days of a resident stay in one group.
        """
        fit, dev, held = int(fit), int(dev), int(held)
        if len({fit, dev, held}) != 3:
            raise ValueError(f'fit/dev/held codes must be distinct; got {fit}/{dev}/{held}')
        valid_days = self.valid_days_per_resident()
        if self.split is not None:
            codes = np.unique(self.split)
            if not np.all(np.isin(codes, [fit, dev, held])):
                raise ValueError(f'split.npy contains codes {codes.tolist()} outside {{{fit}, {dev}, {held}}}')
            groups = {'fit': np.flatnonzero(self.split == fit),
                      'dev': np.flatnonzero(self.split == dev),
                      'held': np.flatnonzero(self.split == held)}
            info = {'source': 'split.npy', 'fit_code': fit, 'dev_code': dev, 'held_code': held,
                    'validation_ratio': None, 'seed': None}
        else:
            if not 0 < validation_ratio < 1:
                raise ValueError(f'--validation-ratio must be in (0, 1); got {validation_ratio}')
            eligible = np.flatnonzero(valid_days > 0)
            if eligible.size < 2:
                raise ValueError('a random split needs at least two residents with an observed day')
            order = np.random.default_rng(seed).permutation(eligible)
            n_val = min(max(int(round(eligible.size * validation_ratio)), 1), eligible.size - 1)
            groups = {'fit': np.sort(order[n_val:]), 'dev': np.sort(order[:n_val]),
                      'held': np.zeros(0, dtype=np.int64)}
            info = {'source': 'random_resident_split', 'fit_code': None, 'dev_code': None, 'held_code': None,
                    'validation_ratio': float(validation_ratio), 'seed': int(seed)}
        for name in ('fit', 'dev'):
            members, count = groups[name], int(valid_days[groups[name]].sum()) if groups[name].size else 0
            if members.size == 0 or count == 0:
                raise ValueError(f'{name} group has no resident with a valid observed day')
        return groups, info

    def selected(self, group, seed, days=0):
        """Day indices for a resident group; ``days=0`` returns every day.

        A positive ``days`` draws that many days per resident, with replacement
        only when the resident has fewer days.
        """
        rng = np.random.default_rng(seed)
        chunks = []
        for user in group:
            lo, hi = self.offsets[user:user+2]
            choices = np.arange(lo, hi)
            chunks.append(rng.choice(choices, days, replace=len(choices) < days) if days else choices)
        return rng.permutation(np.concatenate(chunks))

    def count_days(self, group):
        return int(np.diff(self.offsets)[group].sum())

    def fixed(self, device):
        return torch.from_numpy(self.coords).to(device), torch.from_numpy(self.semantics).to(device)

    def batch(self, indices, fixed, device):
        """Build one training/validation batch of days.

        Returns ``(rows, valid, home_semantics, home, semantics, coords)`` where
        ``valid`` is the observation mask derived from the location ids.
        """
        users = np.searchsorted(self.offsets[1:], indices, side='right')
        home = torch.as_tensor(np.asarray(self.homes[users], dtype=np.int64), device=device)
        rows = torch.as_tensor(np.asarray(self.grids[indices], dtype=np.int64), device=device)
        coords, semantics = fixed
        valid = (rows >= 0) & (rows < len(coords))
        return rows, valid, semantics[home], home, semantics, coords
