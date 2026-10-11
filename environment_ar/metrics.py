"""Mobility JSD, CPC, and OD metrics for the trajectory evaluator."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


METRIC_BINS = {
    "Displacement": np.linspace(0, 80, 50),
    "Radius": np.linspace(0, 80, 50),
    "Duration": np.arange(1, 27),
    "DailyLoc": np.arange(1, 27),
}


def mobility_metrics(grid: np.ndarray, coords: np.ndarray, num_grids: int) -> dict[str, list[float]]:
    """Four per-day mobility statistics for one day's hourly locations.

    Radius is the standard radius of gyration (km) about the day's mean centre.
    Out-of-range ids are dropped; a day with no valid hour yields empty lists.
    """
    valid = [int(value) for value in np.asarray(grid, dtype=np.int64) if 0 <= int(value) < num_grids]
    if not valid:
        return {name: [] for name in METRIC_BINS}

    observed = np.asarray(coords[valid], dtype=np.float64)
    centre = observed.mean(axis=0)
    offsets = observed - centre
    radius = float(np.sqrt(np.mean(np.sum(offsets * offsets, axis=1))))
    radii = [radius]
    durations: list[int] = []
    displacements: list[float] = []
    stay = 1
    for origin, destination in zip(valid[:-1], valid[1:]):
        if origin == destination:
            stay += 1
        else:
            durations.append(stay)
            stay = 1
            displacements.append(float(np.linalg.norm(coords[origin] - coords[destination])))
    durations.append(stay)
    return {
        "Displacement": displacements,
        "Radius": radii,
        "Duration": durations,
        "DailyLoc": [len(set(valid))],
    }


def weighted_histogram(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None,
    bins: np.ndarray,
) -> np.ndarray:
    """Smoothed density histogram suitable for JSD."""
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0:
        fallback = np.zeros(len(bins) - 1, dtype=np.float64)
        fallback[0] = 1.0
        return fallback + 1e-10

    weights_array = None if weights is None else np.asarray(weights, dtype=np.float64)
    histogram, _ = np.histogram(values_array, bins=bins, weights=weights_array, density=True)
    return histogram.astype(np.float64) + 1e-10


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence with natural logarithms."""
    p_array = np.asarray(p, dtype=np.float64)
    q_array = np.asarray(q, dtype=np.float64)
    p_array = p_array / p_array.sum()
    q_array = q_array / q_array.sum()
    midpoint = 0.5 * (p_array + q_array)
    return float(
        0.5
        * np.sum(
            p_array * np.log(np.clip(p_array, 1e-12, None) / np.clip(midpoint, 1e-12, None))
            + q_array * np.log(np.clip(q_array, 1e-12, None) / np.clip(midpoint, 1e-12, None))
        )
    )


class WeightedFlowAccumulator:
    """Weighted hourly OD flow counts keyed by ``origin * n + destination``.

    ``exclude_self`` drops self-transitions, giving the move-only view used by
    CPC; invalid endpoints are dropped.
    """

    def __init__(self, num_grids: int, exclude_self: bool):
        self.n = int(num_grids)
        self.exclude_self = bool(exclude_self)
        self.parts, self.weights, self.part_n, self.counts = [], [], 0, {}
        self.raw_valid = 0

    def _flush(self):
        if not self.parts:
            return
        keys = np.concatenate(self.parts)
        weights = np.concatenate(self.weights)
        unique, inverse = np.unique(keys, return_inverse=True)
        values = np.bincount(inverse, weights=weights)
        for key, value in zip(unique.tolist(), values.tolist()):
            key = int(key); self.counts[key] = self.counts.get(key, 0.0) + float(value)
        self.parts, self.weights, self.part_n = [], [], 0

    def add(self, trajectories, weights):
        arr = np.asarray(trajectories, dtype=np.int64)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        if arr.size == 0: return
        w = np.asarray(weights, dtype=np.float64)
        a, b = arr[:, :-1], arr[:, 1:]
        valid = ((a >= 0) & (a < self.n) & (b >= 0) & (b < self.n))
        if self.exclude_self: valid &= a != b
        if not np.any(valid): return
        keys = (a[valid] * self.n + b[valid]).astype(np.int64)
        row_w = np.repeat(w, arr.shape[1] - 1).reshape(arr.shape[0], -1)[valid]
        self.parts.append(keys); self.weights.append(row_w); self.part_n += int(keys.size)
        self.raw_valid += int(keys.size)
        if self.part_n >= 1_000_000: self._flush()

    def finalize(self):
        self._flush(); return self.counts


def cpc_from_flows(real: dict, generated: dict) -> dict:
    """Common Part of Commuters from two weighted OD flow maps."""
    real_mass, gen_mass = sum(real.values()), sum(generated.values())
    keys = set(real) | set(generated)
    overlap = sum(min(real.get(k, 0.0), generated.get(k, 0.0)) for k in keys)
    cpc = 2 * overlap / (real_mass + gen_mass) if real_mass + gen_mass else 0.
    ceiling = 2 * min(real_mass, gen_mass) / (real_mass + gen_mass) if real_mass + gen_mass else 1.
    return {"cpc_move_only": float(cpc), "real_move_only_flow_mass": float(real_mass),
            "generated_move_only_flow_mass": float(gen_mass), "move_only_flow_overlap": float(overlap),
            "total_flow_ratio": float(gen_mass / real_mass) if real_mass else (1.0 if gen_mass == 0 else float("nan")),
            "cpc_theoretical_max": float(ceiling),
            "cpc_relative_to_theoretical_max": float(cpc / ceiling) if ceiling else float("nan"),
            "definition": "2 * sum of min over raw move-only weighted OD flows; self transitions excluded"}


def pearson_dense(real: dict, generated: dict, n: int) -> float:
    """Pearson correlation of two dense OD matrices (all transitions, incl. self)."""
    keys = set(real) | set(generated)
    real_mass, gen_mass = sum(real.values()), sum(generated.values())
    cells = float(n * n)
    dot = sum(real.get(k, 0.0) * generated.get(k, 0.0) for k in keys)
    sr = sum(v * v for v in real.values())
    sg = sum(v * v for v in generated.values())
    den = np.sqrt(max(sr - real_mass * real_mass / cells, 0) * max(sg - gen_mass * gen_mass / cells, 0))
    return float((dot - real_mass * gen_mass / cells) / den) if den else float("nan")
