"""Generate trajectories from a checkpoint and score them.

Every resident in ``--data-root`` is scored and ``split.npy`` is ignored.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import GridData
from .metrics import (METRIC_BINS as BINS, WeightedFlowAccumulator, cpc_from_flows, jsd,
                      mobility_metrics, pearson_dense, weighted_histogram)
from .train import resolve_device


def load_model(checkpoint, device):
    model_config = dict(checkpoint['model_configuration'])
    model_config.setdefault('repeat_home_environment', True)
    from .model import EnvironmentAR
    model = EnvironmentAR(**model_config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


@torch.inference_mode()
def run(data, checkpoint, device, temperature=1.0, max_days=30, seed=42, batch=256,
        checkpoint_path=None, data_root=None):
    started = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    n = data.n_grids
    model = load_model(checkpoint, device)
    if model.configuration['semantic_dim'] != data.semantic_dim:
        raise ValueError(f"checkpoint semantic_dim={model.configuration['semantic_dim']} "
                         f"does not match data sem_norm dim={data.semantic_dim}")
    grid_sem = torch.from_numpy(data.semantics).to(device)
    grid_coords = torch.from_numpy(data.coords).to(device)

    homes = np.asarray(data.homes)
    offsets = np.asarray(data.offsets)
    home_users = defaultdict(list)
    for user in range(data.n_users):
        home_users[int(homes[user])].append(user)

    real_values = {k: [] for k in BINS}; real_weights = {k: [] for k in BINS}
    gen_values = {k: [] for k in BINS}; gen_weights = {k: [] for k in BINS}
    real_all, gen_all = WeightedFlowAccumulator(n, False), WeightedFlowAccumulator(n, False)
    real_move, gen_move = WeightedFlowAccumulator(n, True), WeightedFlowAccumulator(n, True)
    real_masks_slots = gen_masks_slots = real_adj = gen_adj = 0.0
    real_days = selected_users = 0
    available_counts, selected_counts, generated_parts, day_home_parts, day_idx_parts, user_homes = [], [], [], [], [], []

    for home in sorted(home_users):
        users = home_users[home]
        z_live = torch.from_numpy(np.asarray(data.semantics[home], dtype=np.float32)).to(device).view(1, -1)
        records = []
        for ui, user in enumerate(users):
            days = np.asarray(data.grids[offsets[user]:offsets[user+1]], dtype=np.int64)[:max_days]
            if days.shape[0] == 0:
                continue
            selected_users += 1
            available_counts.append(int(offsets[user+1] - offsets[user]))
            selected_counts.append(int(days.shape[0]))
            user_homes.append(home)
            for di, day in enumerate(days):
                records.append((ui, di, day))
        if not records:
            continue
        real_arr = np.stack([r[2] for r in records])
        record_users = np.asarray([r[0] for r in records], dtype=np.int32)
        real_masks = (real_arr >= 0) & (real_arr < n)
        day_counts = {ui: len(np.asarray(data.grids[offsets[users[ui]]:offsets[users[ui]+1]])[:max_days]) for ui in record_users}
        real_w = np.asarray([1.0 / max(day_counts[int(r[0])], 1) for r in records])
        real_all.add(real_arr, real_w); real_move.add(real_arr, real_w)
        real_masks_slots += float((real_masks * real_w[:, None]).sum())
        real_adj += float(((real_masks[:, :-1] & real_masks[:, 1:]) * real_w[:, None]).sum())
        real_days += len(records)
        for row, w in zip(real_arr, real_w):
            for k, v in mobility_metrics(row, data.coords, n).items():
                real_values[k].extend(v); real_weights[k].extend([w] * len(v))

        generated = []
        for lo in range(0, len(records), batch):
            hi = min(lo + batch, len(records))
            home_ids = torch.full((hi - lo,), home, device=device, dtype=torch.long)
            mask = torch.as_tensor(real_masks[lo:hi], device=device)
            g = model.sample(z_live.expand(hi - lo, -1), home_ids, grid_sem, grid_coords, mask,
                             temperature=temperature).cpu().numpy()
            generated.append(g)
        gen_arr = np.concatenate(generated)
        gen_w = real_w.copy()
        gen_masks = (gen_arr >= 0) & (gen_arr < n)
        gen_all.add(gen_arr, gen_w); gen_move.add(gen_arr, gen_w)
        gen_masks_slots += float((gen_masks * gen_w[:, None]).sum())
        gen_adj += float(((gen_masks[:, :-1] & gen_masks[:, 1:]) * gen_w[:, None]).sum())
        for row, w in zip(gen_arr, gen_w):
            for k, v in mobility_metrics(row, data.coords, n).items():
                gen_values[k].extend(v); gen_weights[k].extend([w] * len(v))
        generated_parts.append(gen_arr.astype(np.int32))
        day_home_parts.append(np.asarray([home] * len(gen_arr), dtype=np.int32))
        day_idx_parts.append(np.asarray([r[1] for r in records], dtype=np.int16))

    real_od, gen_od = real_all.finalize(), gen_all.finalize()
    real_move_flows, gen_move_flows = real_move.finalize(), gen_move.finalize()
    jsd_values = {k: jsd(weighted_histogram(real_values[k], real_weights[k], BINS[k]),
                         weighted_histogram(gen_values[k], gen_weights[k], BINS[k])) for k in BINS}
    result = {
        "schema": "environment_ar_evaluation",
        "scope": {"n_grids": n, "eligible_homes": len(home_users), "eligible_users": data.n_users,
                  "users_scored": selected_users, "selected_days": real_days,
                  "available_day_max": max(available_counts) if available_counts else 0,
                  "selected_day_max": max(selected_counts) if selected_counts else 0,
                  "users_truncated_at_max_days": sum(x > max_days for x in available_counts)},
        "config": {"temperature": temperature, "seed": seed, "device": device, "max_days_per_user": max_days,
                   "primary_metrics": list(BINS) + ["CPC"],
                   "generation_policy": "autoregression over observed slots using environment features; "
                                        "generated history only; exact given validity mask; no personal latent"},
        "day_availability": {"available_day_mean": float(np.mean(available_counts)) if available_counts else 0.0,
                             "selected_day_mean": float(np.mean(selected_counts)) if selected_counts else 0.0},
        "mask_matching": {"real_valid_slot_mass": real_masks_slots, "generated_valid_slot_mass": gen_masks_slots,
                          "real_valid_adjacent_mass": real_adj, "generated_valid_adjacent_mass": gen_adj,
                          "valid_slot_match_exact": abs(real_masks_slots - gen_masks_slots) < 1e-8,
                          "valid_adjacent_match_exact": abs(real_adj - gen_adj) < 1e-8},
        "jsd_multinomial": jsd_values,
        "cpc": cpc_from_flows(real_move_flows, gen_move_flows),
        "od": {"pearson_dense": pearson_dense(real_od, gen_od, n), "self_transitions_included": True,
               "transition_definition": "all valid hourly adjacent transitions, including self loops",
               "real_effective_transition_mass": float(sum(real_od.values())),
               "generated_effective_transition_mass": float(sum(gen_od.values()))},
        "provenance": {"entrypoint": "environment_ar/evaluate.py",
                       "generator_checkpoint": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
                       "data_root": str(Path(data_root).resolve()) if data_root else None},
        "runtime_seconds": time.perf_counter() - started,
    }
    return result, {"GRIDS": np.concatenate(generated_parts) if generated_parts else np.empty((0, 24), np.int32),
                    "HOMES": np.concatenate(day_home_parts) if day_home_parts else np.empty(0, np.int32),
                    "DAY_INDEX": np.concatenate(day_idx_parts) if day_idx_parts else np.empty(0, np.int16),
                    "USER_HOMES": np.asarray(user_homes, dtype=np.int32),
                    "SELECTED_DAY_COUNTS": np.asarray(selected_counts, dtype=np.int16)}


def main():
    parser = argparse.ArgumentParser(description='Generate trajectories with a trained checkpoint and score them.')
    parser.add_argument('--data-root', type=Path, required=True, help='Evaluation population directory (see data.py)')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-days', type=int, default=30)
    parser.add_argument('--batch', type=int, default=256, help='Days generated per forward pass')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error(f'--temperature must be positive; got {args.temperature}')
    if min(args.max_days, args.batch, args.threads) < 1:
        parser.error('--max-days, --batch, and --threads must be positive')
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    if (args.output_dir / 'evaluation.json').exists():
        parser.error(f'{args.output_dir/"evaluation.json"} already exists; choose a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('model_type') != 'environment_ar_v1':
        parser.error(f"unsupported checkpoint model_type: {checkpoint.get('model_type')!r}")
    data = GridData(args.data_root)
    result, generated = run(data, checkpoint, device, temperature=args.temperature, max_days=args.max_days,
                            seed=args.seed, batch=args.batch, checkpoint_path=args.checkpoint, data_root=args.data_root)
    if not (result['mask_matching']['valid_slot_match_exact'] and result['mask_matching']['valid_adjacent_match_exact']):
        raise RuntimeError('generated observation mask does not match the requested mask')
    (args.output_dir / 'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=True), encoding='utf-8')
    np.savez_compressed(args.output_dir / 'generated.npz', **generated)
    print(json.dumps({'stage': 'evaluation', 'output': str(args.output_dir),
                      'cpc': result['cpc']['cpc_move_only'],
                      'jsd': {k: result['jsd_multinomial'][k] for k in BINS}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
