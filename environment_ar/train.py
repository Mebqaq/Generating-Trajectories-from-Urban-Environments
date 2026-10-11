"""Train the generator in one fit/validation run with early stopping."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import GridData
from .full_model import FullTrainingAR
from .model import masked_nll


def dump(path, value):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def save(path, value):
    temp = path.with_name(path.name+'.tmp')
    torch.save(value, temp)
    temp.replace(path)


def log(**data):
    print(json.dumps(data), flush=True)


def resolve_device(requested):
    if requested == 'cpu':
        return 'cpu'
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit('--device cuda requested but CUDA is unavailable')
        return 'cuda'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def copy_state(model):
    """Detached deep copy of the model weights on CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.inference_mode()
def validation(model, source, fixed, selected, batch, device):
    """Masked-NLL on all available validation days; used for early stopping."""
    model.eval()
    total, tokens = 0., 0
    for lo in range(0, len(selected), batch):
        values = source.batch(selected[lo:lo+batch], fixed, device)
        loss = masked_nll(model.teacher_logits(*values), values[0], values[1])
        count = int(values[1].sum())
        total += float(loss)*count; tokens += count
    return total/tokens


def train_run(args, source, groups, split_info, device):
    out = args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    fixed = source.fixed(device)
    model = FullTrainingAR(semantic_dim=args.semantic_dim).to(device)
    model.cache_geometry(fixed[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = dict(variant='main', epochs=args.epochs, patience=args.patience, batch_days=args.batch,
                  days_per_resident_per_epoch=0, fit_users=len(groups['fit']), dev_users=len(groups['dev']),
                  available_train_days=source.count_days(groups['fit']), model=model.configuration,
                  learning_rate=args.lr, weight_decay=args.weight_decay, seed=args.seed, phase='train',
                  checkpoint='lowest-validation-NLL epoch of a single fit/validation run',
                  early_stopping='strictly lower validation NLL; stop after patience epochs without improvement',
                  loss='masked full-vocabulary next-observation NLL', inference_temperature=1.,
                  teacher_mobility_latent=False, city_id_parameters=False,
                  validation_days_per_resident=0, validation_seed=None,
                  data_root=str(source.root), held_out_used=False, split=split_info)
    dump(out/'configuration.json', config)
    first, offset, total, tokens, elapsed = 0, 0, 0., 0, 0.
    history = []
    best_val, best_epoch, bad, best_state, stopped = float('inf'), 0, 0, None, False
    resume = out/'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=True)
        if state['configuration'] != config:
            raise ValueError('Resume configuration differs from this run; use a new --run-dir or remove resume.pt')
        if state.get('stopped'):
            raise ValueError('This run already finished (early stopping or maximum epochs); '
                             'it cannot be resumed. Use a new --run-dir to train again.')
        model.load_state_dict(state['state_dict']); optimizer.load_state_dict(state['optimizer'])
        first, offset = state['epoch'], state['offset']
        total, tokens, elapsed = state['total'], state['tokens'], state['seconds']
        history = state['history']
        best_val, best_epoch, bad, best_state = state['best_val'], state['best_epoch'], state['bad'], state['best_state']
    dev = source.selected(groups['dev'], args.seed, 0)  # all available validation days
    started = time.perf_counter()

    def persist(epoch, offset, total, tokens, done=False):
        save(resume, dict(configuration=config, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
             epoch=epoch, offset=offset, total=total, tokens=tokens,
             seconds=elapsed+time.perf_counter()-started, history=history,
             best_val=best_val, best_epoch=best_epoch, bad=bad, best_state=best_state, stopped=done))

    log(stage='train', configuration=config, resumed_epoch=first, resumed_offset=offset)
    stop_reason = None
    for epoch in range(first, args.epochs):
        model.train()
        selected = source.selected(groups['fit'], 5042+epoch, 0)
        begin = offset if epoch == first else 0
        if not begin:
            total, tokens = 0., 0
        saved_at = time.perf_counter()
        for lo in range(begin, len(selected), args.batch):
            values = source.batch(selected[lo:lo+args.batch], fixed, device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_nll(model.teacher_logits(*values), values[0], values[1])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite gradient')
            optimizer.step()
            count = int(values[1].sum()); total += float(loss.detach())*count; tokens += count
            if (lo//args.batch+1) % 100 == 0:
                progress = dict(stage='train', epoch=epoch+1, epochs=args.epochs,
                    days_done=min(lo+args.batch, len(selected)), days_per_epoch=len(selected),
                    train_nll=total/tokens, seconds=elapsed+time.perf_counter()-started)
                dump(args.run_dir/'progress.json', progress); log(**progress)
            if time.perf_counter()-saved_at > 300:
                persist(epoch, min(lo+args.batch, len(selected)), total, tokens)
                saved_at = time.perf_counter()
        dump(args.run_dir/'progress.json', dict(stage='validation', epoch=epoch+1, epochs=args.epochs))
        val = validation(model, source, fixed, dev, args.batch, device)
        if not math.isfinite(val):
            raise FloatingPointError('Non-finite validation loss')
        improved = val < best_val
        if improved:
            best_val, best_epoch, bad = val, epoch+1, 0
            best_state = copy_state(model)
        else:
            bad += 1
        row = dict(epoch=epoch+1, train_nll=total/tokens, sampled_days=len(selected), validation_nll=val,
                   best_epoch=best_epoch, best_val_loss=best_val, patience_used=bad)
        row['seconds'] = elapsed+time.perf_counter()-started
        history.append(row)
        finished = bad >= args.patience or epoch+1 >= args.epochs
        persist(epoch+1, 0, 0., 0, done=finished)
        dump(out/'history.json', history); log(stage='train_epoch', **row)
        if bad >= args.patience:
            stop_reason = 'early_stopping'
            break
    else:
        stop_reason = 'max_epochs'

    if best_state is None:
        best_state, best_epoch = copy_state(model), len(history)
        best_val = history[-1]['validation_nll']
    summary = dict(stage='train', epochs_run=len(history), epochs=args.epochs, patience=args.patience,
                   best_epoch=best_epoch, best_val_loss=best_val, stop_reason=stop_reason,
                   seconds=elapsed+time.perf_counter()-started, fit_users=config['fit_users'],
                   dev_users=config['dev_users'], available_train_days=config['available_train_days'])
    checkpoint = out/'environment_ar.pth'
    save(checkpoint, dict(state_dict=best_state, model_type='environment_ar_v1',
         model_configuration=model.configuration, training_configuration=config,
         epoch=best_epoch, best_val_loss=best_val, stop_reason=stop_reason))
    summary['checkpoint'] = str(checkpoint.resolve())
    persist(len(history), 0, 0., 0, done=True)
    dump(out/'complete.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description='Train the environment AR main model (single fit/validation run, early stopping).')
    p.add_argument('--data-root', type=Path, required=True, help='Directory with grids/homes/offsets .npy and grid.npz (split.npy optional)')
    p.add_argument('--run-dir', type=Path, required=True, help='Output directory for the checkpoint, history, and resumable state')
    p.add_argument('--epochs', type=int, default=30, help='Maximum number of training epochs')
    p.add_argument('--patience', type=int, default=5, help='Epochs without a strictly lower validation NLL before stopping')
    p.add_argument('--batch', type=int, default=512, help='Days per optimisation batch')
    p.add_argument('--threads', type=int, default=4, help='CPU threads (used when running on CPU)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--semantic-dim', type=int, default=8)
    p.add_argument('--fit-code', type=int, default=0)
    p.add_argument('--dev-code', type=int, default=1)
    p.add_argument('--held-code', type=int, default=2)
    p.add_argument('--validation-ratio', type=float, default=0.1,
                   help='Validation resident fraction when split.npy is absent')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch, args.threads) < 1:
        p.error('--epochs, --patience, --batch, and --threads must be positive')
    if not 0 < args.validation_ratio < 1:
        p.error(f'--validation-ratio must be in (0, 1); got {args.validation_ratio}')
    device = resolve_device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if device == 'cuda' and hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    source = GridData(args.data_root)
    if source.semantic_dim != args.semantic_dim:
        p.error(f'grid.npz sem_norm has {source.semantic_dim} dims but --semantic-dim is {args.semantic_dim}')
    groups, split_info = source.make_groups(fit=args.fit_code, dev=args.dev_code, held=args.held_code,
                                            validation_ratio=args.validation_ratio, seed=args.seed)
    log(stage='data', data_root=str(source.root), n_grids=source.n_grids, n_users=source.n_users, n_days=source.n_days,
        fit_users=len(groups['fit']), dev_users=len(groups['dev']), held_users=len(groups['held']),
        split=split_info, device=device)
    summary = train_run(args, source, groups, split_info, device)
    dump(args.run_dir/'completion.json', dict(status='trained', run=summary))
    log(stage='training_complete', result=summary)


if __name__ == '__main__':
    main()
