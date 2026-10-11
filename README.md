# Daily Trajectory Generation from Environment Embeddings

Reference implementation of the main model in the paper's experiments: a
home-anchored autoregressive generator that produces one day of hourly
locations for an individual resident. At every hour the model scores all
spatial units from a shared function of each unit's static environment
embedding and its planar geometry relative to the resident's home and to the
previous location. The parameters hold no city or location identifiers, so one
trained model can be applied to a new city.

## Requirements

Python 3.9+ with `torch` and `numpy`; the environment autoencoder also requires
`scikit-learn`. Install with:

```bash
pip install -r requirements.txt
```

Training and evaluation run on CPU for small data and use CUDA automatically
when available. Run all commands as modules from the package root (the
directory containing `environment_ar/`).

## Data format

All inputs are plain NumPy arrays. A data root is a directory holding:

| File | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `grids.npy` | `(n_days, 24)` | integer | hourly location ids; an id outside `[0, n_grids)` marks a missing hour |
| `homes.npy` | `(n_users,)` | integer | home location id of each resident |
| `offsets.npy` | `(n_users + 1,)` | integer | `offsets[i]:offsets[i+1]` are resident `i`'s rows in `grids.npy` |
| `split.npy` | `(n_users,)` | integer | optional split code per resident (training only) |
| `grid.npz` | see below | float32 | coordinates and environment embeddings |

`grid.npz`:

- `coords_km` — `(n_grids, 2)` float32, planar coordinates in **kilometres**.
  Only coordinate differences enter the model, so any consistent planar origin
  works.
- `sem_norm` — `(n_grids, S)` float32, the environment embedding. For the paper
  model `S = 8`, produced by the frozen autoencoder below and normalised with
  the source-domain latent statistics.

A missing hour is any id outside `[0, n_grids)` (for example `-1` or
`n_grids`); the mask is `0 <= id < n_grids`. Missing hours advance the recurrent
state but leave the previous observed location unchanged.

**Splits.** `split.npy` is optional. When present it holds one integer code per
resident, drawn from three possible codes `{0, 1, 2}` by default:

- `0` = fit residents, which update the weights;
- `1` = validation residents, whose loss is only monitored;
- `2` = test residents, which are excluded from both.

The codes can be changed with `--fit-code`, `--dev-code`, and `--held-code`
(they must stay distinct), and the file is used as given.

When `split.npy` is absent, `train.py` draws a resident-level random split from
the non-empty residents (those with at least one observed day):
`--validation-ratio` (default `0.1`) of them become validation residents and the
rest training residents, using `--seed` (default `42`). All days of a resident
stay in one group, both groups are non-empty, and nothing is written back to the
data directory. Either way the split source, ratio, and seed are recorded in the
run's `configuration.json`.

Every city reuses the source encoder, scaling, and latent statistics, so its
embeddings share the same 8-dimensional scale. Coordinates are supplied
separately for each city.

## Environment encoder

For each location the paper builds a 33-dimensional static feature vector:

- `prop_dim = 11` land-use proportions (row-normalised to sum to one), then
- `cont_dim = 22` continuous features: nightlight, 19 POI counts, and 2
  road-length measures (log1p-transformed and standardised).

Train the autoencoder on the source city (33 → 128 → 8, GELU + LayerNorm, with
separate decoder heads for proportions (KL) and continuous values (MSE)):

```bash
python -m environment_embedding.train_autoencoder \
    --features source_features.npy \
    --output-dir ae_bundle \
    --prop-dim 11 --latent-dim 8
```

`source_features.npy` is a `(n_locations, 33)` float array with the 11
proportion columns first. The run writes to `ae_bundle/`: `ae.pth`, `scaler.npz`
(`cont_mean`, `cont_scale`, `prop_dim`, `cont_dim`), `sem_stats.npz`
(`sem_mean`, `sem_std`, `sem_min`, `sem_max`), and `raw_embeddings.npy`.

Encode a new city with the frozen source bundle, which applies the same
transforms, source scaler, and source latent normalisation:

```bash
python -m environment_embedding.encode \
    --features new_city_features.npy \
    --bundle ae_bundle \
    --output new_city_sem_norm.npy
```

Use `new_city_sem_norm.npy` as `sem_norm` in that city's `grid.npz`. Fill any
feature channels missing for a partial city with the source statistics before
encoding.

## Main model

`environment_ar/model.py` defines `EnvironmentAR` (8-dim semantics, 96-dim
hidden state, 32-dim candidate scores). The residential embedding initialises
the hidden state and is not repeated at each recurrent step. Training uses
batched teacher forcing (`full_model.py`) and the masked next-observation
negative log-likelihood over observed hours only.

### Train

```bash
python -m environment_ar.train \
    --data-root shanghai_data \
    --run-dir runs/main \
    --epochs 30 --patience 5 --batch 512 --threads 4 --seed 42 \
    --validation-ratio 0.1 --device auto
```

Training is a single fit/validation run with early stopping: fit residents
update the weights for at most `--epochs` epochs, and validation residents are
scored every epoch on all their available days. Improvement means a strictly
lower validation NLL; after `--patience` consecutive epochs without improvement
the run stops. Test residents are never used. `--validation-ratio` applies only
when `split.npy` is absent. Each epoch visits every fit day once. The optimizer
is AdamW with learning rate `1e-3`, weight decay `1e-5`, and gradient clipping
`1.0`; batch size counts days.

The saved `environment_ar.pth` holds the model with the lowest validation NLL.
The paper instead selected an epoch on source validation and then refit on all
source data; this public code performs the single fit/validation run only.

Outputs under `--run-dir`: `configuration.json` (including the split source,
ratio, and seed), `history.json` (per-epoch train and validation NLL),
`complete.json` (`best_epoch`, `best_val_loss`, `epochs_run`, `stop_reason`),
`progress.json`, `resume.pt`, and `environment_ar.pth`. The checkpoint is a dict
with `state_dict`, `model_type = "environment_ar_v1"`, `model_configuration`,
`training_configuration`, `epoch` (the best epoch), and `best_val_loss`.

Training resumes from `resume.pt`, restoring the best weights, best loss, and
patience counter. A finished run — stopped early or reaching `--epochs` — cannot
be resumed; use a new `--run-dir`. A resume whose configuration differs from the
current arguments raises an error.

### Generate and evaluate

```bash
python -m environment_ar.evaluate \
    --data-root shanghai_eval \
    --checkpoint runs/main/environment_ar.pth \
    --output-dir eval/shanghai \
    --temperature 1.0 --max-days 30 --seed 42
```

The evaluator generates each day's 24 hourly slots free-running. Each step
scores candidate locations from the resident's home, the environment
embeddings, the coordinates, the observation mask, and the hours generated so
far. The first observed slot is sampled, not fixed to home. It takes at most
`--max-days` days per resident and writes `evaluation.json` and `generated.npz`
(generated grids, home ids, day indices).

`--data-root` should be test or new-city data: the evaluator scores every
resident in that directory and ignores `split.npy`.

Metrics (following the paper's evaluator):

- **Displacement, Radius, Duration, DailyLoc** — per-day statistics compared by
  Jensen–Shannon divergence (lower is better). Radius is the standard radius of
  gyration (km), the root-mean-square distance from the day's mean centre.
- **CPC** — Common Part of Commuters over move-only hourly OD flows
  (`2·Σ min(real, generated) / (Σ real + Σ generated)`, self-transitions
  excluded; higher is better).

Every resident is weighted equally, and within a resident every observed day
contributes equally. Real and generated masks must match exactly.

## Quick CPU check

Use a small dataset to check training and evaluation:

```bash
python -m environment_ar.train --data-root tiny_data --run-dir tiny_run \
    --epochs 10 --patience 3 --batch 32 --device cpu
python -m environment_ar.evaluate --data-root tiny_data \
    --checkpoint tiny_run/environment_ar.pth --output-dir tiny_eval --device cpu
```

## Layout

```
environment_ar/          main generator, training, generation, metrics
  model.py               EnvironmentAR, geometry, masked NLL
  full_model.py          batched teacher forcing
  data.py                NumPy population reader
  train.py               single fit/validation training with early stopping
  metrics.py             JSD and CPC primitives
  evaluate.py            generation and evaluation entry point
environment_embedding/   static-environment autoencoder
  model.py               MultiHeadGridAutoencoder
  train_autoencoder.py   train on source features
  encode.py              encode a new city in the frozen latent space
```
