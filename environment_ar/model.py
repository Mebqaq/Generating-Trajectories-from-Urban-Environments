"""Autoregressive next-location model over static environment embeddings."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def geometry(delta):
    """5 relative-geometry features per planar displacement (km).

    Signed log offsets (2), log distance (1), and bounded direction (2).
    """
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    return torch.cat((delta.sign() * delta.abs().log1p() / math.log(101.),
                      distance.log1p() / math.log(101.), delta / (1 + distance)), dim=-1)


class EnvironmentAR(nn.Module):
    """Home-initialised recurrent generator over observed hourly slots.

    The home embedding initialises the hidden state and is fed to every step
    only when ``repeat_home_environment`` is true.
    """

    def __init__(self, semantic_dim=8, hidden_dim=96, score_dim=32, repeat_home_environment=False):
        super().__init__()
        self.configuration = dict(semantic_dim=semantic_dim, hidden_dim=hidden_dim, score_dim=score_dim)
        self.initial = nn.Sequential(nn.Linear(semantic_dim, hidden_dim), nn.Tanh())
        self.recurrent = nn.GRUCell(2 * semantic_dim + 13, hidden_dim)
        self.keys = nn.Sequential(nn.Linear(semantic_dim, 64), nn.SiLU(), nn.Linear(64, score_dim))
        self.bias = nn.Linear(semantic_dim, 1)
        self.query = nn.Linear(hidden_dim, score_dim)
        self.spatial = nn.Linear(hidden_dim, 10)
        self.special = nn.Linear(hidden_dim, 2)
        self.decay = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.spatial.weight); nn.init.zeros_(self.spatial.bias)
        nn.init.zeros_(self.special.weight)
        with torch.no_grad(): self.special.bias.copy_(torch.tensor([1., 2.]))
        nn.init.zeros_(self.decay.weight); nn.init.constant_(self.decay.bias, -1.)
        self.repeat_home_environment = repeat_home_environment
        if not repeat_home_environment:
            # Drop the home-input columns, keeping the same initialisation for
            # every shared parameter.
            self.recurrent.input_size -= semantic_dim
            self.recurrent.weight_ih = nn.Parameter(self.recurrent.weight_ih[:, semantic_dim:].detach().clone())
            self.configuration['repeat_home_environment'] = False

    def setup(self, live, home, semantics, coords):
        """Precompute per-location and per-home quantities shared across hours."""
        return {'live': live, 'home': home, 'semantics': semantics, 'coords': coords,
                'home_coord': coords[home], 'home_geometry': geometry(coords[None] - coords[home, None]),
                'keys': self.keys(semantics), 'bias': self.bias(semantics).squeeze(-1)}

    def step(self, hidden, previous, last_hour, hour, active, context):
        """Advance one hour and score every candidate location.

        ``active`` is this hour's observation mask; ``previous``/``last_hour``
        hold the most recent observed location and hour.
        """
        live, coords = context['live'], context['coords']
        b = len(previous)
        phase = live.new_full((b, 1), float(hour) * 2 * math.pi / 24)
        time = torch.cat((phase.sin(), phase.cos(), (2 * phase).sin(), (2 * phase).cos()), -1)
        first = last_hour < 0
        gap = (hour - last_hour).to(live.dtype)[:, None] / 24
        home_features = (live,) if self.repeat_home_environment else ()
        features = torch.cat(home_features + (context['semantics'][previous],
                              geometry(coords[previous] - context['home_coord']), time, gap,
                              first.to(live.dtype)[:, None], active.to(live.dtype)[:, None],
                              (previous == context['home']).to(live.dtype)[:, None]), -1)
        hidden = self.recurrent(features, hidden)
        logits = self.query(hidden) @ context['keys'].T / math.sqrt(context['keys'].shape[-1]) + context['bias']
        relative = geometry(coords[None] - coords[previous, None])
        coefficients = self.spatial(hidden)
        logits = logits + torch.einsum('bnf,bf->bn', context['home_geometry'], coefficients[:, :5])
        logits = logits + torch.einsum('bnf,bf->bn', relative, coefficients[:, 5:])
        decay = F.softplus(self.decay(hidden))
        logits = logits - decay[:, :1] * context['home_geometry'][:, :, 2] * math.log(101.)
        logits = logits - decay[:, 1:] * relative[:, :, 2] * math.log(101.)
        bonuses = self.special(hidden)
        logits = logits.scatter_add(1, previous[:, None], bonuses[:, :1])
        logits = logits.scatter_add(1, context['home'][:, None], bonuses[:, 1:])
        return logits, hidden

    def teacher_logits(self, rows, valid, live, home, semantics, coords):
        """Reference (stepwise) teacher-forced logits over a full day."""
        context = self.setup(live, home, semantics, coords)
        hidden = self.initial(live)
        previous = home
        last_hour = torch.full_like(home, -1)
        output = []
        for hour in range(rows.shape[1]):
            logits, hidden = self.step(hidden, previous, last_hour, hour, valid[:, hour], context)
            output.append(logits)
            previous = torch.where(valid[:, hour], rows[:, hour], previous)
            last_hour = torch.where(valid[:, hour], hour, last_hour)
        return torch.stack(output, 1)

    @torch.inference_mode()
    def sample(self, live, home, semantics, coords, valid, temperature=1., uniforms=None):
        """Free-running generation from the observation mask.

        The first observed slot is sampled, not fixed to home. Slots with
        ``valid == False`` are emitted as ``len(coords)`` (the out-of-range
        sentinel) and leave the previous location unchanged.
        """
        context = self.setup(live, home, semantics, coords)
        hidden = self.initial(live)
        previous = home
        last_hour = torch.full_like(home, -1)
        b, t = valid.shape
        if uniforms is None: uniforms = torch.rand(b, t, device=live.device)
        output = []
        for hour in range(t):
            logits, hidden = self.step(hidden, previous, last_hour, hour, valid[:, hour], context)
            cdf = (logits / temperature).softmax(-1).cumsum(-1)
            cdf[:, -1] = 1.
            chosen = torch.searchsorted(cdf.contiguous(), uniforms[:, hour, None].contiguous()).squeeze(-1).clamp_max(len(coords)-1)
            output.append(torch.where(valid[:, hour], chosen, len(coords)))
            previous = torch.where(valid[:, hour], chosen, previous)
            last_hour = torch.where(valid[:, hour], hour, last_hour)
        return torch.stack(output, 1)


def masked_nll(logits, rows, valid):
    """Mean negative log-likelihood over valid hourly observations."""
    target = torch.where(valid, rows, 0)
    values = -logits.log_softmax(-1).gather(-1, target[:, :, None]).squeeze(-1)
    return (values * valid).sum() / valid.sum().clamp_min(1)
