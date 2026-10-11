"""Batched teacher forcing for the same EnvironmentAR parameters."""
import math
import torch
from torch.nn import functional as F
from .model import EnvironmentAR, geometry


class FullTrainingAR(EnvironmentAR):
    @torch.no_grad()
    def cache_geometry(self, coords):
        """Precompute pairwise relative-geometry features for all locations."""
        self.geometry_bank = torch.empty((len(coords), len(coords), 5), device=coords.device, dtype=coords.dtype)
        for lo in range(0, len(coords), 128):
            self.geometry_bank[lo:lo+128] = geometry(coords[None]-coords[lo:lo+128, None])

    def teacher_logits(self, rows, valid, live, home, semantics, coords):
        b, t = rows.shape
        hours = torch.arange(t, device=rows.device)[None].expand(b, -1)
        observed = torch.where(valid, hours, -1).cummax(1).values
        last_hour = torch.cat([torch.full_like(home[:, None], -1), observed[:, :-1]], 1)
        previous = torch.cat([home[:, None], rows], 1).gather(1, last_hour+1)
        phase = hours.to(live.dtype)[..., None]*(2*math.pi/24)
        time = torch.cat([phase.sin(), phase.cos(), (2*phase).sin(), (2*phase).cos()], -1)
        home_features = [live[:, None].expand(-1, t, -1)] if self.repeat_home_environment else []
        features = torch.cat(home_features + [
            semantics[previous],
            geometry(coords[previous]-coords[home, None]), time,
            (hours-last_hour).to(live.dtype)[..., None]/24,
            (last_hour < 0).to(live.dtype)[..., None], valid.to(live.dtype)[..., None],
            (previous == home[:, None]).to(live.dtype)[..., None]], -1)
        hidden = self.initial(live)
        history = []
        for hour in range(t):
            hidden = self.recurrent(features[:, hour], hidden)
            history.append(hidden)
        hidden = torch.stack(history, 1)
        keys = self.keys(semantics)
        logits = self.query(hidden) @ keys.T / math.sqrt(keys.shape[-1]) + self.bias(semantics).squeeze(-1)
        coefficients = self.spatial(hidden)
        home_geometry = self.geometry_bank[home]
        relative = self.geometry_bank[previous]
        logits = logits + torch.einsum('bnf,btf->btn', home_geometry, coefficients[..., :5])
        logits = logits + torch.einsum('btnf,btf->btn', relative, coefficients[..., 5:])
        decay = F.softplus(self.decay(hidden))
        logits = logits - decay[..., :1]*home_geometry[:, None, :, 2]*math.log(101.)
        logits = logits - decay[..., 1:]*relative[..., 2]*math.log(101.)
        bonuses = self.special(hidden)
        logits = logits.scatter_add(2, previous[..., None], bonuses[..., :1])
        return logits.scatter_add(2, home[:, None, None].expand(-1, t, 1), bonuses[..., 1:])
