"""Q-network: a small MLP over the feature vector.

Why plain rather than dueling
-----------------------------
An earlier version used a dueling head (V + A - mean(A)). It was removed for a
concrete, measured reason: cross-entropy imitation constrains only the
*differences* between action values, so with a dueling decomposition the shared
value stream is free to drift without bound. After imitation warm starting, the
dueling network's Q-values sat around -32,000 with spreads of hundreds of
thousands, and it played worse than an untrained network. A plain MLP has a
single output path, so the same warm start produces Q-values at a sane scale
(~±30). Since the Q-values become bootstrap targets for TD learning, an
unbounded scale is not cosmetic -- it destroys training.

The input is a compact feature vector, not a board image, so a small MLP is both
sufficient and fast enough that CPU training is limited by simulation rather
than by the optimiser.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .engine import COLS
from .features import DEFAULT_VERSION, n_features

N_ACTIONS = 4 * COLS      # 4 rotations x 10 columns, flat id = rot * COLS + col
N_INPUTS = n_features(DEFAULT_VERSION)


class QNet(nn.Module):
    """Q(s, a) as a plain MLP.

    Weights are initialised small so that early Q-values are near zero: the
    targets are bootstrapped from these outputs, and a large random initial
    scale makes early TD targets noisy for no benefit.

    ``n_features`` is the input width and must match the feature version used to
    build the inputs. A checkpoint records the width it was trained with, so a
    network can be rebuilt to match older features.
    """

    def __init__(self, n_features=N_INPUTS, n_actions=N_ACTIONS, hidden=256,
                 init=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -init, init)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


# The name used by the rest of the project.
DuelingQNet = QNet


def masked_argmax(q, mask, neg_inf=-1e9):
    """Argmax over legal actions only. ``mask`` is a bool array/tensor."""
    return int(torch.argmax(q.masked_fill(~mask, neg_inf)).item())
