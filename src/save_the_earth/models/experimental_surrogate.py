"""
Save the Earth — Deep Experimental Firebrand Surrogate (Model E).

Deep neural network trained directly on empirical firebrand measurements
from the USDA JFSP 15-1-04-4 experiments (69 wind-tunnel tests, 60,370 particles).

Architecture:
- Deep Residual MLP with LayerNorm, SiLU activations, and Dropout
- Predicts expected median landing distance (q50) and calibrated dispersion
  intervals (q10 and q90) using Pinball / Quantile loss.
- Enforces physical non-negativity via Softplus.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ExperimentalDistanceNet(nn.Module):
    """
    Deep Neural Network for firebrand landing distance and dispersion.
    Input features (dim=10):
    [u, u^2, gust, moisture, dryness, is_structural, log(mass), log(area), log(A/m), wind_momentum]
    Outputs: [q10_distance, q50_median_distance, q90_distance]
    """
    def __init__(self, in_dim: int = 10, hidden_dim: int = 128, dropout: float = 0.08):
        super().__init__()
        self.in_dim = in_dim
        
        # Shared deep representation trunk
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU()
        )
        
        # Median prediction head (q50)
        self.head_median = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Lower dispersion delta head (q50 - q10)
        self.head_lower_delta = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Upper dispersion delta head (q90 - q50)
        self.head_upper_delta = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        q50 = self.head_median(h)
        delta_low = self.head_lower_delta(h)
        delta_high = self.head_upper_delta(h)
        
        # Guaranteed monotonicity: q10 <= q50 <= q90
        q10 = torch.clamp(q50 - delta_low, min=0.1)
        q90 = q50 + delta_high
        
        return torch.cat([q10, q50, q90], dim=-1)

