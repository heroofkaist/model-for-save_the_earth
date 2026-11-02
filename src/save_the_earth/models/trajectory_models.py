"""
Save the Earth — Neural and Physics-Informed Trajectory Models.

Implements architectures for 3D firebrand flight prediction:
1. NeuralTrajectorySurrogate (Model B):
   Deep residual MLP with dual heads predicting full 3D time-resolved positions
   (x(t), y(t), z(t)) and final landing metrics (x_land, y_land, t_flight).
2. PhysicsInformedHybridSurrogate (Model C):
   Learns residual corrections relative to simplified ballistic flight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.ln1(self.fc1(x)))
        out = self.drop(out)
        out = self.ln2(self.fc2(out))
        return self.act(out + residual)


class NeuralTrajectorySurrogate(nn.Module):
    """
    Model B: Deep Neural Network predicting complete 3D firebrand trajectory.
    Input: [h, u_wind, wind_angle, mass, area, cd, z0, w_updraft, v0z] (dim=9)
    Outputs:
    - 3D Position states across 50 normalized timesteps: shape [batch, 50, 3]
    - Final landing outcomes: [x_land, y_land, t_flight]: shape [batch, 3]
    """
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        num_trajectory_points: int = 50,
        dropout: float = 0.05
    ):
        super().__init__()
        self.num_pts = num_trajectory_points
        
        # Feature projector
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Residual backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        
        # Head 1: 3D Trajectory coordinate predictor (50 timesteps * 3 coords = 150 values)
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_trajectory_points * 3)
        )
        
        # Head 2: Landing outcomes [landing_x, landing_y, flight_time]
        self.landing_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
            
        # Reshape trajectory output to [B, 50, 3]
        traj_flat = self.traj_head(h)
        traj_3d = traj_flat.view(-1, self.num_pts, 3)
        
        # Landing targets [x_land, y_land, t_flight]
        landing_out = self.landing_head(h)
        
        return traj_3d, landing_out


class PhysicsInformedHybridSurrogate(nn.Module):
    """
    Model C: Physics-Informed Residual Network.
    Predicts residual aerodynamic offsets to ballistic trajectory predictions:
    x_pred(t) = x_ballistic(t) + delta_x(t)
    """
    def __init__(self, input_dim: int = 9, hidden_dim: int = 96, num_pts: int = 50):
        super().__init__()
        self.num_pts = num_pts
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_pts * 3 + 3)
        )

    def forward(self, x: torch.Tensor, base_traj: torch.Tensor, base_landing: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        delta_traj = out[:, :self.num_pts * 3].view(-1, self.num_pts, 3)
        delta_landing = out[:, self.num_pts * 3:]
        
        pred_traj = base_traj + delta_traj
        pred_landing = base_landing + delta_landing
        return pred_traj, pred_landing
