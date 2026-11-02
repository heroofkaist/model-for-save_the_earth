from dataclasses import dataclass
from typing import Dict, Any, List
from .base import ResearchModule, ModuleMetadata


@dataclass
class TrajectoryState:
    t_s: float
    x_m: float
    y_m: float
    z_m: float
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float


class AtmosphericTransportInterface(ResearchModule):
    """Interface for 3D Atmospheric Particle Transport module."""

    @property
    def metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            module_name="atmospheric_transport",
            task_family="particle_trajectory",
            description="3D trajectories governed by gravity, drag, lift, wind shear, and buoyant plume dynamics.",
            target_variable="trajectory_vector_3d",
            target_units="meters (x, y, z)",
            is_train_ready=False,
            data_source_requirements=["CSIRO 3D wind-tunnel PIV", "High-speed stereo videogrammetry"],
            scientific_disclaimer="Requires synchronized 3D velocity fields and time-resolved position tracks."
        )

    def validate_schema(self, data: Dict[str, Any]) -> bool:
        required = {"record_id", "trajectory_points", "wind_velocity_3d"}
        return required.issubset(data.keys())

