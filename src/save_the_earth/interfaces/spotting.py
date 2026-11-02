from dataclasses import dataclass
from typing import Dict, Any, List
from .base import ResearchModule, ModuleMetadata


@dataclass
class WildfireSpottingRecord:
    fire_id: str
    incident_name: str
    spot_fire_distance_m: float
    main_front_rate_of_spread_m_min: float
    ambient_wind_speed_10m_m_s: float
    terrain_slope_degrees: float
    fuel_model: str
    drought_code: float
    spotting_density_per_km2: float


class WildfireSpottingInterface(ResearchModule):
    """Interface for Landscape/Mesoscale Wildfire Spotting module."""

    @property
    def metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            module_name="wildfire_spotting",
            task_family="wildfire_spotting",
            description="Predicts macroscopic spot fire distances and new cluster ignition ahead of the active fire front.",
            target_variable="spot_fire_distance_m",
            target_units="meters",
            is_train_ready=False,
            data_source_requirements=["Australian Black Summer airborne line-scan archives", "MDPI operational databases"],
            scientific_disclaimer="Never train particle trajectory or pan landing models on macroscopic spot fire distances."
        )

    def validate_schema(self, data: Dict[str, Any]) -> bool:
        required = {"fire_id", "spot_fire_distance_m", "ambient_wind_speed_10m_m_s"}
        return required.issubset(data.keys())
