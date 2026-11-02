from dataclasses import dataclass
from typing import Dict, Any, List
from .base import ResearchModule, ModuleMetadata


@dataclass
class DepositionRunRecord:
    record_id: str
    experiment_id: str
    fuel_category: str
    fuel_type: str
    wind_speed_m_s: float
    sample_size: int
    landing_distance_median_m: float
    landing_distance_mean_m: float
    landing_distance_std_m: float
    landing_distance_skewness: float
    measurement_method: str  # water_pan, collector_trough


class FirebrandDepositionInterface(ResearchModule):
    """Interface for Firebrand Landing & Deposition module (Current operational focus)."""

    @property
    def metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            module_name="firebrand_deposition",
            task_family="particle_distance",
            description="Predicts downwind travel distance and spatial landing dispersion from wind-tunnel water pan collections.",
            target_variable="landing_distance_m",
            target_units="meters",
            is_train_ready=True,
            data_source_requirements=["USDA JFSP 15-1-04-4 water-pan collection runs"],
            scientific_disclaimer="Models experimental median landing distance; does not represent ignition or wild spot fires."
        )

    def validate_schema(self, data: Dict[str, Any]) -> bool:
        required = {"record_id", "experiment_id", "wind_speed_m_s", "landing_distance_m"}
        return required.issubset(data.keys())

