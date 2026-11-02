from dataclasses import dataclass
from typing import Dict, Any, List
from .base import ResearchModule, ModuleMetadata


@dataclass
class ThermalSurvivalRecord:
    record_id: str
    initial_temperature_c: float
    surface_temperature_c: float
    flight_time_s: float
    mass_loss_rate_percent: float
    heat_flux_kw_m2: float
    flame_state: str  # flaming, glowing, extinguished


class ThermalSurvivalInterface(ResearchModule):
    """Interface for In-Flight Thermal Survival and Combustion module."""

    @property
    def metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            module_name="thermal_survival",
            task_family="thermal_survival",
            description="Models ember surface temperature, combustion duration, and residual heat flux upon landing.",
            target_variable="heat_flux_kw_m2",
            target_units="kW/m²",
            is_train_ready=False,
            data_source_requirements=["Zenodo 1406207 calibrated thermal infrared archives"],
            scientific_disclaimer="Thermal survival requires time-resolved radiometer or calibrated IR imaging data."
        )

    def validate_schema(self, data: Dict[str, Any]) -> bool:
        required = {"record_id", "surface_temperature_c", "heat_flux_kw_m2"}
        return required.issubset(data.keys())
