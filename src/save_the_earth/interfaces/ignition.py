from dataclasses import dataclass
from typing import Dict, Any, List
from .base import ResearchModule, ModuleMetadata


@dataclass
class ReceptiveFuelIgnitionRecord:
    record_id: str
    fuel_bed_type: str  # pine_needles, shredded_mulch, dry_grass
    fuel_moisture_content_percent: float
    ambient_relative_humidity_percent: float
    wind_velocity_m_s: float
    ember_mass_g: float
    ember_temperature_c: float
    ignited: bool
    time_to_ignition_s: Optional[float]


class ReceptiveFuelIgnitionInterface(ResearchModule):
    """Interface for Receptive Fuel Ignition module."""

    @property
    def metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            module_name="fuel_ignition",
            task_family="receptive_ignition",
            description="Predicts binary ignition probability conditional on fuel bed moisture, fuel bulk density, and heat flux.",
            target_variable="ignited",
            target_units="boolean (0 or 1)",
            is_train_ready=False,
            data_source_requirements=["Laboratory hot-particle spot-ignition testbeds"],
            scientific_disclaimer="Ignition cannot be inferred from particle landing alone; fuel bed condition is decisive."
        )

    def validate_schema(self, data: Dict[str, Any]) -> bool:
        required = {"record_id", "fuel_bed_type", "fuel_moisture_content_percent", "ignited"}
        return required.issubset(data.keys())
