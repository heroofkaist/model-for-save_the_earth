from dataclasses import dataclass
from typing import Dict, Any, List
from .base import ResearchModule, ModuleMetadata


@dataclass
class FirebrandGenerationRecord:
    experiment_id: str
    fuel_species: str
    combustion_regime: str  # flaming, smoldering, glowing
    source_mass_loss_rate_g_s: float
    generation_rate_particles_s: float
    particle_mass_g: float
    particle_projected_area_cm2: float
    morphology: str  # cylinder, disc, flake, irregular


class FirebrandGenerationInterface(ResearchModule):
    """Interface for Firebrand Generation research module."""

    @property
    def metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            module_name="firebrand_generation",
            task_family="particle_generation",
            description="Characterizes particle emission rates, geometry, and mass detached from burning fuel.",
            target_variable="generation_rate_particles_s",
            target_units="particles/s",
            is_train_ready=False,
            data_source_requirements=["Laboratory ember generators", "Dragon-flame experiments"],
            scientific_disclaimer="Generation rates depend heavily on fuel moisture and local airflow around the canopy."
        )

    def validate_schema(self, data: Dict[str, Any]) -> bool:
        required = {"experiment_id", "fuel_species", "generation_rate_particles_s"}
        return required.issubset(data.keys())
