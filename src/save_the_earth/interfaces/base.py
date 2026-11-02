"""
Save the Earth — Scientific Domain Interfaces for Wildfire Firebrand Modeling.

Defines modular abstract interfaces and data schemas across six interconnected research targets:
1. Firebrand Generation (generation.py)
2. Atmospheric Transport (transport.py)
3. Deposition & Landing (deposition.py)
4. Thermal Survival (thermal.py)
5. Receptive Fuel Ignition (ignition.py)
6. Mesoscale Wildfire Spotting (spotting.py)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class ModuleMetadata:
    module_name: str
    task_family: str
    description: str
    target_variable: str
    target_units: str
    is_train_ready: bool
    data_source_requirements: List[str]
    scientific_disclaimer: str


class ResearchModule(ABC):
    """Abstract Base Class for all scientific research modules in Save the Earth."""

    @property
    @abstractmethod
    def metadata(self) -> ModuleMetadata:
        """Returns the module metadata and readiness status."""
        pass

    @abstractmethod
    def validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validates that input record conforms to domain specifications."""
        pass
