from .base import ResearchModule, ModuleMetadata
from .generation import FirebrandGenerationInterface, FirebrandGenerationRecord
from .transport import AtmosphericTransportInterface, TrajectoryState
from .deposition import FirebrandDepositionInterface, DepositionRunRecord
from .thermal import ThermalSurvivalInterface, ThermalSurvivalRecord
from .ignition import ReceptiveFuelIgnitionInterface, ReceptiveFuelIgnitionRecord
from .spotting import WildfireSpottingInterface, WildfireSpottingRecord

__all__ = [
    "ResearchModule",
    "ModuleMetadata",
    "FirebrandGenerationInterface",
    "FirebrandGenerationRecord",
    "AtmosphericTransportInterface",
    "TrajectoryState",
    "FirebrandDepositionInterface",
    "DepositionRunRecord",
    "ThermalSurvivalInterface",
    "ThermalSurvivalRecord",
    "ReceptiveFuelIgnitionInterface",
    "ReceptiveFuelIgnitionRecord",
    "WildfireSpottingInterface",
    "WildfireSpottingRecord",
]
