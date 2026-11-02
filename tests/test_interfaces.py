import unittest
from src.save_the_earth.interfaces import (
    FirebrandDepositionInterface,
    FirebrandGenerationInterface,
    AtmosphericTransportInterface,
    ThermalSurvivalInterface,
    ReceptiveFuelIgnitionInterface,
    WildfireSpottingInterface
)


class TestScientificInterfaces(unittest.TestCase):
    def test_deposition_interface(self):
        dep = FirebrandDepositionInterface()
        self.assertTrue(dep.metadata.is_train_ready)
        self.assertEqual(dep.metadata.task_family, "particle_distance")
        valid = dep.validate_schema({
            "record_id": "r1",
            "experiment_id": "e1",
            "wind_speed_m_s": 5.36,
            "landing_distance_m": 1.17
        })
        self.assertTrue(valid)
        invalid = dep.validate_schema({"record_id": "r1"})
        self.assertFalse(invalid)

    def test_unready_interfaces(self):
        modules = [
            FirebrandGenerationInterface(),
            AtmosphericTransportInterface(),
            ThermalSurvivalInterface(),
            ReceptiveFuelIgnitionInterface(),
            WildfireSpottingInterface()
        ]
        for m in modules:
            self.assertFalse(m.metadata.is_train_ready, f"{m.metadata.module_name} should not be marked train_ready without verified data.")


if __name__ == "__main__":
    unittest.main()
