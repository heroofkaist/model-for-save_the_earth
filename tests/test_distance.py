import csv
import tempfile
import unittest
from pathlib import Path
from train_distance import REQUIRED, read_data, train

class TestDistance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.file = Path(self.temp.name)/"obs.csv"
        self.rows = []
        for group in range(8):
            for particle in range(5):
                self.rows.append(dict(sample_id=f"{group}-{particle}",
                    experiment_id=f"session-{group}",source_id="test-fixture",
                    data_kind="measured",mass_kg=.001+.0001*particle,
                    area_m2=.0002+.00001*particle,wind_speed_m_s=2+group,
                    fuel_type="wood",flight_distance_m=3+.5*group+.1*particle))
        self.write()
    def tearDown(self):
        self.temp.cleanup()
    def write(self):
        with self.file.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=REQUIRED)
            writer.writeheader()
            writer.writerows(self.rows)
    def test_group_isolation(self):
        report,model,heldout=train(read_data(self.file))
        held=set(report["held_out_experiments"])
        self.assertTrue(all(row["experiment_id"] in held for row,_,_ in heldout))
        self.assertEqual(len(heldout),report["test_size"])
        self.assertGreaterEqual(report["ridge_regression"]["mae_m"],0)
    def test_refuses_synthetic(self):
        self.rows[0]["data_kind"]="synthetic"
        self.write()
        with self.assertRaisesRegex(ValueError,"refusing synthetic"):
            read_data(self.file)
    def test_refuses_duplicate(self):
        self.rows[1]["sample_id"]=self.rows[0]["sample_id"]
        self.write()
        with self.assertRaisesRegex(ValueError,"duplicate"):
            read_data(self.file)

if __name__=="__main__":
    unittest.main()
