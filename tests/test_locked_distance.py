import csv
import tempfile
import unittest
from pathlib import Path
from train_locked_distance import evaluate

class LockedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        p=Path(self.temp.name)
        self.data=p/"data.csv"
        self.split=p/"split.csv"
        rows=[]; splits=[]
        for g,part in [("a","train"),("b","train"),("c","validation"),("d","test")]:
            for i in range(2):
                rid=f"{g}{i}"
                rows.append(dict(record_id=rid,source_id="usda",experiment_id=g,
                                 qc_status="pass",task_family="particle_distance",
                                 data_kind="measured",landing_distance_m=str(i+1)))
                splits.append(dict(record_id=rid,source_id="usda",group_id=g,
                                   task_family="particle_distance",split=part))
        for path,data in ((self.data,rows),(self.split,splits)):
            with path.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=data[0].keys())
                writer.writeheader(); writer.writerows(data)
    def tearDown(self):
        self.temp.cleanup()
    def test_deterministic_baseline(self):
        result=evaluate(self.data,self.split)
        self.assertEqual(result["test"]["n"],2)
        self.assertEqual(result["predictor"],"training_partition_median")
    def test_no_group_leakage(self):
        with self.split.open(newline="") as f:
            rr=list(csv.DictReader(f))
        rr[-1]["split"]="train"
        with self.split.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rr[0].keys())
            w.writeheader();w.writerows(rr)
        with self.assertRaisesRegex(ValueError,"group leakage"):
            evaluate(self.data,self.split)

if __name__=="__main__":
    unittest.main()
