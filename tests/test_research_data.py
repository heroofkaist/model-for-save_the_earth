import unittest
from research_data import audit, partition

SHA="a"*64

def row(i,group="run1",task="particle_distance"):
    return {"record_id":f"r{i}", "source_id":"usda", "source_file":"archive.csv",
            "source_file_sha256":SHA,"source_row_id":str(i),
            "observation_type":"wind_tunnel_particle", "task_family":task,
            "experiment_id":group, "fire_id":"", "license_spdx":"CC0-1.0",
            "measurement_method":"water_pan", "data_kind":"measured",
            "landing_distance_m":"10","distance_definition":"pan_center_distance",
            "measurement_stage":"post_landing","mass_kg":"0.001",
            "projected_area_m2":"0.0001"}

class ResearchDataTests(unittest.TestCase):
    def test_correct_qc(self):
        self.assertEqual(audit([row(1)])[0]["qc_status"],"pass")
    def test_reject_synthetic(self):
        r=row(1);r["data_kind"]="synthetic"
        self.assertIn("not_measured",audit([r])[0]["qc_reason"])
    def test_no_invented_group(self):
        r=row(1);r["experiment_id"]=""
        self.assertIn("missing_independent_group",audit([r])[0]["qc_reason"])
    def test_invalid_units(self):
        r=row(1);r["mass_kg"]="-1"
        self.assertIn("invalid_mass_kg",audit([r])[0]["qc_reason"])
    def test_duplicate_source_row(self):
        records=audit([row(1),{**row(1),"record_id":"r2"}])
        self.assertIn("duplicate_source_row",records[1]["qc_reason"])
    def test_group_disjoint_and_deterministic(self):
        rows=audit([row(i,group=f"session{i//3}") for i in range(30)])
        a=partition(rows);b=partition(rows)
        self.assertEqual(a,b)
        groups={}
        for rec in a:
            groups.setdefault(rec["group_id"],set()).add(rec["split"])
        self.assertTrue(all(len(s)==1 for s in groups.values()))
        self.assertEqual({r["split"] for r in a},{"train","validation","test"})
    def test_distinct_spotting_family(self):
        r=row(1,task="wildfire_spotting")
        r.update({"observation_type":"spot_fire","fire_id":"fireA","spot_fire_distance_m":"250"})
        self.assertEqual(audit([r])[0]["qc_status"],"pass")
    def test_trajectory_missing_tracks(self):
        r=row(1,task="particle_trajectory")
        r["flight_time_s"]="4"
        self.assertIn("missing_trajectory_points",audit([r])[0]["qc_reason"])
if __name__=="__main__":
    unittest.main()
