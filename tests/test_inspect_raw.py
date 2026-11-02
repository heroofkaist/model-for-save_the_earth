import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from inspect_raw import inspect

class RawInspectorTests(unittest.TestCase):
    def test_csv_preserves_rows_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"measurements.csv"
            with path.open("w",newline="") as f:
                w=csv.writer(f);w.writerows([["mass (g)","distance (m)"],["1","2"],["3","4"]])
            a=inspect(path)
            self.assertEqual(a["records"],2)
            self.assertEqual(a["columns"],["mass (g)","distance (m)"])
            self.assertEqual(len(a["sha256"]),64)
    def test_zip_list_without_extracting(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"source.zip"
            with zipfile.ZipFile(path,"w") as z:z.writestr("data.csv","a,b\n1,2\n")
            a=inspect(path)
            self.assertEqual(a["entries"][0]["name"],"data.csv")
            self.assertFalse((Path(d)/"data.csv").exists())
if __name__=="__main__":
    unittest.main()
