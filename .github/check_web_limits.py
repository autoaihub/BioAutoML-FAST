"""Exercise the same per-set checks used by the Home and Repository forms."""
from io import BytesIO
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bioautoml import web_limits


class WebLimitsTests(unittest.TestCase):
    def mode(self, setting):
        with patch.dict(os.environ):
            if setting is None:
                os.environ.pop("BIOAUTOML_LIMIT_SAMPLES", None)
            else:
                os.environ["BIOAUTOML_LIMIT_SAMPLES"] = setting
            importlib.reload(web_limits)
        self.addCleanup(importlib.reload, web_limits)

    def test_default_checks_each_set_independently(self):
        self.mode(None)
        for kind in ("Protein", "Structured data"):
            def upload(n):
                if kind == "Protein":
                    return BytesIO(b">seq\nAC\n" * n)
                return BytesIO(b"value,label\n" + b'"a\nb",1\n' * n)

            files = [upload(2500), upload(2500)]
            for label in ("Training set", "Test set", "Prediction set"):
                self.assertIsNone(web_limits.sample_limit_error(files, kind, label))
                error = web_limits.sample_limit_error(files + [upload(1)], kind, label)
                self.assertIn("5,001", error)
                self.assertTrue(all(f.tell() == 0 for f in files))

    def test_opt_out_accepts_large_sets(self):
        self.mode("no")
        self.assertIsNone(web_limits.sample_limit_error(BytesIO(b">s\nAC\n" * 5001), "DNA/RNA", "Prediction set"))
        self.assertIn("no sample-count limit", web_limits.SAMPLE_LIMIT_TEXT)
        self.assertIn("no sequence-count limit", web_limits.PREDICTION_LIMIT_TEXT)


if __name__ == "__main__":
    unittest.main()
