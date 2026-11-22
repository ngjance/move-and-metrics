"""Run with:  python -m unittest discover tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from layer1_analytics import churn_model
from layer1_analytics.features import client_status, trailing_inactive_streak, weekly_summary
from layer1_analytics.synthetic import generate_clients


class TestStreak(unittest.TestCase):
    def test_counts_only_trailing_false(self):
        self.assertEqual(trailing_inactive_streak([True, False, False, True, False, False, False]), 3)

    def test_zero_when_last_day_active(self):
        self.assertEqual(trailing_inactive_streak([False, False, True]), 0)


class TestStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.truth = generate_clients(n_clients=20, days=60)
        cls.status = client_status(cls.df).merge(cls.truth, on="Id")

    def test_one_row_per_client(self):
        self.assertEqual(len(self.status), 20)

    def test_declining_clients_are_flagged_high(self):
        dec = self.status[self.status["persona"] == "declining"]
        self.assertTrue((dec["status"] == "High").all(), dec[["status", "reasons"]])

    def test_steady_clients_are_not_high(self):
        steady = self.status[self.status["persona"] == "steady"]
        self.assertFalse((steady["status"] == "High").any(), steady[["status", "reasons"]])

    def test_missing_days_count_as_inactive(self):
        df, _ = generate_clients(n_clients=2, days=30)
        cid = df["Id"].iloc[0]
        cutoff = df["ActivityDate"].max() - pd.Timedelta(days=4)
        df = df[~((df["Id"] == cid) & (df["ActivityDate"] > cutoff))]  # client stops logging
        s = client_status(df).set_index("Id")
        self.assertGreaterEqual(s.loc[cid, "inactive_streak_days"], 4)
        self.assertEqual(s.loc[cid, "status"], "High")

    def test_weekly_summary_columns(self):
        w = weekly_summary(self.df)
        self.assertTrue({"Id", "Week", "avg_steps", "active_minutes", "active_days"} <= set(w.columns))


class TestChurnPipeline(unittest.TestCase):
    def test_demo_pipeline_runs_and_beats_chance(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = churn_model.demo_dataset(n=1500)
            metrics, imp = churn_model.run(df, "churn", out_dir=Path(tmp), label="test")
            self.assertGreater(metrics["roc_auc"], 0.6)
            self.assertTrue((Path(tmp) / "metrics.json").exists())
            self.assertIn("visits_per_week", imp["feature"].tolist())

    def test_target_mapping(self):
        self.assertEqual(churn_model.to_binary(pd.Series(["Yes", "no", "Churned"])).tolist(), [1, 0, 1])
        with self.assertRaises(ValueError):
            churn_model.to_binary(pd.Series(["maybe", "yes"]))


if __name__ == "__main__":
    unittest.main()
