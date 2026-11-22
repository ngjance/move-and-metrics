"""Run with:  python -m unittest discover tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from layer1_analytics.pt_config import PTConfig
from layer1_analytics.pt_risk import (NUMERIC, _snapshot_date, normalize, package_features, risk_table,
                                      train_model, training_frame)
from layer1_analytics.pt_analysis import build_tables
from layer1_analytics.pt_synthetic import generate
from layer1_analytics.pt_wearable_features import KEYS

CFG = PTConfig()
T = build_tables(CFG)


class TestTrainingFrame(unittest.TestCase):
    def test_excludes_pending_and_life_stage_non_renewals(self):
        tf = training_frame(T, CFG)
        pend = set(T["outcomes"].loc[T["outcomes"].renewed == "Pending", "package_id"])
        life = set(T["outcomes"].loc[T["outcomes"].reason_category == "Life stage change", "package_id"])
        self.assertFalse(set(tf["package_id"]) & (pend | life))
        self.assertIn("C001", set(T["clients"]["client_id"]))   # the planted maternity case exists but is excluded

    def test_label_matches_outcomes(self):
        tf = training_frame(T, CFG).merge(T["outcomes"][["package_id", "renewed"]], on="package_id")
        self.assertTrue(((tf["renewed"] == "No").astype(int) == tf["not_renewed"]).all())

    def test_no_future_leakage(self):
        """Features at the snapshot must not change if everything after the snapshot is deleted."""
        cl = T["clients"].set_index("client_id")
        checked = 0
        for _, pkg in T["packages"].iterrows():
            if pkg["status"] != "Completed":
                continue
            snap = _snapshot_date(pkg, T["sessions"], CFG)
            full = package_features(pkg, T["sessions"], T["pauses"], cl.loc[pkg["client_id"]], snap, CFG, T["daily_activity"])
            trunc = T["sessions"][T["sessions"]["date"] <= snap]
            trunc_daily = T["daily_activity"][T["daily_activity"]["ActivityDate"] <= snap]
            cut = package_features(pkg, trunc, T["pauses"], cl.loc[pkg["client_id"]], snap, CFG, trunc_daily)
            for k in NUMERIC + KEYS:
                a, b = full[k], cut[k]
                self.assertTrue((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, k)
            checked += 1
            if checked >= 12:
                break
        self.assertGreater(checked, 5)


class TestNoExpiry(unittest.TestCase):
    def test_features_do_not_depend_on_expiry(self):
        self.assertNotIn("finish_gap_days", NUMERIC)
        self.assertIn("weeks_to_finish", NUMERIC)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe, cls.cols, cls.bounds, cls.metrics, cls.coefs = train_model(CFG, seeds=range(1, 7))

    def test_progress_variables_are_not_model_inputs(self):
        self.assertNotIn("used_pct", self.cols)
        self.assertNotIn("days_since_start", self.cols)

    def test_activity_inputs_are_in_the_model(self):
        self.assertTrue(set(KEYS) <= set(self.cols))
        self.assertTrue(self.metrics["uses_activity_last_30_days"])

    def test_beats_chance_under_grouped_cv(self):
        self.assertGreater(self.metrics["grouped_cv_auc_mean"], 0.6)

    def test_scores_are_probabilities_and_not_degenerate(self):
        rt = risk_table(T, CFG, (self.pipe, self.cols, self.bounds))
        p = rt["model_risk"].dropna()
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(p.min(), 1e-4)      # the earlier bug produced scores like 1e-11
        self.assertLess(p.max(), 1 - 1e-4)

    def test_risk_table_priorities_and_planted_client(self):
        rt = risk_table(T, CFG, (self.pipe, self.cols, self.bounds))
        self.assertTrue(set(rt["priority"]) <= {"High", "Watch", "OK", "Too early", "Paused", "Renewal pending"})
        self.assertFalse({"rule_level", "rule_points", "rule_reasons"} & set(rt.columns))   # no hand-set rules
        self.assertEqual(rt["client_id"].is_unique, True)
        scored = rt[rt["model_risk"].notna()]
        self.assertTrue(((scored["priority"] == "High") == (scored["model_risk"] >= CFG.model_high)).all())


class TestActivityComparison(unittest.TestCase):
    def test_comparison_has_both_models(self):
        from layer1_analytics.pt_risk import build_training_data, compare_activity_value
        cmp = compare_activity_value(CFG, repeats=1, data=build_training_data(CFG, seeds=range(1, 5)))
        self.assertEqual(len(cmp), 2)
        self.assertGreater(cmp.loc[1, "features"], cmp.loc[0, "features"])
        self.assertTrue((cmp["cv_auc"] > 0.5).all())


if __name__ == "__main__":
    unittest.main()
