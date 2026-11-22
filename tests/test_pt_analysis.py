"""Run with:  python -m unittest discover tests -v"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from layer1_analytics.pt_analysis import (analysis_frame, analyze, association_table, build_tables, dose_response, holm,
                                          measurement_features, wilson)
from layer1_analytics.pt_wearable_features import KEYS, window_features
from layer1_analytics.pt_config import PTConfig
from layer1_analytics.pt_risk import normalize
from layer1_analytics.pt_synthetic import generate

CFG = PTConfig()
T = build_tables(CFG)
DF = analysis_frame(T, "cohort")


def pooled(cfg, n=8, tag="p"):
    return pd.concat([analysis_frame(build_tables(replace(cfg, seed=7000 + k)), f"{tag}{k}") for k in range(n)],
                     ignore_index=True)


class TestFrame(unittest.TestCase):
    def test_only_decided_packages(self):
        pend = set(T["outcomes"].loc[T["outcomes"].renewed == "Pending", "package_id"])
        self.assertFalse(set(DF["package_id"]) & pend)
        self.assertFalse(set(DF["package_id"]) & set(T["packages"].loc[T["packages"].status != "Completed", "package_id"]))

    def test_maternity_case_is_included_and_tagged(self):
        row = DF[DF["client_id"] == "C001"].iloc[0]
        self.assertEqual((row.maternity_pause, row.not_renewed, row.reason_category), (1, 1, "Life stage change"))
        self.assertGreater(row.pause_days, 100)

    def test_skip_rate_is_share_of_attempts_not_attended(self):
        pid = DF["package_id"].iloc[0]
        s = T["sessions"][T["sessions"].package_id == pid]
        self.assertAlmostEqual(DF.iloc[0].skip_rate, (s.outcome != "Attended").mean(), places=9)


class TestActivityAndMeasurementFeatures(unittest.TestCase):
    def test_activity_features_present_and_plausible(self):
        self.assertTrue(DF[["steps_30", "active_min_30", "wear_30"]].notna().all().all())
        self.assertTrue(DF["wear_30"].between(0, 1).all())
        self.assertTrue(DF["steps_30"].between(500, 20000).all())

    def test_activity_window_uses_only_data_up_to_the_reference_date(self):
        daily = T["daily_activity"]
        cid = DF["client_id"].iloc[0]
        d = daily[daily["Id"] == cid]
        ref = d["ActivityDate"].iloc[len(d) // 2]
        full = window_features(d, ref)
        cut = window_features(d[d["ActivityDate"] <= ref], ref)
        for k in KEYS:
            self.assertTrue((np.isnan(full[k]) and np.isnan(cut[k])) or abs(full[k] - cut[k]) < 1e-9, k)

    def test_body_measures_exist_and_are_plausible(self):
        m = T["measurements"]
        for name in ("Body weight", "Body fat percentage", "Skeletal muscle mass"):
            v = m.loc[m.measure_name == name, "value"]
            self.assertGreater(len(v), 50)
        self.assertTrue(m.loc[m.measure_name == "Body weight", "value"].between(35, 150).all())
        self.assertTrue(m.loc[m.measure_name == "Body fat percentage", "value"].between(5, 60).all())
        self.assertTrue(DF["baseline_weight"].notna().all())

    def test_change_needs_28_days_and_progress_is_not_defined_for_prenatal(self):
        meas = T["measurements"]
        cid = DF.loc[DF.goal == "Fat loss", "client_id"].iloc[0]
        first = meas[(meas.client_id == cid) & (meas.measure_name == "Body weight")]["date"].min()
        out = measurement_features(meas, cid, "Fat loss", first, first + pd.Timedelta(days=10))
        self.assertTrue(np.isnan(out["delta_weight"]))
        out2 = measurement_features(meas, "C001", "Prenatal", pd.Timestamp("2025-10-06"), pd.Timestamp("2026-08-01"))
        self.assertTrue(np.isnan(out2["goal_progress"]))

    def test_postpartum_progress_uses_body_fat_direction(self):
        rows = DF[DF["goal"].isin(["Fat loss", "Postpartum"]) & DF["goal_progress"].notna()]
        self.assertTrue(((rows["delta_bodyfat"] < 0).astype(float) == rows["goal_progress"]).all())


class TestHelpers(unittest.TestCase):
    def test_holm(self):
        out = holm([0.01, 0.04, 0.03, np.nan])
        self.assertAlmostEqual(out[0], 0.03)       # 0.01 x 3
        self.assertAlmostEqual(out[2], 0.06)       # 0.03 x 2
        self.assertAlmostEqual(out[1], 0.06)       # monotone: not below the previous adjusted value
        self.assertTrue(np.isnan(out[3]))

    def test_wilson_interval(self):
        lo, hi = wilson(5, 10)
        self.assertTrue(0.2 < lo < 0.5 < hi < 0.8)
        self.assertTrue(np.isnan(wilson(0, 0)[0]))               # NaN when n = 0


class TestMethodValidation(unittest.TestCase):
    """The method must find what was built in, and not invent what was not."""

    @classmethod
    def setUpClass(cls):
        cls.base = association_table(pooled(CFG, 8, "null"), B=150)

    def test_finds_the_built_in_skip_effect(self):
        r = self.base.query("key == 'skip_rate'").iloc[0]
        self.assertEqual(r.evidence, "More non-renewal")
        self.assertGreater(r.effect, 0.65)

    def test_finds_activity_outside_the_gym_and_not_body_measures(self):
        a = self.base.set_index("key")
        self.assertEqual(a.loc["steps_30", "evidence"], "Less non-renewal")        # more active -> fewer non-renewals
        self.assertLess(a.loc["steps_30", "effect"], 0.40)
        for k in ("baseline_weight", "baseline_bodyfat", "baseline_smm"):
            self.assertNotIn(a.loc[k, "evidence"], ("More non-renewal", "Less non-renewal"), k)

    def test_does_not_invent_a_pause_effect(self):
        r = self.base.query("key == 'pause_days'").iloc[0]
        self.assertNotIn(r.evidence, ("More non-renewal", "Less non-renewal"))

    def test_detects_a_strong_planted_pause_effect(self):
        strong = replace(PTConfig(share_random_pause=0.30), renewal_pause_penalty_per_30d=0.30)
        r = association_table(pooled(strong, 8, "pen"), B=150).query("key == 'pause_days'").iloc[0]
        self.assertEqual(r.evidence, "More non-renewal")     # longer pause -> less renewal was planted

    def test_small_cohort_says_too_few_cases_for_pauses(self):
        r = association_table(DF, B=100).query("key == 'pause_days'").iloc[0]
        self.assertTrue(r.evidence.startswith("Too few cases"))


class TestOutputs(unittest.TestCase):
    def test_dose_response_rates_rise_with_skip_rate(self):
        d = dose_response(pooled(CFG, 4, "d")).query("analysis == 'Skip rate'")
        self.assertTrue(d["rate"].is_monotonic_increasing)

    def test_analyze_returns_all_parts(self):
        res = analyze(DF, B=100)
        self.assertEqual(set(res), {"association", "dose", "adjusted", "info", "extras", "n", "events", "clients"})
        self.assertIn("steps_30", set(res["extras"]["key"]))
        self.assertEqual(res["n"], len(DF))


if __name__ == "__main__":
    unittest.main()
