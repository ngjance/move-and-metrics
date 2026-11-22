"""Run with:  python -m unittest discover tests -v"""
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from layer1_analytics.pt_config import PTConfig
from layer1_analytics.pt_features import package_metrics
from layer1_analytics.pt_synthetic import generate
from layer1_analytics.wearables import build_daily_activity

CFG = PTConfig()
T = generate(CFG)
C, P, S, Z, O = (T[k] for k in ["clients", "packages", "sessions", "pauses", "outcomes"])
M = package_metrics(P, S, Z, CFG)


class TestSpecs(unittest.TestCase):
    def test_client_count_and_goal_mix(self):
        self.assertEqual(len(C), 35)
        self.assertEqual(C["goal"].value_counts().to_dict(), CFG.goal_counts)
        self.assertGreater(CFG.goal_counts["Fat loss"] / 35, 0.5)

    def test_longevity_clients_are_above_55_and_others_are_not(self):
        self.assertTrue((C.loc[C.goal == "Longevity", "age"] > 55).all())
        self.assertTrue((C.loc[C.goal != "Longevity", "age"] <= 55).all())

    def test_package_sizes_and_prices_match_spec(self):
        self.assertTrue(set(P["sessions_signed"]) <= {20, 50, 100})
        expected = P["sessions_signed"].map({20: 195, 50: 165, 100: 155})
        self.assertTrue((P["price_per_session"] == expected).all())

    def test_outcome_rates_are_close_to_spec(self):
        r = S["outcome"].value_counts(normalize=True)
        self.assertAlmostEqual(r["Early cancel"], 0.30, delta=0.03)
        self.assertAlmostEqual(r["Late cancel"], 0.10, delta=0.03)
        self.assertLess(r["No-show"], 0.03)

    def test_cancel_notice_rules(self):
        early = S[S.outcome == "Early cancel"]["cancel_notice_hours"]
        late = S[S.outcome == "Late cancel"]["cancel_notice_hours"]
        self.assertTrue((early >= 24).all())
        self.assertTrue((late < 24).all())


class TestIntegrity(unittest.TestCase):
    def test_deterministic(self):
        again = generate(PTConfig())
        pd.testing.assert_frame_equal(again["sessions"], S)

    def test_ids_are_unique_and_linked(self):
        self.assertTrue(S["session_id"].is_unique and P["package_id"].is_unique)
        self.assertTrue(set(S["package_id"]) <= set(P["package_id"]))
        self.assertTrue(set(P["client_id"]) <= set(C["client_id"]))
        self.assertTrue(set(Z["package_id"]) <= set(P["package_id"]))
        self.assertTrue(set(O["package_id"]) <= set(P["package_id"]))

    def test_no_sessions_after_as_of_or_before_start(self):
        self.assertTrue((pd.to_datetime(S["date"]) <= pd.Timestamp(CFG.as_of)).all())
        first = S.merge(P[["package_id", "start_date"]], on="package_id")
        self.assertTrue((pd.to_datetime(first["date"]) >= pd.to_datetime(first["start_date"])).all())

    def test_sessions_used_never_exceed_signed(self):
        self.assertTrue((M["sessions_left"] >= 0).all())

    def test_no_sessions_during_a_pause(self):
        for _, z in Z.iterrows():
            end = z["pause_end"] if pd.notna(z["pause_end"]) else CFG.as_of
            inside = S[(S.package_id == z.package_id) & (pd.to_datetime(S.date) >= pd.Timestamp(z.pause_start))
                       & (pd.to_datetime(S.date) < pd.Timestamp(end))]
            self.assertTrue(inside.empty, z.pause_id)

    def test_hidden_variables_not_in_model_tables(self):
        for df in (C, P, S, Z, O):
            self.assertFalse({"hidden_engagement", "p_early", "p_renew"} & set(df.columns))


class TestPlantedScenarios(unittest.TestCase):
    def test_prenatal_pause_with_exactly_ten_left_then_postpartum_finish_no_renewal(self):
        pause = Z[(Z.client_id == "C001") & (Z.reason_category == "Maternity")].iloc[0]
        pk = M[M.package_id == pause.package_id].iloc[0]
        before = S[(S.package_id == pause.package_id) & (pd.to_datetime(S.date) < pd.Timestamp(pause.pause_start))]
        used = (before.outcome == "Attended").sum() + before.outcome.isin(["Late cancel", "No-show"]).sum()
        self.assertEqual(pk.sessions_signed - used, 10)
        self.assertEqual(set(before["phase"]), {"Prenatal"})
        after = S[(S.package_id == pause.package_id) & (pd.to_datetime(S.date) >= pd.Timestamp(pause.pause_end))]
        self.assertEqual(set(after["phase"]), {"Postpartum"})
        self.assertEqual(pk.status, "Completed")
        out = O[O.package_id == pause.package_id].iloc[0]
        self.assertEqual((out.renewed, out.reason_category), ("No", "Life stage change"))

    def test_second_prenatal_client_renews(self):
        out = O[(O.client_id == "C002")].iloc[0]
        self.assertEqual(out.renewed, "Yes")
        self.assertEqual(len(P[P.client_id == "C002"]), 2)

    def test_clients_who_stopped_attending_stay_open_with_no_outcome(self):
        quiet = M[(M["status"] != "Completed") & (M["days_since_last_session"] >= 60)]
        self.assertGreaterEqual(len(quiet), 2)
        self.assertTrue(set(quiet["status"]) <= {"Active", "Paused"})
        self.assertFalse(set(quiet["package_id"]) & set(O["package_id"]))

    def test_no_lapse_no_expiry(self):
        self.assertTrue(set(P["status"]) <= {"Active", "Completed", "Paused"})
        self.assertFalse({"original_expiry", "current_expiry"} & set(P.columns))
        self.assertEqual(set(O["outcome"]), {"Completed"})
        self.assertTrue((M["status"] == M["status_derived"]).all())

    def test_all_pause_types_present(self):
        self.assertIn("Maternity", set(Z.reason_category))
        self.assertGreaterEqual(len(Z[Z.reason_category != "Maternity"]), 2)


class TestMeasurements(unittest.TestCase):
    ME = T["measurements"]

    def test_every_client_with_attended_sessions_has_body_composition(self):
        att = set(S.loc[S.outcome == "Attended", "client_id"])
        for name in ("Body weight", "Body fat percentage", "Skeletal muscle mass"):
            self.assertEqual(set(self.ME.loc[self.ME.measure_name == name, "client_id"]), att, name)

    def test_units(self):
        u = self.ME.groupby("measure_name")["unit"].first()
        self.assertEqual((u["Body weight"], u["Body fat percentage"], u["Skeletal muscle mass"]), ("kg", "%", "kg"))

    def test_prenatal_weight_rises_before_the_pause(self):
        w = self.ME[(self.ME.client_id == "C001") & (self.ME.measure_name == "Body weight")].sort_values("date")
        self.assertGreater(w["value"].iloc[2], w["value"].iloc[0])


class TestWearables(unittest.TestCase):
    def test_synthetic_wearables_cover_whole_history_with_gaps(self):
        w = build_daily_activity(T, CFG)
        self.assertGreater(w["Id"].nunique(), 25)
        self.assertEqual(set(w["source"]), {"synthetic"})
        span = (pd.Timestamp(CFG.as_of) - pd.to_datetime(C.set_index("client_id")["join_date"])).dt.days + 1
        share = w.groupby("Id").size() / span.reindex(w.groupby("Id").size().index)
        self.assertTrue((share < 1).all() and (share > 0.2).all())      # tracker not worn every day
        self.assertLess(pd.to_datetime(w["ActivityDate"]).min(), pd.Timestamp(CFG.as_of) - pd.Timedelta(days=200))

    def test_real_fitbit_path_maps_series_to_recent_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "dailyActivity_merged.csv"
            rows = [dict(Id=111 + k // 10, ActivityDate=f"4/{(k % 10) + 1}/2016", TotalSteps=5000 + k,
                         VeryActiveMinutes=10, FairlyActiveMinutes=10, LightlyActiveMinutes=100,
                         SedentaryMinutes=800, Calories=2000) for k in range(20)]
            pd.DataFrame(rows).to_csv(f, index=False)
            w = build_daily_activity(T, CFG, fitbit_path=f)
            self.assertEqual(set(w["source"]), {"fitbit_shifted"})
            self.assertEqual(pd.to_datetime(w["ActivityDate"]).max().date(), CFG.as_of)
            self.assertEqual(w["Id"].nunique(), 35)
            pairing = w.groupby("Id")["fitbit_id"].nunique()
            self.assertTrue((pairing == 1).all())          # one Fitbit user per client
            self.assertEqual(w["fitbit_id"].nunique(), 2)  # the stand-in file has 2 users


if __name__ == "__main__":
    unittest.main()
