"""Package-level metrics in pandas. The Excel workbook computes the same things with formulas;
tests check that the two agree."""
import pandas as pd

from .pt_config import PTConfig


def package_metrics(packages, sessions, pauses, cfg: PTConfig = None):
    cfg = cfg or PTConfig()
    s = sessions
    att = s[s["outcome"] == "Attended"].groupby("package_id")
    out = packages.copy().set_index("package_id")
    out["sessions_attended"] = att.size().reindex(out.index).fillna(0).astype(int)
    late = s[s["outcome"] == "Late cancel"].groupby("package_id").size().reindex(out.index).fillna(0)
    nosh = s[s["outcome"] == "No-show"].groupby("package_id").size().reindex(out.index).fillna(0)
    out["sessions_forfeited"] = (late * int(cfg.late_cancel_forfeits) + nosh * int(cfg.no_show_forfeits)).astype(int)
    out["sessions_left"] = out["sessions_signed"] - out["sessions_attended"] - out["sessions_forfeited"]
    out["package_used_pct"] = (out["sessions_attended"] + out["sessions_forfeited"]) / out["sessions_signed"]
    done = pauses[pauses["pause_end"].notna()].copy()
    done["days"] = (pd.to_datetime(done["pause_end"]) - pd.to_datetime(done["pause_start"])).dt.days
    out["days_paused"] = done.groupby("package_id")["days"].sum().reindex(out.index).fillna(0).astype(int)
    out["first_session"] = att["date"].min().reindex(out.index)
    out["last_session"] = att["date"].max().reindex(out.index)
    span = (pd.to_datetime(out["last_session"]) - pd.to_datetime(out["first_session"])).dt.days
    out["weeks_active_excl_pause"] = (span - out["days_paused"]).clip(lower=1) / 7
    out.loc[out["sessions_attended"] == 0, "weeks_active_excl_pause"] = float("nan")
    out["sessions_per_week_actual"] = out["sessions_attended"] / out["weeks_active_excl_pause"]
    out["package_value"] = out["sessions_signed"] * out["price_per_session"]
    as_of = pd.Timestamp(cfg.as_of)
    ref = pd.to_datetime(out["last_session"]).fillna(pd.to_datetime(out["start_date"]))
    out["days_since_last_session"] = (as_of - ref).dt.days
    ongoing = set(pauses.loc[pauses["pause_end"].isna(), "package_id"])
    out["status_derived"] = [
        "Completed" if left <= 0 else "Paused" if pid in ongoing else "Active"
        for pid, left in zip(out.index, out["sessions_left"])]
    return out.reset_index()
