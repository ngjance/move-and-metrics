"""Engagement features and a transparent rule-based status.

The rule-based status is a BASELINE. The ML churn model (churn_model.py) is trained on a
different dataset; do not describe this table as model output.
"""
import numpy as np
import pandas as pd

from . import config


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ActivityDate"] = pd.to_datetime(out["ActivityDate"])
    out = out.sort_values(["Id", "ActivityDate"]).reset_index(drop=True)
    out["ActiveMinutes"] = out["VeryActiveMinutes"] + out["FairlyActiveMinutes"]
    out["IsActiveDay"] = out["TotalSteps"] >= config.ACTIVE_STEP_THRESHOLD
    return out


def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = prepare(df)
    d["Week"] = d["ActivityDate"].dt.to_period("W-SUN").dt.start_time
    return (d.groupby(["Id", "Week"])
              .agg(avg_steps=("TotalSteps", "mean"),
                   active_minutes=("ActiveMinutes", "sum"),
                   active_days=("IsActiveDay", "sum"))
              .reset_index())


def trailing_inactive_streak(flags) -> int:
    streak = 0
    for f in reversed(list(flags)):
        if f:
            break
        streak += 1
    return streak


def _full_calendar(d: pd.DataFrame) -> pd.DataFrame:
    """Reindex every client to the same calendar; missing days count as 0 steps."""
    all_days = pd.date_range(d["ActivityDate"].min(), d["ActivityDate"].max(), freq="D")
    parts = []
    for cid, g in d.groupby("Id"):
        g = g.set_index("ActivityDate").reindex(all_days)
        g["Id"] = cid
        g["TotalSteps"] = g["TotalSteps"].fillna(0)
        g["IsActiveDay"] = g["TotalSteps"] >= config.ACTIVE_STEP_THRESHOLD
        g.index.name = "ActivityDate"
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def client_status(df: pd.DataFrame) -> pd.DataFrame:
    d = _full_calendar(prepare(df))
    w = config.COMPARE_WINDOW_DAYS
    rows = []
    for cid, g in d.groupby("Id"):
        g = g.sort_values("ActivityDate")
        steps = g["TotalSteps"].to_numpy()
        flags = g["IsActiveDay"].to_numpy()
        streak = trailing_inactive_streak(flags)
        recent = steps[-w:].mean() if len(steps) >= 1 else np.nan
        prior = steps[-2 * w:-w].mean() if len(steps) >= 2 * w else np.nan
        change = (recent - prior) / prior if prior and prior > 0 else np.nan
        active_last7 = int(flags[-7:].sum())

        reasons = []
        status = "OK"
        if streak >= config.STREAK_DAYS_HIGH:
            status = "High"
            reasons.append(f"{streak} inactive days in a row")
        if not np.isnan(change) and change <= -config.DROP_PCT_HIGH:
            status = "High"
            reasons.append(f"steps down {abs(change):.0%} vs prior {w} days")
        if status != "High":
            if not np.isnan(change) and change <= -config.DROP_PCT_HIGH / 2:
                status = "Watch"
                reasons.append(f"steps down {abs(change):.0%} vs prior {w} days")
            if active_last7 <= config.LOW_ACTIVE_DAYS_WATCH:
                status = "Watch"
                reasons.append(f"only {active_last7} active days in last 7")
        rows.append({
            "Id": cid, "status": status, "inactive_streak_days": streak,
            "active_days_last7": active_last7,
            "avg_steps_recent": round(float(recent), 0) if not np.isnan(recent) else np.nan,
            "change_vs_prior": change, "reasons": "; ".join(reasons) or "on track",
        })
    order = {"High": 0, "Watch": 1, "OK": 2}
    out = pd.DataFrame(rows)
    return out.sort_values("status", key=lambda s: s.map(order)).reset_index(drop=True)
