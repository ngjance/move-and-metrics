"""Outside-the-gym activity features over a client's most recent 30 days (matches the length of the Fitbit file).

The same function is used (a) on past packages, with the 30 days BEFORE the decision / snapshot date, and (b) on open
packages, with the 30 days before the as-of date. Only the window up to the reference date is ever read, so nothing leaks.
Features:
  steps_30         average daily steps on days with data
  active_min_30    average daily active minutes (very + fairly active)
  wear_30          share of days in the window with any Fitbit data ("checking in"; no data = not worn or not synced)
  steps_trend_30   average steps in the last half of the window vs the first half
"""
import numpy as np
import pandas as pd

WINDOW = 30
KEYS = ["steps_30", "active_min_30", "wear_30", "steps_trend_30"]


def window_features(daily_client, ref, since=None, days=WINDOW):
    """daily_client: rows for ONE client (ActivityDate as Timestamp). ref: reference date. since: earliest day data can exist."""
    ref = pd.Timestamp(ref)
    start = ref - pd.Timedelta(days=days - 1)
    if since is not None:
        start = max(start, pd.Timestamp(since))
    n_days = max(1, (ref - start).days + 1)
    if daily_client is None:
        return dict(steps_30=np.nan, active_min_30=np.nan, wear_30=np.nan, steps_trend_30=np.nan)
    w = daily_client[(daily_client["ActivityDate"] >= start) & (daily_client["ActivityDate"] <= ref)]
    out = dict(steps_30=float(w["TotalSteps"].mean()) if len(w) else np.nan,
               active_min_30=float((w["VeryActiveMinutes"] + w["FairlyActiveMinutes"]).mean()) if len(w) else np.nan,
               wear_30=len(w) / n_days, steps_trend_30=np.nan)
    mid = start + pd.Timedelta(days=n_days // 2)
    a, b = w.loc[w["ActivityDate"] < mid, "TotalSteps"], w.loc[w["ActivityDate"] >= mid, "TotalSteps"]
    if len(a) >= 5 and len(b) >= 5 and a.mean() > 0:
        out["steps_trend_30"] = float((b.mean() - a.mean()) / a.mean())
    return out
