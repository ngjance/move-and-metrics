"""Daily activity for the synthetic clients, in the same columns as the Fitbit file.

Default: SYNTHETIC steps for each client's WHOLE history (or the last `days` days if given), higher on session days and lower
when a client has gone more than 3 weeks without an attended session. A day with no row means the tracker was not worn or
not synced; engaged clients wear it more often. Both steps and wear depend on the client's hidden engagement level (the
same variable that drives attendance and renewal), so any link between wearable data and renewal here comes only through
engagement; there is no direct effect built in. If you pass the real Kaggle Fitbit file, each client instead gets one
Fitbit user's real series, mapped (shifted in time) onto the last N days up to the as-of date.
Wearable data therefore exists only for that recent window. Disclose this in your README.
Assignment is deterministic, not random: client i gets Fitbit user (i mod number of users), so with
35 clients and 30 Fitbit users, 5 clients share a series. The `fitbit_id` column records the pairing;
when you split data for modelling, group by `fitbit_id` so the same series never lands in both train and test.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from .data_loading import load_fitbit_daily
from .pt_config import PTConfig

BASE_STEPS = {"Fat loss": 7500, "Strength": 8500, "Prenatal": 6000, "Postpartum": 5500, "Longevity": 6500}
COLS = ["Id", "ActivityDate", "TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes",
        "LightlyActiveMinutes", "SedentaryMinutes", "Calories"]


def build_daily_activity(tables, cfg: PTConfig = None, days: int = None, fitbit_path=None):
    cfg = cfg or PTConfig()
    rng = np.random.default_rng(cfg.seed + 1)
    clients, sessions, truth = tables["clients"], tables["sessions"], tables["truth"].set_index("client_id")
    att = sessions[sessions["outcome"] == "Attended"]
    rows = []

    if fitbit_path is not None:
        fb = load_fitbit_daily(fitbit_path).sort_values(["Id", "ActivityDate"])
        ids = sorted(fb["Id"].unique())
        for i, cid in enumerate(clients["client_id"]):
            series = fb[fb["Id"] == ids[i % len(ids)]].reset_index(drop=True)
            n = len(series)
            dates = [cfg.as_of - timedelta(days=n - 1 - k) for k in range(n)]
            part = series[COLS].copy()
            part["fitbit_id"] = ids[i % len(ids)]   # keep the original Fitbit user (needed to avoid leakage)
            part["Id"], part["ActivityDate"] = cid, pd.to_datetime(dates)
            part["source"] = "fitbit_shifted"
            rows.append(part)
        return pd.concat(rows, ignore_index=True)

    from bisect import bisect_right
    for _, c in clients.iterrows():
        cid = c["client_id"]
        e = truth.loc[cid, "hidden_engagement"]
        sdays = sorted(set(att.loc[att["client_id"] == cid, "date"]))
        if not sdays:
            continue
        sset = set(sdays)
        join = pd.Timestamp(c["join_date"]).date()
        first_day = join if days is None else max(join, cfg.as_of - timedelta(days=days - 1))
        base = BASE_STEPS[c["goal"]] * (0.7 + 0.6 * e)
        wear_p = float(np.clip(0.55 + 0.40 * e, 0.30, 0.97))
        day = first_day
        while day <= cfg.as_of:
            k = bisect_right(sdays, day)
            gap = (day - (sdays[k - 1] if k else join)).days     # days since the latest attended session
            quiet = gap > 21
            if rng.random() < wear_p * (0.7 if quiet else 1.0):  # no row = tracker not worn / not synced
                steps = base + (1500 if day in sset else 0)
                steps = max(0, int(rng.normal(steps * (0.75 if quiet else 1.0), base * 0.2)))
                very = int(steps / 1000 * rng.uniform(1, 3))
                fairly = int(steps / 1000 * rng.uniform(0.5, 2))
                lightly = int(min(400, steps / 40 * rng.uniform(0.8, 1.2)))
                rows.append(dict(Id=cid, ActivityDate=pd.Timestamp(day), TotalSteps=steps,
                                 VeryActiveMinutes=very, FairlyActiveMinutes=fairly, LightlyActiveMinutes=lightly,
                                 SedentaryMinutes=max(0, 1440 - very - fairly - lightly - 420),
                                 Calories=int(1500 + steps * 0.04 + very * 8), source="synthetic", fitbit_id=None))
            day += timedelta(days=1)
    return pd.DataFrame(rows)
