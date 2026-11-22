"""Renewal risk for open personal-training packages (piece 3, on the linked synthetic tables).

The risk score comes from a MODEL LEARNED FROM PAST PACKAGES (logistic regression), not from hand-set rules.
The score is the model's estimated chance that a client does not renew; the dashboard shows the main drivers.
For tests of specific questions (do long pauses or skipped sessions go with non-renewal?) see pt_analysis.py.

No leakage: for each past package, features are computed only from data up to a SNAPSHOT date (when half the
package is used, or halfway through its planned schedule, whichever is first), and the label is whether the
client later renewed. Packages have NO expiry date and there is NO lapse rule, so a client who simply stops
attending never gets a label; those packages are not in the training data and show up as open packages with a
long gap since their last session. Pending renewals and "Life stage change" non-renewals (for example the
maternity case) are excluded from labels here: they are not churn.
The model is trained on 10 other synthetic cohorts (other random seeds); the dashboard's 35-client cohort
is held out. All data is synthetic, so results show a method, not real retention drivers.
"""
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .pt_config import PTConfig
from .pt_synthetic import generate
from .pt_wearable_features import KEYS as WEARABLE, window_features
from .wearables import build_daily_activity

DATE_COLS = {"clients": ["join_date"], "packages": ["purchase_date", "start_date"],
             "pauses": ["pause_start", "pause_end"], "sessions": ["date"], "outcomes": ["end_date"],
             "measurements": ["date"], "daily_activity": ["ActivityDate"]}
GOALS = ["Fat loss", "Strength", "Prenatal", "Postpartum", "Longevity"]
# used_pct and days_since_start are NOT model inputs: the training snapshot fixes used_pct near 50%,
# so standardising it would explode on packages at other stages. They are still shown on the dashboard.
NUMERIC = ["pace_vs_plan_overall", "pace_vs_plan_28", "days_since_last",
           "forfeit_rate_all", "early_rate_all", "forfeit_rate_28", "early_rate_28", "weeks_to_finish",
           "had_pause", "package_no", "size", "planned_per_week", "age"]
FILL = {"steps_30": 7000.0, "active_min_30": 25.0, "wear_30": 0.8, "steps_trend_30": 0.0}   # used only when a value is missing
LABELS = {
    "steps_30": "daily steps over the last 30 days", "active_min_30": "active minutes per day, last 30 days",
    "wear_30": "days with Fitbit data in the last 30 days", "steps_trend_30": "step trend within the last 30 days",
    "used_pct": "package already heavily used", "days_since_start": "long time since the package started",
    "pace_vs_plan_overall": "low attendance vs plan overall", "pace_vs_plan_28": "low attendance vs plan (last 4 weeks)",
    "days_since_last": "long gap since the last session", "forfeit_rate_all": "many late cancels / no-shows",
    "early_rate_all": "many early cancels", "forfeit_rate_28": "recent late cancels / no-shows",
    "early_rate_28": "recent early cancels", "weeks_to_finish": "slow pace: many weeks to use the remaining sessions",
    "had_pause": "package was paused", "package_no": "number of packages so far",
    "size": "package size", "planned_per_week": "planned sessions per week", "age": "age"}


# ----------------------------------------------------------------------------- data helpers
def normalize(tables):
    out = {}
    for k, df in tables.items():
        df = df.copy()
        for c in DATE_COLS.get(k, []):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        out[k] = df
    return out


def load_tables(folder):
    folder = Path(folder)
    names = ["clients", "packages", "pauses", "sessions", "outcomes", "measurements", "daily_activity"]
    return normalize({n: pd.read_csv(folder / f"{n}.csv") for n in names if (folder / f"{n}.csv").exists()})


def _pause_days(pauses, lo, hi):
    total = 0
    for _, z in pauses.iterrows():
        pe = z["pause_end"] if pd.notna(z["pause_end"]) else hi
        a, b = max(z["pause_start"], lo), min(pe, hi)
        if b > a:
            total += (b - a).days
    return total


# ----------------------------------------------------------------------------- features
def package_features(pkg, sessions, pauses, client, snap, cfg: PTConfig, daily=None):
    s = sessions[(sessions["package_id"] == pkg["package_id"]) & (sessions["date"] <= snap)]
    z = pauses[(pauses["package_id"] == pkg["package_id"]) & (pauses["pause_start"] <= snap)]
    att = s[s["outcome"] == "Attended"]
    late, nosh, early = (int((s["outcome"] == o).sum()) for o in ("Late cancel", "No-show", "Early cancel"))
    forfeited = late * int(cfg.late_cancel_forfeits) + nosh * int(cfg.no_show_forfeits)
    used = len(att) + forfeited
    size = int(pkg["sessions_signed"])
    start = pkg["start_date"]
    days_since_start = max(0, (snap - start).days)
    active_days = max(1, days_since_start - _pause_days(z, start, snap))
    planned = float(pkg["planned_per_week"])
    pace_all = len(att) / (active_days / 7)
    last_att = att["date"].max() if len(att) else None
    days_since_last = (snap - last_att).days if last_att is not None else days_since_start
    w0 = max(snap - pd.Timedelta(days=27), start)
    w = s[s["date"] >= w0]
    w_days = (snap - w0).days + 1
    w_active = max(1, w_days - _pause_days(z, w0, snap))
    n28 = len(w)
    att28 = int((w["outcome"] == "Attended").sum())
    pace28 = att28 / (w_active / 7) if w_active >= 14 else pace_all
    forfeit28 = int(w["outcome"].isin(["Late cancel", "No-show"]).sum())
    early28 = int((w["outcome"] == "Early cancel").sum())
    left = size - used
    weeks_to_finish = float(np.clip(left / max(pace28, 0.1), 0, 104)) if left > 0 else 0.0   # no expiry dates
    paused_now = bool(((z["pause_end"].isna()) | (z["pause_end"] > snap)).any()) if len(z) else False
    dc = None if daily is None else daily[daily["Id"] == pkg["client_id"]]
    act = window_features(dc, snap, since=max(pd.Timestamp(client["join_date"]), snap - pd.Timedelta(days=29)))
    return dict(**act,
        used_pct=used / size, days_since_start=days_since_start,
        pace_vs_plan_overall=pace_all / planned, pace_vs_plan_28=pace28 / planned,
        days_since_last=days_since_last,
        forfeit_rate_all=forfeited / len(s) if len(s) else 0.0, early_rate_all=early / len(s) if len(s) else 0.0,
        forfeit_rate_28=forfeit28 / n28 if n28 else 0.0, early_rate_28=early28 / n28 if n28 else 0.0,
        weeks_to_finish=weeks_to_finish, had_pause=int(len(z) > 0), package_no=int(pkg["package_no"]), size=size,
        planned_per_week=planned, age=int(client["age"]), goal=client["goal"],
        sessions_left=left, paused_now=paused_now, snapshot=snap)


def _snapshot_date(pkg, sess, cfg: PTConfig):
    s = sess[sess["package_id"] == pkg["package_id"]].sort_values("date")
    weight = s["outcome"].map({"Attended": 1, "Late cancel": int(cfg.late_cancel_forfeits),
                               "No-show": int(cfg.no_show_forfeits)}).fillna(0)
    cum = weight.cumsum() / pkg["sessions_signed"]
    hit = s.loc[cum >= cfg.snapshot_used_pct, "date"]
    planned_days = pkg["sessions_signed"] / pkg["planned_per_week"] * 7
    mid = pkg["start_date"] + pd.Timedelta(days=planned_days / 2)    # halfway through the planned schedule
    snap = min(hit.iloc[0], mid) if len(hit) else mid
    return min(max(snap, pkg["start_date"] + pd.Timedelta(days=14)), pd.Timestamp(cfg.as_of))


def training_frame(tables, cfg: PTConfig):
    cl = tables["clients"].set_index("client_id")
    rows = []
    for _, pkg in tables["packages"].iterrows():
        out = tables["outcomes"][tables["outcomes"]["package_id"] == pkg["package_id"]]
        if out.empty:
            continue
        o = out.iloc[0]
        if o["renewed"] == "Pending" or o["reason_category"] == "Life stage change":
            continue
        snap = _snapshot_date(pkg, tables["sessions"], cfg)
        f = package_features(pkg, tables["sessions"], tables["pauses"], cl.loc[pkg["client_id"]], snap, cfg,
                             tables.get("daily_activity"))
        f.update(package_id=pkg["package_id"], client_id=pkg["client_id"], not_renewed=int(o["renewed"] == "No"))
        rows.append(f)
    return pd.DataFrame(rows)


def open_frame(tables, cfg: PTConfig):
    as_of = pd.Timestamp(cfg.as_of)
    cl = tables["clients"].set_index("client_id")
    out_ids = set(tables["outcomes"]["package_id"])
    pend = set(tables["outcomes"].loc[tables["outcomes"]["renewed"] == "Pending", "package_id"])
    rows = []
    for _, pkg in tables["packages"].iterrows():
        state = {"Active": "Open", "Paused": "Paused"}.get(pkg["status"])
        if pkg["package_id"] in pend:
            state = "Renewal pending"
        if state is None:
            continue
        f = package_features(pkg, tables["sessions"], tables["pauses"], cl.loc[pkg["client_id"]], as_of, cfg,
                             tables.get("daily_activity"))
        z = tables["pauses"][(tables["pauses"]["package_id"] == pkg["package_id"]) & tables["pauses"]["pause_end"].isna()]
        f.update(package_id=pkg["package_id"], client_id=pkg["client_id"], state=state,
                 pause_reason=z["reason_category"].iloc[0] if len(z) else "")
        rows.append(f)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- model
def to_matrix(df, wearable=False):
    X = df[NUMERIC].astype(float).copy()
    if wearable:
        for k in WEARABLE:
            X[k] = df[k].astype(float).fillna(FILL[k])
    for g in GOALS:
        X[f"goal_{g}"] = (df["goal"] == g).astype(float)
    return X


def build_training_data(cfg: PTConfig, seeds=range(1, 11)):
    frames = []
    for sd in seeds:
        c = replace(cfg, seed=sd)
        raw = generate(c)
        raw["daily_activity"] = build_daily_activity(raw, c)
        f = training_frame(normalize(raw), c)
        f["cohort"] = sd
        f["group"] = f["cohort"].astype(str) + "-" + f["client_id"]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def train_model(cfg: PTConfig = None, seeds=range(1, 11), use_wearable=True, data=None):
    cfg = cfg or PTConfig()
    df = data if data is not None else build_training_data(cfg, seeds)
    X, y, g = to_matrix(df, use_wearable), df["not_renewed"], df["group"]
    make = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        if y.iloc[te].nunique() == 2:
            aucs.append(roc_auc_score(y.iloc[te], make().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]))
    bounds = (X.quantile(0.01), X.quantile(0.99))
    pipe = make().fit(X.clip(bounds[0], bounds[1], axis=1), y)
    raw = generate(cfg)
    raw["daily_activity"] = build_daily_activity(raw, cfg)
    hold = training_frame(normalize(raw), cfg)
    hold_auc = None
    if hold["not_renewed"].nunique() == 2:
        hold_auc = float(roc_auc_score(hold["not_renewed"], pipe.predict_proba(to_matrix(hold, use_wearable)[X.columns].clip(bounds[0], bounds[1], axis=1))[:, 1]))
    metrics = dict(uses_activity_last_30_days=bool(use_wearable), n_train=int(len(df)), train_non_renewal_rate=round(float(y.mean()), 3),
                   grouped_cv_auc_mean=round(float(np.mean(aucs)), 3), grouped_cv_auc_sd=round(float(np.std(aucs)), 3),
                   heldout_cohort_n=int(len(hold)), heldout_cohort_auc=None if hold_auc is None else round(hold_auc, 3),
                   note="Synthetic data: shows the method, not real retention drivers.")
    coefs = pd.DataFrame({"feature": X.columns, "coefficient": pipe[-1].coef_[0]}).sort_values(
        "coefficient", key=abs, ascending=False)
    return pipe, list(X.columns), bounds, metrics, coefs


def _fmt(name, v):
    if name in ("used_pct", "forfeit_rate_all", "early_rate_all", "forfeit_rate_28", "early_rate_28",
                "pace_vs_plan_overall", "pace_vs_plan_28"):
        return f"{v:.0%}"
    return f"{v:.0f}"


def _fmt_feature(name, v):
    if name == "steps_30":
        return f"{v:,.0f} steps"
    if name == "active_min_30":
        return f"{v:.0f} min"
    if name in ("wear_30", "steps_trend_30"):
        return f"{v:.0%}"
    if name == "weeks_to_finish":
        return f"{int(v)} weeks at the recent pace"
    return _fmt(name, v)


def score(pipe, cols, bounds, frame, top=3):
    X = to_matrix(frame, wearable=any(c in cols for c in WEARABLE))[cols].clip(bounds[0], bounds[1], axis=1)   # keep inputs inside the training range
    prob = pipe.predict_proba(X)[:, 1]
    contrib = pipe[0].transform(X) * pipe[-1].coef_[0]
    reasons = []
    for i in range(len(X)):
        order = np.argsort(-contrib[i])[:top]
        reasons.append("; ".join(f"{LABELS[cols[j]]} ({_fmt_feature(cols[j], X.iloc[i][cols[j]])})"
                                 for j in order if cols[j] in LABELS and contrib[i][j] > 0.15) or "no strong risk drivers")
    return prob, reasons


def risk_table(tables, cfg: PTConfig, model=None):
    pipe, cols, bounds = (model or train_model(cfg)[:3])
    fr = open_frame(tables, cfg)
    if fr.empty:
        return fr
    prob, reasons = score(pipe, cols, bounds, fr)
    rows = []
    for i, f in fr.iterrows():
        too_early = f["days_since_start"] < cfg.min_days_to_score or f["used_pct"] < cfg.min_used_pct_to_score
        p, drivers, note = prob[i], reasons[i], ""
        if f["state"] == "Paused":
            priority, p, drivers = "Paused", np.nan, ""
            note = ("Maternity pause: life stage, not churn. Plan the return." if f["pause_reason"] == "Maternity"
                    else f"Paused ({f['pause_reason'] or 'reason not recorded'}). Check in before the return date.")
        elif f["state"] == "Renewal pending":
            priority, p, drivers = "Renewal pending", np.nan, ""
            note = "Package finished recently: have the renewal conversation."
        elif too_early:
            priority, p, drivers = "Too early", np.nan, ""
            note = "Package started recently: not enough history to score."
        else:
            priority = "High" if p >= cfg.model_high else "Watch" if p >= cfg.model_watch else "OK"
        rows.append(dict(client_id=f["client_id"], goal=f["goal"], age=f["age"], package_id=f["package_id"],
                         package_size=f["size"], state=f["state"], sessions_left=int(f["sessions_left"]),
                         used_pct=f["used_pct"], days_since_last_session=int(f["days_since_last"]),
                         pace_vs_plan_4wk=f["pace_vs_plan_28"], avg_steps_30=f["steps_30"], fitbit_days_30=f["wear_30"],
                         priority=priority, model_risk=p,
                         model_drivers=drivers, note=note))
    order = {"High": 0, "Watch": 1, "Renewal pending": 2, "Paused": 3, "Too early": 4, "OK": 5}
    out = pd.DataFrame(rows)
    return out.sort_values(["priority", "model_risk"], key=lambda s: s.map(order) if s.name == "priority" else -s.fillna(-1)
                           ).reset_index(drop=True)


def compare_activity_value(cfg: PTConfig = None, repeats=5, data=None):
    """Does adding the last-30-days activity features improve prediction beyond gym attendance? Grouped CV AUC with vs without."""
    cfg = cfg or PTConfig()
    df = data if data is not None else build_training_data(cfg)
    y, g = df["not_renewed"], df["group"]
    rows = []
    for name, wear in (("Gym attendance only", False), ("Gym attendance + activity, last 30 days", True)):
        X, scores = to_matrix(df, wear), []
        for rep_i in range(repeats):
            rng = np.random.default_rng(rep_i)
            perm = dict(zip(g.unique(), rng.permutation(g.nunique())))
            for tr, te in GroupKFold(5).split(X, y, g.map(perm)):
                lo, hi = X.iloc[tr].quantile(.01), X.iloc[tr].quantile(.99)
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(
                    X.iloc[tr].clip(lo, hi, axis=1), y.iloc[tr])
                scores.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te].clip(lo, hi, axis=1))[:, 1]))
        rows.append(dict(model=name, features=X.shape[1], cv_auc=round(float(np.mean(scores)), 3),
                         spread=round(float(np.std(scores)), 3), packages=len(df)))
    return pd.DataFrame(rows)


def main():
    cfg = PTConfig()
    out_dir = config.OUTPUT_DIR / "pt_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe, cols, bounds, metrics, coefs = train_model(cfg)
    joblib.dump((pipe, cols, bounds), out_dir / "renewal_model.joblib")
    coefs.to_csv(out_dir / "coefficients.csv", index=False)
    import json
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    folder = config.ROOT / "data" / "synthetic"
    tables = load_tables(folder) if (folder / "packages.csv").exists() else normalize(generate(cfg))
    table = risk_table(tables, cfg, (pipe, cols, bounds))
    table.to_csv(folder / "renewal_risk.csv", index=False) if folder.exists() else None
    print(json.dumps(metrics, indent=2))
    print(table[["client_id", "goal", "state", "sessions_left", "days_since_last_session", "priority", "model_risk"]].to_string(index=False))


if __name__ == "__main__":
    main()
