"""What is associated with NOT renewing a package? (exploratory analysis on finished packages)

Questions this answers, with uncertainty:
  * Is a longer pause associated with not renewing?
  * Are frequently skipped sessions associated with not renewing?
plus the same for other factors, and a multivariable model that adjusts them for each other.

Method notes (read before trusting any number):
  * Unit: one finished package with a renewal decision (Yes/No). Open packages and "Pending" renewals have no
    outcome yet and are excluded. There is no lapse rule, so clients who simply stop attending never get an
    outcome and are NOT in the analysis.
  * Activity outside the gym = the 30 days of Fitbit-style data BEFORE the package finished (steps, active minutes, share of days
    with data = "checking in", and the step trend inside the window). Body measures = body weight, body fat % and skeletal
    muscle mass: baseline values, and change from the start of the package to its last check-in (needs 28+ days between them).
  * Skipped session = a booking attempt that was not attended (early cancel, late cancel or no-show).
    skip_rate = skipped / booking attempts. forfeit_rate counts only late cancels and no-shows.
  * Associations only, not causes. Holm correction is applied across the factors tested together.
  * Confidence intervals resample whole CLIENTS (clustered bootstrap), because one client's packages are not independent.
  * Small samples cannot detect real effects. Every row says how many cases it rests on.
On synthetic data the answers reflect how the data was generated, so this validates the METHOD, not real retention drivers.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .pt_wearable_features import window_features

MIN_CASES = 10          # fewer exposed cases than this => "too few cases to judge"
GOALS = ["Strength", "Prenatal", "Postpartum", "Longevity"]       # reference level: Fat loss


# ----------------------------------------------------------------------------- data
BODY = {"weight": "Body weight", "bodyfat": "Body fat percentage", "smm": "Skeletal muscle mass"}


def measurement_features(meas, cid, goal, start, end):
    keys = [f"baseline_{k}" for k in BODY] + [f"delta_{k}" for k in BODY] + ["goal_progress"]
    out = {k: np.nan for k in keys}
    if meas is None:
        return out
    m = meas[meas["client_id"] == cid]
    for key, name in BODY.items():
        g = m[m["measure_name"] == name].sort_values("date")
        if g.empty:
            continue
        pool = g[g["date"] <= start + pd.Timedelta(days=14)]
        base = pool.iloc[-1] if len(pool) else g.iloc[0]
        out[f"baseline_{key}"] = float(base["value"])
        fin = g[g["date"] <= end]
        if len(fin) and (fin.iloc[-1]["date"] - base["date"]).days >= 28:
            out[f"delta_{key}"] = float(fin.iloc[-1]["value"] - base["value"])
    if goal in ("Fat loss", "Postpartum") and not np.isnan(out["delta_bodyfat"]):
        out["goal_progress"] = float(out["delta_bodyfat"] < 0)      # body fat went down
    elif goal in ("Strength", "Longevity") and not np.isnan(out["delta_smm"]):
        out["goal_progress"] = float(out["delta_smm"] > 0)          # skeletal muscle went up
    return out


def analysis_frame(tables, cohort="0"):
    """One row per finished package with a renewal decision, with whole-package exposures."""
    cl = tables["clients"].set_index("client_id")
    pk = tables["packages"].set_index("package_id")
    daily, meas = tables.get("daily_activity"), tables.get("measurements")
    rows = []
    for _, o in tables["outcomes"].iterrows():
        if o["renewed"] not in ("Yes", "No"):
            continue
        pid, cid = o["package_id"], o["client_id"]
        s = tables["sessions"][tables["sessions"]["package_id"] == pid]
        z = tables["pauses"][(tables["pauses"]["package_id"] == pid) & tables["pauses"]["pause_end"].notna()]
        spans = [(pe - ps).days for ps, pe in zip(z["pause_start"], z["pause_end"])]
        n = len(s)
        early, late, nosh = (int((s["outcome"] == k).sum()) for k in ("Early cancel", "Late cancel", "No-show"))
        start, end, join = pd.Timestamp(pk.loc[pid, "start_date"]), pd.Timestamp(o["end_date"]), pd.Timestamp(cl.loc[cid, "join_date"])
        dc = None if daily is None else daily[daily["Id"] == cid]
        row = dict(
            package_id=pid, client_id=cid, cluster=f"{cohort}-{cid}", goal=cl.loc[cid, "goal"], age=int(cl.loc[cid, "age"]),
            package_size=int(pk.loc[pid, "sessions_signed"]), package_no=int(pk.loc[pid, "package_no"]),
            planned_per_week=float(pk.loc[pid, "planned_per_week"]), attempts=n,
            skip_rate=(early + late + nosh) / n if n else 0.0, forfeit_rate=(late + nosh) / n if n else 0.0,
            early_rate=early / n if n else 0.0, had_pause=int(len(spans) > 0), n_pauses=len(spans),
            pause_days=int(sum(spans)), longest_pause=int(max(spans)) if spans else 0,
            maternity_pause=int((z["reason_category"] == "Maternity").any()),
            not_renewed=int(o["renewed"] == "No"), reason_category=o["reason_category"])
        row.update(window_features(dc, end, since=max(join, end - pd.Timedelta(days=29))))
        row.update(measurement_features(meas, cid, row["goal"], start, end))
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables(cfg):
    """Generate one synthetic cohort with wearable history, ready for analysis_frame / pt_risk."""
    from .pt_risk import normalize
    from .pt_synthetic import generate
    from .wearables import build_daily_activity
    raw = generate(cfg)
    raw["daily_activity"] = build_daily_activity(raw, cfg)
    return normalize(raw)


# ----------------------------------------------------------------------------- statistics helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m, running = len(idx), 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def _cluster_boot(df, stat, B, seed):
    rng = np.random.default_rng(seed)
    idx = [np.asarray(v) for v in df.reset_index(drop=True).groupby("cluster").indices.values()]
    base = df.reset_index(drop=True)
    vals = []
    for _ in range(B):
        take = np.concatenate([idx[i] for i in rng.choice(len(idx), size=len(idx), replace=True)])
        v = stat(base.iloc[take])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) > 20 else (np.nan, np.nan)


def _auc(df, col):
    if df["not_renewed"].nunique() < 2 or df[col].nunique() < 2:
        return None
    return roc_auc_score(df["not_renewed"], df[col])


def _riskdiff(df, col):
    e, u = df[df[col] == 1], df[df[col] == 0]
    return None if len(e) == 0 or len(u) == 0 else e["not_renewed"].mean() - u["not_renewed"].mean()


# ----------------------------------------------------------------------------- association table
CORE, ACT, MEAS = "Gym and pauses", "Activity outside the gym (last 30 days)", "Body measurements"
# (column, label, family, kind, how "cases" is counted)
SPECS = [
    ("skip_rate", "Skipped sessions (share of bookings not attended)", CORE, "continuous", "nonzero"),
    ("forfeit_rate", "Late cancels + no-shows (share of bookings)", CORE, "continuous", "nonzero"),
    ("early_rate", "Early cancels (share of bookings)", CORE, "continuous", "nonzero"),
    ("pause_days", "Pause length, all packages (days; 0 if none)", CORE, "continuous", "nonzero"),
    ("package_no", "Package number (1 = first)", CORE, "continuous", "all"),
    ("age", "Age", CORE, "continuous", "all"),
    ("had_pause", "Had any pause", CORE, "binary", None),
    ("maternity_pause", "Had a maternity pause", CORE, "binary", None),
    ("steps_30", "Average daily steps, last 30 days", ACT, "continuous", "all"),
    ("active_min_30", "Active minutes per day, last 30 days", ACT, "continuous", "all"),
    ("wear_30", "Share of days with Fitbit data (checking in), last 30 days", ACT, "continuous", "all"),
    ("steps_trend_30", "Step trend within the last 30 days (second half vs first half)", ACT, "continuous", "all"),
    ("baseline_weight", "Body weight at start of package (kg)", MEAS, "continuous", "all"),
    ("baseline_bodyfat", "Body fat % at start of package", MEAS, "continuous", "all"),
    ("baseline_smm", "Skeletal muscle mass at start of package (kg)", MEAS, "continuous", "all"),
    ("delta_weight", "Change in body weight over the package (kg)", MEAS, "continuous", "all"),
    ("delta_bodyfat", "Change in body fat % over the package (points)", MEAS, "continuous", "all"),
    ("delta_smm", "Change in skeletal muscle mass over the package (kg)", MEAS, "continuous", "all"),
    ("goal_progress", "Moved in the goal direction (body fat down for fat loss/postpartum; muscle up for strength/longevity)",
     MEAS, "binary", None),
]


def _assoc_row(df, col, label, family, kind, count, B, seed):
    d = df.dropna(subset=[col]) if col in df else df.iloc[0:0]
    y = d["not_renewed"] if len(d) else d.get("not_renewed", pd.Series(dtype=float))
    base = dict(factor=label, key=col, family=family, kind=kind, n=len(d))
    if kind == "continuous":
        x = d[col] if len(d) else pd.Series(dtype=float)
        a = _auc(d, col) if len(d) else None
        p = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided").pvalue if a is not None else np.nan
        lo, hi = _cluster_boot(d, lambda b: _auc(b, col), B, seed) if a is not None else (np.nan, np.nan)
        cases = int((x > 0).sum()) if count == "nonzero" else len(d)
        return dict(**base, cases=cases, effect=a, ci_low=lo, ci_high=hi, p_value=p, null=0.5,
                    summary=f"AUC {a:.2f}" if a is not None else "n/a")
    e, u = d[d[col] == 1] if len(d) else d, d[d[col] == 0] if len(d) else d
    eff = _riskdiff(d, col) if len(d) else None
    p = np.nan
    if eff is not None:
        p = stats.fisher_exact([[e.not_renewed.sum(), len(e) - e.not_renewed.sum()],
                                [u.not_renewed.sum(), len(u) - u.not_renewed.sum()]])[1]
    lo, hi = _cluster_boot(d, lambda b: _riskdiff(b, col), B, seed) if eff is not None else (np.nan, np.nan)
    return dict(**base, cases=min(len(e), len(u)), effect=eff, ci_low=lo, ci_high=hi, p_value=p, null=0.0,
                summary=(f"{e.not_renewed.mean():.0%} vs {u.not_renewed.mean():.0%} did not renew ({len(e)} vs {len(u)} packages)")
                if eff is not None else "n/a")


def association_table(df, B=500, seed=1):
    rows = [_assoc_row(df, c, l, f, k, n, B, seed) for c, l, f, k, n in SPECS]
    # pause length among packages that had a pause (answers "does a LONGER pause matter?")
    pz = df[df["had_pause"] == 1]
    a = _auc(pz, "pause_days") if len(pz) >= 4 else None
    p = (stats.mannwhitneyu(pz.loc[pz.not_renewed == 1, "pause_days"], pz.loc[pz.not_renewed == 0, "pause_days"],
                            alternative="two-sided").pvalue if a is not None else np.nan)
    lo, hi = _cluster_boot(pz, lambda b: _auc(b, "pause_days"), B, seed) if a is not None else (np.nan, np.nan)
    rows.insert(4, dict(factor="Pause length, only packages with a pause (days)", key="pause_days_paused_only", family=CORE,
                        kind="continuous", n=len(pz), cases=len(pz), effect=a, ci_low=lo, ci_high=hi, p_value=p, null=0.5,
                        summary=f"AUC {a:.2f}" if a is not None else "n/a"))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for fam in t["family"].unique():                       # Holm correction within each family of tests
        m = t["family"] == fam
        t.loc[m, "p_holm"] = holm(t.loc[m, "p_value"].tolist())
    t["evidence"] = [_label(r) for _, r in t.iterrows()]
    return t


def _label(r):
    if pd.isna(r["effect"]) or r["cases"] < MIN_CASES:
        return f"Too few cases to judge ({int(r['cases'])})"
    excl = (r["ci_low"] > r["null"]) or (r["ci_high"] < r["null"])
    if r["p_holm"] < 0.05 and excl:
        return "More non-renewal" if r["effect"] > r["null"] else "Less non-renewal"
    return "No clear association"


# ----------------------------------------------------------------------------- dose-response
def _thirds(df, col, name, fmt):
    d = df.dropna(subset=[col])
    out = []
    if len(d) < 9:
        return out
    try:
        bins = pd.qcut(d[col], 3, labels=["Lowest third", "Middle third", "Highest third"], duplicates="drop")
    except ValueError:
        return out
    for lab, g in d.groupby(bins, observed=True):
        k, n = int(g.not_renewed.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append(dict(analysis=name, group=str(lab), detail=f"{fmt(g[col].min())} to {fmt(g[col].max())}", packages=n,
                        not_renewed=k, rate=k / n, ci_low=lo, ci_high=hi))
    return out


def dose_response(df):
    out = _thirds(df, "skip_rate", "Skip rate", lambda v: f"{v:.0%}")
    pb = pd.cut(df["pause_days"], [-1, 0, 30, 90, 10_000], labels=["No pause", "1-30 days", "31-90 days", "91+ days"])
    for name, g in df.groupby(pb, observed=True):
        k, n = int(g.not_renewed.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append(dict(analysis="Pause length", group=str(name), detail="", packages=n, not_renewed=k,
                        rate=k / n, ci_low=lo, ci_high=hi))
    out += _thirds(df, "steps_30", "Steps, last 30 days", lambda v: f"{v:,.0f}")
    out += _thirds(df, "wear_30", "Fitbit days with data, last 30 days", lambda v: f"{v:.0%}")
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- multivariable model
def _design(df):
    X = pd.DataFrame({"skip rate (+10 points)": df["skip_rate"] * 10, "pause length (+30 days)": df["pause_days"] / 30,
                      "age (+10 years)": df["age"] / 10, "package number (+1)": df["package_no"].astype(float),
                      "package size 50 (vs 20)": (df["package_size"] == 50).astype(float),
                      "package size 100 (vs 20)": (df["package_size"] == 100).astype(float)})
    for g in GOALS:
        X[f"goal {g} (vs Fat loss)"] = (df["goal"] == g).astype(float)
    return X


def _fit(X, y):
    return LogisticRegression(C=10.0, max_iter=5000).fit(X, y)


def adjusted_model(df, B=300, seed=1):
    """Odds ratios for each factor adjusted for the others, with clustered-bootstrap 95% intervals."""
    X, y = _design(df), df["not_renewed"]
    if y.nunique() < 2:
        return pd.DataFrame(), {}
    m = _fit(X, y)
    est = np.exp(m.coef_[0])
    rng = np.random.default_rng(seed)
    groups = {c: i for c, i in df.reset_index(drop=True).groupby("cluster").groups.items()}
    keys, boots = list(groups), []
    Xr, yr = X.reset_index(drop=True), y.reset_index(drop=True)
    for _ in range(B):
        idx = np.concatenate([groups[keys[i]] for i in rng.choice(len(keys), len(keys), replace=True)])
        if yr.iloc[idx].nunique() < 2:
            continue
        boots.append(np.exp(_fit(Xr.iloc[idx], yr.iloc[idx]).coef_[0]))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0) if len(boots) > 20 else (np.full(len(est), np.nan),) * 2
    tab = pd.DataFrame({"term": X.columns, "odds_ratio": est, "ci_low": lo, "ci_high": hi})
    tab["ci_excludes_1"] = (tab["ci_low"] > 1) | (tab["ci_high"] < 1)
    aucs = []
    for tr, te in GroupKFold(n_splits=min(5, df["cluster"].nunique())).split(X, y, df["cluster"]):
        if y.iloc[te].nunique() == 2 and y.iloc[tr].nunique() == 2:
            aucs.append(roc_auc_score(y.iloc[te], _fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]))
    info = dict(n=len(df), events=int(y.sum()), clients=int(df["cluster"].nunique()),
                grouped_cv_auc=float(np.mean(aucs)) if aucs else np.nan)
    return tab, info


EXTRAS = ["steps_30", "active_min_30", "wear_30", "steps_trend_30", "baseline_weight", "baseline_bodyfat", "baseline_smm",
          "delta_weight", "delta_bodyfat", "delta_smm", "goal_progress"]


def adjusted_extras(df, B=200, seed=1):
    """Does each extra factor add anything BEYOND skipped sessions? One logistic model per factor: skip rate + the factor."""
    rows = []
    for col in EXTRAS:
        d = df.dropna(subset=[col, "skip_rate"]).reset_index(drop=True)
        if len(d) < 30 or d["not_renewed"].nunique() < 2 or d[col].nunique() < 2:
            continue
        sd = d[col].std() or 1.0
        X = pd.DataFrame({"skip": d["skip_rate"] * 10, "extra": (d[col] - d[col].mean()) / sd})
        y = d["not_renewed"]
        m = _fit(X, y)
        rng = np.random.default_rng(seed)
        idx = [np.asarray(v) for v in d.groupby("cluster").indices.values()]
        boots = []
        for _ in range(B):
            take = np.concatenate([idx[i] for i in rng.choice(len(idx), len(idx), replace=True)])
            if y.iloc[take].nunique() == 2:
                boots.append(np.exp(_fit(X.iloc[take], y.iloc[take]).coef_[0]))
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0) if len(boots) > 20 else (np.full(2, np.nan),) * 2
        label = next(l for c, l, *_ in SPECS if c == col)
        rows.append(dict(factor=label, key=col, n=len(d), or_per_sd=float(np.exp(m.coef_[0][1])), ci_low=float(lo[1]),
                         ci_high=float(hi[1]), or_skip_rate_per_10pts=float(np.exp(m.coef_[0][0])),
                         adds_beyond_skip_rate=bool(lo[1] > 1 or hi[1] < 1)))
    return pd.DataFrame(rows)


def analyze(df, B=500, seed=1):
    assoc = association_table(df, B=B, seed=seed)
    adj, info = adjusted_model(df, B=max(100, B // 2), seed=seed)
    return dict(association=assoc, dose=dose_response(df), adjusted=adj, info=info,
                extras=adjusted_extras(df, B=max(100, B // 2), seed=seed),
                n=len(df), events=int(df["not_renewed"].sum()), clients=int(df["cluster"].nunique()))


# ----------------------------------------------------------------------------- validation, charts, report
def validate(trials=8, cohorts=8, penalties=(0.0, 0.15, 0.30), B=150):
    """Can the method detect a PLANTED pause effect, and how often does it flag one when none exists?"""
    from dataclasses import replace

    from .pt_config import PTConfig
    from .pt_risk import normalize
    from .pt_synthetic import generate
    rows = []
    for pen in penalties:
        flagged, paused = 0, []
        for r in range(trials):
            frames = [analysis_frame(normalize(generate(replace(PTConfig(share_random_pause=0.30),
                                                                 seed=1000 * (r + 1) + k,
                                                                 renewal_pause_penalty_per_30d=pen))), f"{r}-{k}")
                      for k in range(cohorts)]
            df = pd.concat(frames, ignore_index=True)
            row = association_table(df, B=B, seed=r).query("key == 'pause_days'").iloc[0]
            flagged += row["evidence"] in ("More non-renewal", "Less non-renewal")
            paused.append(int(row["cases"]))
        rows.append(dict(planted_pause_effect=pen, trials=trials, cohorts_per_trial=cohorts, flagged=flagged,
                         avg_packages_with_a_pause=float(np.mean(paused))))
    return pd.DataFrame(rows)


def _md(df, cols, names=None):
    names = names or cols
    out = ["| " + " | ".join(names) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(r[c], float) and np.isnan(r[c])) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def charts(res, out_dir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = res["dose"]
    for analysis, fname, title in (("Skip rate", f"skip_rate_{tag}.png", "Not renewing, by share of skipped sessions"),
                                   ("Pause length", f"pause_length_{tag}.png", "Not renewing, by pause length"),
                                   ("Steps, last 30 days", f"steps_30_{tag}.png", "Not renewing, by daily steps (last 30 days)"),
                                   ("Fitbit days with data, last 30 days", f"checkins_30_{tag}.png",
                                    "Not renewing, by share of days with Fitbit data (last 30 days)")):
        g = d[d["analysis"] == analysis]
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.6))
        y = g["rate"].to_numpy()
        err = np.vstack([y - g["ci_low"].to_numpy(), g["ci_high"].to_numpy() - y])
        ax.bar(g["group"], y, yerr=err, capsize=4, color="#5B7DB1")
        for i, (r, n) in enumerate(zip(y, g["packages"])):
            ax.text(i, 0.02, f"n={n}", ha="center", color="white", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share who did not renew")
        ax.set_title(title + f" ({tag})", fontsize=10)
        ax.axhline(res["events"] / res["n"], ls="--", color="grey", lw=1)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)
    a = res["adjusted"]
    if len(a):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ypos = np.arange(len(a))[::-1]
        ax.errorbar(a["odds_ratio"], ypos, xerr=[a["odds_ratio"] - a["ci_low"], a["ci_high"] - a["odds_ratio"]],
                    fmt="o", color="#1F3864", capsize=3)
        ax.axvline(1, color="grey", ls="--")
        ax.set_xscale("log")
        ax.set_yticks(ypos)
        ax.set_yticklabels(a["term"], fontsize=8)
        ax.set_xlabel("Odds ratio of NOT renewing (log scale; 1 = no effect)")
        ax.set_title(f"Adjusted model ({tag})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"adjusted_{tag}.png", dpi=130)
        plt.close(fig)


def _sentence(assoc, key, what_more, what_less):
    r = assoc[assoc["key"] == key].iloc[0]
    ci = f"95% CI {r.ci_low:.2f} to {r.ci_high:.2f}" if not np.isnan(r.ci_low) else "interval not available"
    if r.evidence.startswith("Too few"):
        return f"Only {int(r.cases)} packages are in this group, too few to judge."
    if r.evidence == "More non-renewal":
        return f"{what_more} ({r.summary}; {ci}; Holm-adjusted p = {r.p_holm:.3g})."
    if r.evidence == "Less non-renewal":
        return f"{what_less} ({r.summary}; {ci}; Holm-adjusted p = {r.p_holm:.3g})."
    return f"No clear association ({r.summary}; {ci}; Holm-adjusted p = {r.p_holm:.2g}; {int(r.cases)} packages in the group)."


def _cap(t):
    return t[:1].upper() + t[1:]


def _p(v):
    return "" if pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}")


def _extra_sentence(extras, key):
    r = extras[extras["key"] == key]
    if r.empty:
        return "Not enough data to test."
    r = r.iloc[0]
    verdict = ("it still adds information beyond skipped sessions" if r.adds_beyond_skip_rate
               else "it adds nothing clear beyond skipped sessions")
    return f"after accounting for skipped sessions, {verdict} (odds ratio {r.or_per_sd:.2f} per 1 SD, 95% CI {r.ci_low:.2f} to {r.ci_high:.2f})."


def write_report(res_c, res_p, out_dir, validation=None, data_note="SYNTHETIC", model_compare=None):
    ac, ap, ex = res_c["association"], res_p["association"], res_p["extras"]
    ap_show, ac_show = ap.round(3).assign(p_value=ap["p_value"].map(_p), p_holm=ap["p_holm"].map(_p)), \
        ac.round(3).assign(p_value=ac["p_value"].map(_p), p_holm=ac["p_holm"].map(_p))
    cols = ["factor", "cases", "summary", "ci_low", "ci_high", "p_value", "p_holm", "evidence"]
    names = ["Factor", "Cases", "Effect", "CI low", "CI high", "p", "p (Holm)", "Verdict"]
    act_more = "Clients who were more active in the last 30 days were less likely to leave"
    act_less = "Clients who were MORE active in the last 30 days were more likely to leave"
    txt = [f"# What is associated with not renewing? ({data_note} data)", "",
           "**Read this first.** The data is synthetic. In the generator, renewal odds depend on how much of a package the client attended "
           "and on a hidden engagement level. Wearable activity and body measurements also follow that hidden engagement, but **nothing "
           "(pause length, activity or body measures) has a direct effect on renewal**. So these results show that the method behaves "
           "correctly (it finds what was built in, and does not invent what was not). They do **not** tell you what drives renewal at a "
           "real gym. The same code can be rerun on real, consented data.", "",
           "## Your questions", "",
           "**1. Are frequently skipped sessions associated with not renewing?**", "",
           f"- 35-client cohort ({res_c['n']} finished packages): " + _sentence(ac, "skip_rate", "Yes: more skipped sessions went with more non-renewal", "More skipped sessions went with less non-renewal"),
           f"- Pooled cohorts ({res_p['n']} finished packages): " + _sentence(ap, "skip_rate", "Yes: more skipped sessions went with more non-renewal", "More skipped sessions went with less non-renewal"), "",
           "**2. Is a longer pause associated with not renewing?**", "",
           "- 35-client cohort: " + _sentence(ac, "pause_days", "Longer pauses went with more non-renewal", "Longer pauses went with less non-renewal"),
           "- Pooled cohorts: " + _sentence(ap, "pause_days", "Longer pauses went with more non-renewal", "Longer pauses went with less non-renewal"), ""]
    if validation is not None and len(validation):
        txt += ["**How much pause data is needed?** The method was tested by planting a known pause effect in synthetic data (table below). "
                "Weak effects need hundreds of clients to detect, so a null result on a small real sample means 'cannot tell', not 'no effect'.", "",
                _md(validation.assign(flagged=validation["flagged"].astype(str) + "/" + validation["trials"].astype(str)),
                    ["planted_pause_effect", "cohorts_per_trial", "flagged", "avg_packages_with_a_pause"],
                    ["Planted effect (renewal odds change per 30 days paused)", "Cohorts pooled per trial (35 clients each)",
                     "Trials where a pause effect was flagged", "Avg packages with a pause"]), ""]
    txt += ["**3. Does activity outside the gym (the most recent 30 days) go with renewal?**", "",
            "Activity is measured over the 30 days before the package finished, matching the 30 days a Fitbit file holds.", "",
            f"- Daily steps, 35-client cohort: " + _sentence(ac, "steps_30", act_less, act_more),
            f"- Daily steps, pooled: " + _sentence(ap, "steps_30", act_less, act_more) + " " + _cap(_extra_sentence(ex, "steps_30")),
            f"- Checking in (share of days with Fitbit data), pooled: " + _sentence(ap, "wear_30", "Clients who checked in more were more likely to leave", "Clients who checked in more often were less likely to leave") + " " + _cap(_extra_sentence(ex, "wear_30")),
            f"- Step trend inside the 30 days, pooled: " + _sentence(ap, "steps_trend_30", "Rising steps went with more non-renewal", "Rising steps went with less non-renewal"), "",
            "**4. Do body measurements go with renewal?** (body weight, body fat %, skeletal muscle mass)", "",
            "- Starting values (pooled): " + "; ".join(f"{lab}: {ap[ap['key'] == k].iloc[0].evidence.lower()}" for k, lab in
                                                  (("baseline_weight", "body weight"), ("baseline_bodyfat", "body fat %"), ("baseline_smm", "muscle mass"))) + ".",
            "- Change over the package (pooled): " + "; ".join(f"{lab}: {ap[ap['key'] == k].iloc[0].evidence.lower()}" for k, lab in
                                                            (("delta_weight", "body weight"), ("delta_bodyfat", "body fat %"), ("delta_smm", "muscle mass"))) + ".",
            "- Moved in the goal direction: " + _sentence(ap, "goal_progress", "Clients who progressed were more likely to leave", "Clients who progressed were less likely to leave"), "",
            "The 'beyond skipped sessions' checks are exploratory: 11 factors are checked and the intervals are not corrected for that, so an "
            "occasional false alarm is expected (about 1 in 20).", ""]
    if model_compare is not None and len(model_compare):
        txt += ["## Does activity improve the prediction model?", "",
                "Cross-validated AUC (grouped by client, repeated over different splits; 0.5 is a coin flip). A difference smaller than the spread is not a real improvement.", "",
                _md(model_compare, ["model", "features", "cv_auc", "spread", "packages"],
                    ["Model", "Inputs", "AUC", "Spread (+/-)", "Packages"]), ""]
    txt += ["## Full results: pooled cohorts", "", _md(ap_show, ["family"] + cols, ["Family"] + names), "",
            "Effect is AUC (0.5 = no association; above 0.5 = higher values go with not renewing) for continuous factors, and the difference in "
            "non-renewal rate for yes/no factors. CIs resample whole clients. Holm correction is applied within each family of tests.", "",
            "![skip rate](skip_rate_pooled.png)", "", "![pause length](pause_length_pooled.png)", "",
            "![steps](steps_30_pooled.png)", "", "![check-ins](checkins_30_pooled.png)", "",
            "## Do activity and measurement factors add anything beyond skipped sessions? (pooled, exploratory)", "",
            _md(ex.round(2), ["factor", "n", "or_per_sd", "ci_low", "ci_high", "adds_beyond_skip_rate"],
                ["Factor", "n", "Odds ratio per 1 SD", "CI low", "CI high", "Adds beyond skip rate"]), "",
            "## Full results: the 35-client cohort", "", _md(ac_show, ["family"] + cols, ["Family"] + names), "",
            "## Adjusted model (pooled)", "",
            f"Odds ratios of not renewing, each adjusted for the others. n = {res_p['info'].get('n')}, events = {res_p['info'].get('events')}, "
            f"grouped cross-validation AUC = {res_p['info'].get('grouped_cv_auc', float('nan')):.2f}.", "",
            _md(res_p["adjusted"].round(2), ["term", "odds_ratio", "ci_low", "ci_high", "ci_excludes_1"],
                ["Factor", "Odds ratio", "CI low", "CI high", "CI excludes 1"]), "",
            "![adjusted](adjusted_pooled.png)", "",
            "## Limits", "",
            "- Associations, not causes. A client who skips sessions or has fewer Fitbit days may be losing interest for reasons the data does not show.",
            "- Only finished packages with a renewal decision are included. There is no lapse rule, so clients who stop attending have no outcome and are missing from this analysis. Record a closure status in real data to include them.",
            "- When real Fitbit series are attached to made-up clients (`--fitbit`), that activity has no real link to those clients' bookings, so it only demonstrates the plumbing.",
            "- Body measures need consent and are not appropriate to track for every client (for example where there is a history of disordered eating).",
            "- A maternity pause is a life stage, not disengagement; it is included so its association can be examined, but very few cases exist.",
            "- Synthetic data: results show the method works, not how real clients behave."]
    (out_dir / "report.md").write_text("\n".join(txt), encoding="utf-8")


def main():
    import argparse
    from dataclasses import replace

    from . import config
    from .pt_config import PTConfig
    from .pt_risk import build_training_data, compare_activity_value, load_tables
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--validate", action="store_true", help="also run the planted-effect validation (about 3 minutes)")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    cfg = PTConfig()
    folder = config.ROOT / "data" / "synthetic"
    tables = load_tables(folder) if (folder / "packages.csv").exists() else build_tables(cfg)
    out = config.ROOT / "docs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    cohort = analysis_frame(tables, "cohort")
    pooled = pd.concat([cohort] + [analysis_frame(build_tables(replace(cfg, seed=s)), str(s)) for s in range(1, 11)],
                       ignore_index=True)
    res_c, res_p = analyze(cohort, B=a.boot), analyze(pooled, B=a.boot)
    for name, res in (("cohort", res_c), ("pooled", res_p)):
        res["association"].to_csv(out / f"association_{name}.csv", index=False)
        res["dose"].to_csv(out / f"dose_response_{name}.csv", index=False)
        res["adjusted"].to_csv(out / f"adjusted_model_{name}.csv", index=False)
        res["extras"].to_csv(out / f"beyond_skip_rate_{name}.csv", index=False)
        charts(res, out, name)
    cohort.to_csv(out / "analysis_frame_cohort.csv", index=False)
    compare = compare_activity_value(cfg, data=build_training_data(cfg))
    compare.to_csv(out / "model_comparison_activity.csv", index=False)
    vpath = out / "validation.csv"
    if a.validate:
        validate().to_csv(vpath, index=False)
    validation = pd.read_csv(vpath) if vpath.exists() else None
    write_report(res_c, res_p, out, validation, model_compare=compare)
    print("Wrote", out)
    print(res_c["association"][["family", "factor", "cases", "summary", "evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
