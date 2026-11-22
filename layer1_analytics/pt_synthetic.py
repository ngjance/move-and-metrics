"""Generator for a linked, SYNTHETIC personal-training database.

Tables: clients, packages, pauses, sessions, outcomes, measurements (+ truth, kept apart).
Every client has a hidden 'engagement' level that influences cancellations, drop-off and
renewal. Renewal also depends on how the client actually behaved in that package (share of
booking attempts attended), plus noise, so a model cannot just read back a single rule. The hidden values are
saved in `truth` only for validation; never feed them to a model.

Planted scenarios (always present):
  C001 prenatal -> exactly 10 sessions left -> maternity pause -> returns postpartum, finishes, does not renew
  C002 prenatal -> maternity pause -> returns, finishes, renews
  C003 prenatal -> maternity pause -> returns, attends a few sessions, then stops (drop-off)
plus 3 more drop-offs, 3 non-maternity pauses and some early renewals, all randomised.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from .pt_config import PTConfig

ONE = timedelta(days=1)


def _age_band(a):
    return "<30" if a < 30 else "30-39" if a < 40 else "40-49" if a < 50 else "50-59" if a < 60 else "60+"


def _weighted(rng, mapping):
    keys = list(mapping)
    p = np.array([mapping[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def build_specs(cfg: PTConfig, rng):
    from datetime import date
    specs = []
    planted = [
        dict(goal="Prenatal", age=32, join=date(2025, 10, 6), spw=2, e=0.70, size=20,
             maternity=dict(left=10, days=200), renew=[False], renew_reason="Life stage change",
             scenario="Planted: prenatal -> maternity pause -> postpartum finish -> no renewal"),
        dict(goal="Prenatal", age=30, join=date(2025, 12, 1), spw=2, e=0.75, size=20,
             maternity=dict(left=10, days=215), renew=[True],
             scenario="Planted: prenatal -> maternity pause -> postpartum finish -> renews"),
        dict(goal="Prenatal", age=35, join=date(2026, 1, 12), spw=1, e=0.25, size=20,
             maternity=dict(left=10, days=160), dropout_at_total=13,
             scenario="Planted: prenatal -> maternity pause -> returns, then drops off"),
    ]
    specs += planted
    remaining = dict(cfg.goal_counts)
    remaining["Prenatal"] -= 3
    for goal, n in remaining.items():
        for _ in range(n):
            specs.append(dict(goal=goal, scenario=""))
    for i, s in enumerate(specs):
        g = s["goal"]
        lo, hi = cfg.age_ranges[g]
        s.setdefault("age", int(rng.integers(lo, hi + 1)))
        if "join" not in s:
            span = (cfg.join_window[1] - cfg.join_window[0]).days
            s["join"] = cfg.join_window[0] + timedelta(days=int(rng.integers(0, span + 1)))
        s.setdefault("spw", int(rng.choice(cfg.planned_per_week_choices[g])))
        s.setdefault("e", float(np.clip(rng.beta(4, 3), 0.08, 0.97)))
        s["weekdays"] = sorted(rng.choice(6, size=s["spw"], replace=False).tolist())  # Mon-Sat
        s["referral"] = str(rng.choice(cfg.referral_sources))
        s["weeks_pregnant_at_join"] = int(rng.integers(10, 26)) if g == "Prenatal" else None
        s["weeks_postpartum_at_join"] = int(rng.integers(6, 40)) if g == "Postpartum" else None

    # random drop-offs (one is already planted) and non-maternity pausers
    pool = [i for i, s in enumerate(specs) if s["goal"] != "Prenatal"]
    n_drop = max(1, round(cfg.share_dropout * cfg.n_clients)) - 1
    n_pause = round(cfg.share_random_pause * cfg.n_clients)
    chosen = rng.choice(pool, size=n_drop + n_pause, replace=False).tolist()
    for i in chosen[:n_drop]:
        s = specs[i]
        s["e"] = float(rng.uniform(0.08, 0.30))
        s["dropout_frac"] = float(rng.uniform(0.25, 0.70))
        s["join"] = min(s["join"], date(2026, 7, 1))
        s["scenario"] = "Random: drop-off mid-package"
    for i in chosen[n_drop:]:
        s = specs[i]
        s["pause"] = dict(frac=float(rng.uniform(0.25, 0.75)), days=int(rng.integers(14, 61)),
                          reason=str(rng.choice(cfg.pause_reasons)))
        s["join"] = min(s["join"], date(2026, 6, 1))
        s["scenario"] = "Random: non-maternity pause"

    mean_e = float(np.mean([s["e"] for s in specs]))
    for s in specs:
        k = 1 - 1.2 * (s["e"] - mean_e)
        s["p_early"] = float(np.clip(cfg.p_early_cancel * k, 0.03, 0.70))
        s["p_late"] = float(np.clip(cfg.p_late_cancel * k, 0.01, 0.40))
        s["p_noshow"] = float(np.clip(cfg.p_no_show * k, 0.0, 0.10))
        s["p_renew"] = float(np.clip(cfg.renewal_rate + 0.45 * (s["e"] - mean_e), 0.05, 0.95))
    rng.shuffle(specs[3:])  # keep planted clients first; shuffle the rest below
    rest = specs[3:]
    rng.shuffle(rest)
    specs = specs[:3] + rest
    for i, s in enumerate(specs):
        s["client_id"] = f"C{i + 1:03d}"
    return specs


def _pick_size(cfg, rng, spec):
    sizes, w = list(cfg.package_sizes), list(cfg.package_size_weights)
    if spec["spw"] < 2:
        w = [wi if sz != 100 else 0 for sz, wi in zip(sizes, w)]
    w = np.array(w, float)
    return int(sizes[int(rng.choice(len(sizes), p=w / w.sum()))])


def simulate_client(spec, cfg, rng, counters):
    sessions, packages, pauses, outcomes = [], [], [], []
    start = purchase = spec["join"]
    goal = spec["goal"]
    total_attended = 0
    dropout_active, fade_left = False, 0
    post_maternity = maternity_done = random_pause_done = False
    dropout_at = spec.get("dropout_at_total")
    pkg_no = 0

    def phase():
        if goal == "Prenatal":
            return "Postpartum" if post_maternity else "Prenatal"
        return "Postpartum" if goal == "Postpartum" else "Standard"

    while start <= cfg.as_of:
        pkg_no += 1
        counters["pkg"] += 1
        pid = f"P{counters['pkg']:03d}"
        size = spec["size"] if (pkg_no == 1 and "size" in spec) else _pick_size(cfg, rng, spec)
        if pkg_no > 1 and "size" in spec:
            size = spec["size"]
        if pkg_no == 1 and "dropout_frac" in spec and dropout_at is None:
            dropout_at = max(3, int(spec["dropout_frac"] * size))
        consumed = attended = 0
        pending, d, paused_until = {}, start, None
        near_end, last_consumed, last_att = None, None, None
        pause_ongoing = False

        def add(date_, outcome, notice=None, retry=0):
            sessions.append(dict(package_id=pid, client_id=spec["client_id"], date=date_,
                                 phase=phase(), outcome=outcome, cancel_notice_hours=notice,
                                 reschedule_attempt=retry))

        def start_pause(day, days, reason, maternity):
            nonlocal paused_until, pause_ongoing, post_maternity
            ps, pe = day + ONE, day + ONE + timedelta(days=days)
            counters["pause"] += 1
            ongoing = pe > cfg.as_of
            pauses.append(dict(pause_id=f"Z{counters['pause']:03d}", package_id=pid,
                               client_id=spec["client_id"], pause_start=ps,
                               pause_end=None if ongoing else pe, reason_category=reason))
            paused_until, pause_ongoing = pe, ongoing
            pending.clear()
            if maternity:
                post_maternity = True

        while d <= cfg.as_of and consumed < size:
            if paused_until is not None:
                if d < paused_until:
                    d += ONE
                    continue
                paused_until = None
            attempts = pending.pop(d, [])
            if d.weekday() in spec["weekdays"]:
                attempts.append(0)
            for retry in attempts:
                if consumed >= size:
                    break
                if dropout_active:
                    if fade_left > 0:
                        add(d, "Early cancel", int(rng.integers(24, 73)), retry)
                        fade_left -= 1
                    continue
                u = rng.random()
                if u < spec["p_noshow"]:
                    add(d, "No-show", None, retry)
                    if cfg.no_show_forfeits:
                        consumed += 1
                        last_consumed = d
                elif u < spec["p_noshow"] + spec["p_late"]:
                    add(d, "Late cancel", int(rng.integers(1, 24)), retry)
                    if cfg.late_cancel_forfeits:
                        consumed += 1
                        last_consumed = d
                elif u < spec["p_noshow"] + spec["p_late"] + spec["p_early"]:
                    add(d, "Early cancel", int(rng.integers(24, 73)), retry)
                    if retry < cfg.max_reschedules:
                        rd = d + timedelta(days=int(rng.integers(2, 5)))
                        while rd in pending or rd.weekday() in spec["weekdays"] or rd.weekday() == 6:
                            rd += ONE
                        if rd <= cfg.as_of:
                            pending.setdefault(rd, []).append(retry + 1)
                else:
                    add(d, "Attended", None, retry)
                    attended += 1
                    last_att = d
                    total_attended += 1
                    consumed += 1
                    last_consumed = d
                    if near_end is None and consumed >= size - 3:
                        near_end = d
                    rp = spec.get("pause")
                    if rp and not random_pause_done and attended >= int(rp["frac"] * size):
                        random_pause_done = True
                        start_pause(d, rp["days"], rp["reason"], False)
                        break
                    if dropout_at is not None and total_attended >= dropout_at:
                        dropout_active, fade_left = True, int(rng.integers(1, 4))
                        dropout_at = None
                mat = spec.get("maternity")
                if mat and not maternity_done and size - consumed <= mat["left"]:
                    maternity_done = True   # pause starts when exactly `left` sessions remain
                    start_pause(d, mat["days"], "Maternity", True)
                    break
            d += ONE

        completed = consumed >= size
        if completed:
            status = "Completed"
        elif pause_ongoing:
            status = "Paused"
        else:
            status = "Active"          # no expiry dates and no lapse rule: a client who stops attending stays Active
        packages.append(dict(
            package_id=pid, client_id=spec["client_id"], package_no=pkg_no, sessions_signed=size,
            price_per_session=cfg.price_per_session[size], purchase_date=purchase, start_date=start,
            status=status,
            planned_per_week=spec["spw"]))

        if status == "Completed":
            end = last_consumed
            plan = spec.get("renew")
            pending_decision = (cfg.as_of - end).days < cfg.pending_window_days
            if pending_decision:
                renewed, reason = "Pending", None
            else:
                mine = [x for x in sessions if x["package_id"] == pid]
                share = sum(x["outcome"] == "Attended" for x in mine) / len(mine) if mine else None
                typical = cfg.renewal_typical_attendance_share
                pause_days = sum((z["pause_end"] - z["pause_start"]).days for z in pauses
                                 if z["package_id"] == pid and z["pause_end"] is not None)
                p_ren = spec["p_renew"] if share is None else float(
                    np.clip(spec["p_renew"] + cfg.renewal_behaviour_weight * (share - typical)
                            - cfg.renewal_pause_penalty_per_30d * pause_days / 30, 0.05, 0.95))
                decision = plan[pkg_no - 1] if plan and pkg_no <= len(plan) else bool(rng.random() < p_ren)
                renewed = "Yes" if decision else "No"
                reason = None if decision else spec.get("renew_reason") or _weighted(rng, cfg.non_renewal_reasons)
            counters["out"] += 1
            outcomes.append(dict(outcome_id=f"O{counters['out']:03d}", package_id=pid,
                                 client_id=spec["client_id"], end_date=end, outcome="Completed",
                                 renewed=renewed, reason_category=reason,
                                 renewed_package_id=f"P{counters['pkg'] + 1:03d}" if renewed == "Yes" else None))
            if renewed != "Yes":
                break
            early = near_end is not None and near_end < end and rng.random() < cfg.share_early_renewal
            start = min(end + timedelta(days=1 + int(rng.integers(0, 15))), cfg.as_of)
            purchase = near_end if early else start
        else:
            break
    return sessions, packages, pauses, outcomes


def _measurements(specs, sessions, cfg, rng):
    """Goal-specific measure plus body composition (body weight, body fat %, skeletal muscle mass) at each check-in.
    Changes follow attended sessions, scaled by the client's hidden engagement, plus noise."""
    rows, n = [], 0
    att = sessions[sessions["outcome"] == "Attended"]
    cfgm = {"Fat loss": ("Waist circumference", "cm", (78, 100), -0.06),
            "Strength": ("Deadlift 5RM", "kg", (40, 90), 0.5),
            "Prenatal": ("Energy level (1-10)", "score", (5, 7), 0.01),
            "Postpartum": ("Plank hold", "sec", (20, 50), 0.6),
            "Longevity": ("30-second sit-to-stand", "reps", (8, 14), 0.06)}
    # change per attended session: (weight kg, body fat points, skeletal muscle kg)
    drift = {"Fat loss": (-0.040, -0.050, 0.005), "Strength": (0.020, -0.015, 0.020),
             "Longevity": (-0.005, -0.010, 0.008), "Postpartum": (-0.030, -0.035, 0.006), "Prenatal": (0.0, 0.0, 0.0)}
    for s in specs:
        a = att[att["client_id"] == s["client_id"]].sort_values("date")
        if a.empty:
            continue
        g = s["goal"]
        name, unit, (lo, hi), slope = cfgm[g]
        base = rng.uniform(lo, hi)
        w0 = float(rng.normal(64 if g in ("Prenatal", "Postpartum") else 72, 9))
        bf0 = float(np.clip(rng.normal({"Strength": 24, "Longevity": 32}.get(g, 31), 5), 14, 46))
        smm0 = 0.5 * w0 * (1 - bf0 / 100)
        post = sessions[(sessions["client_id"] == s["client_id"]) & (sessions["phase"] == "Postpartum")]
        post_start = post["date"].min() if len(post) else None
        first, last = a["date"].iloc[0], a["date"].iloc[-1]
        day = first
        while day <= last:
            done = int((a["date"] <= day).sum())
            f = 0.5 + s["e"]
            dw, dbf, dsm = (x * done * f for x in drift[g])
            if g == "Prenatal":                                  # weight rises in pregnancy, eases postpartum
                weeks = (day - first).days / 7
                if post_start is None or day < post_start:
                    dw, dbf = 0.25 * weeks, 0.15 * weeks
                else:
                    pw = (post_start - first).days / 7
                    dw, dbf = 0.25 * pw - 0.15 * (day - post_start).days / 7, 0.15 * pw - 0.10 * (day - post_start).days / 7
            for label, u, val in (("Body weight", "kg", w0 + dw + rng.normal(0, 0.6)),
                                  ("Body fat percentage", "%", bf0 + dbf + rng.normal(0, 0.8)),
                                  ("Skeletal muscle mass", "kg", smm0 + dsm + rng.normal(0, 0.3))):
                n += 1
                rows.append(dict(measurement_id=f"M{n:04d}", client_id=s["client_id"], date=day,
                                 measure_name=label, value=round(float(val), 1), unit=u))
            n += 1
            val = base + slope * done * f + rng.normal(0, abs(slope) * 3 + 0.2)
            rows.append(dict(measurement_id=f"M{n:04d}", client_id=s["client_id"], date=day,
                             measure_name=name, value=round(float(val), 1), unit=unit))
            day += timedelta(days=56)
    return pd.DataFrame(rows)


def generate(cfg: PTConfig = None):
    cfg = cfg or PTConfig()
    rng = np.random.default_rng(cfg.seed)
    specs = build_specs(cfg, rng)
    counters = {"pkg": 0, "pause": 0, "out": 0}
    S, P, Z, O = [], [], [], []
    for s in specs:
        a, b, c, d = simulate_client(s, cfg, rng, counters)
        S += a; P += b; Z += c; O += d
    sessions = pd.DataFrame(S).sort_values(["date", "package_id"]).reset_index(drop=True)
    sessions.insert(0, "session_id", [f"S{i + 1:05d}" for i in range(len(sessions))])
    clients = pd.DataFrame([dict(
        client_id=s["client_id"], age=s["age"], goal=s["goal"], join_date=s["join"],
        referral_source=s["referral"], planned_per_week=s["spw"],
        weeks_pregnant_at_join=s["weeks_pregnant_at_join"],
        weeks_postpartum_at_join=s["weeks_postpartum_at_join"]) for s in specs])
    clients["age_band"] = clients["age"].map(_age_band)
    truth = pd.DataFrame([dict(client_id=s["client_id"], hidden_engagement=round(s["e"], 3),
                               p_early=round(s["p_early"], 3), p_renew=round(s["p_renew"], 3),
                               scenario=s["scenario"]) for s in specs])
    meas = _measurements(specs, sessions, cfg, rng)
    return dict(clients=clients, packages=pd.DataFrame(P), pauses=pd.DataFrame(Z),
                sessions=sessions, outcomes=pd.DataFrame(O), measurements=meas, truth=truth)
