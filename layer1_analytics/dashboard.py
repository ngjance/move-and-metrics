"""Piece 1: coach dashboard (Streamlit): renewal risk + what predicts non-renewal.

Run from the repo root:   streamlit run layer1_analytics/dashboard.py
Reads the synthetic personal-training tables in data/synthetic/ (builds them in memory if absent).
All data is SYNTHETIC. The first-version Fitbit-only dashboard is in dashboard_fitbit.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from layer1_analytics import config
from layer1_analytics.pt_analysis import analysis_frame, analyze
from layer1_analytics.pt_config import PTConfig
from layer1_analytics.pt_risk import load_tables, normalize, risk_table, train_model
from layer1_analytics.pt_synthetic import generate

st.set_page_config(page_title="Move & Metrics: Renewal", layout="wide")
CFG = PTConfig()
ANALYSIS_DIR = config.ROOT / "docs" / "analysis"


@st.cache_resource(show_spinner="Loading data and the renewal model (first run can take a few seconds)...")
def load_everything():
    folder = config.ROOT / "data" / "synthetic"
    tables = load_tables(folder) if (folder / "packages.csv").exists() else normalize(generate(CFG))
    mdir = config.OUTPUT_DIR / "pt_model"
    if (mdir / "renewal_model.joblib").exists() and (mdir / "metrics.json").exists():
        model, metrics = joblib.load(mdir / "renewal_model.joblib"), json.loads((mdir / "metrics.json").read_text())
    else:
        pipe, cols, bounds, metrics, _ = train_model(CFG)
        model = (pipe, cols, bounds)
    return tables, risk_table(tables, CFG, model), metrics


@st.cache_resource(show_spinner="Running the analysis...")
def cohort_analysis(_tables):
    return analyze(analysis_frame(_tables, "cohort"), B=200)


tables, risk, metrics = load_everything()
st.title("Renewal: who needs a conversation, and what predicts non-renewal?")
st.warning("All data is SYNTHETIC (made up for a portfolio project). Informational coaching aid, not medical advice.")
tab_risk, tab_why = st.tabs(["Renewal risk", "What predicts non-renewal?"])

# ----------------------------------------------------------------------------- tab 1
with tab_risk:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High risk", int((risk["priority"] == "High").sum()))
    c2.metric("Watch", int((risk["priority"] == "Watch").sum()))
    c3.metric("Renewal pending", int((risk["priority"] == "Renewal pending").sum()))
    c4.metric("Paused", int((risk["priority"] == "Paused").sum()))

    f1, f2 = st.columns(2)
    levels = f1.multiselect("Priority", list(risk["priority"].unique()), default=list(risk["priority"].unique()))
    goals = f2.multiselect("Goal", sorted(risk["goal"].unique()), default=sorted(risk["goal"].unique()))
    view = risk[risk["priority"].isin(levels) & risk["goal"].isin(goals)]

    st.subheader("Packages with sessions left, highest risk first")
    st.caption("Model risk is the model's estimated chance the client does not renew, learned from past packages. "
               "A maternity pause is life stage, not churn.")
    st.dataframe(
        view[["client_id", "goal", "age", "package_size", "sessions_left", "used_pct", "days_since_last_session",
              "pace_vs_plan_4wk", "avg_steps_30", "fitbit_days_30", "priority", "model_risk", "model_drivers", "note"]],
        use_container_width=True, hide_index=True,
        column_config={
            "used_pct": st.column_config.ProgressColumn("Package used", min_value=0.0, max_value=1.0, format="percent"),
            "model_risk": st.column_config.ProgressColumn("Model risk", min_value=0.0, max_value=1.0, format="percent"),
            "pace_vs_plan_4wk": st.column_config.NumberColumn("Pace vs plan (4 wks)", format="percent"),
            "avg_steps_30": st.column_config.NumberColumn("Steps/day (30d)", format="%d"),
            "fitbit_days_30": st.column_config.NumberColumn("Fitbit days (30d)", format="percent"),
            "days_since_last_session": st.column_config.NumberColumn("Days since last session"),
            "model_drivers": st.column_config.TextColumn("Why (model)", width="large"),
            "note": st.column_config.TextColumn("Note", width="medium"),
        })
    st.caption("Clients who stopped attending long ago have no outcome (there is no lapse rule), so the model has never seen "
               "one labelled. Their high scores come from the model reaching the edge of what it has seen. Treat them as "
               "'check in' flags, not predictions.")

    st.subheader("Client detail")
    cid = st.selectbox("Client", view["client_id"].tolist() or risk["client_id"].tolist())
    row = risk[risk["client_id"] == cid].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Sessions left", int(row["sessions_left"]), help=f"Package of {int(row['package_size'])} sessions")
    b.metric("Priority", row["priority"])
    c.metric("Model risk", "n/a" if pd.isna(row["model_risk"]) else f"{row['model_risk']:.0%}")
    st.progress(float(min(max(row["used_pct"], 0), 1)), text=f"Package used: {row['used_pct']:.0%}")
    if row["model_drivers"]:
        st.write(f"**Why (model):** {row['model_drivers']}")
    if row["note"]:
        st.write(f"**Note:** {row['note']}")

    ses = tables["sessions"][tables["sessions"]["client_id"] == cid].copy()
    ses["week"] = ses["date"].dt.to_period("W-SUN").dt.start_time
    weekly = ses.pivot_table(index="week", columns="outcome", values="session_id", aggfunc="count", fill_value=0)
    left, right = st.columns(2)
    with left:
        st.caption("Booking outcomes per week")
        st.bar_chart(weekly)
    with right:
        if "daily_activity" in tables:
            act = tables["daily_activity"]
            act = act[act["Id"] == cid].set_index("ActivityDate")
            if len(act):
                st.caption(f"Daily steps (source: {act['source'].iloc[0]})")
                st.line_chart(act["TotalSteps"])
    st.caption("Packages, pauses and outcomes for this client")
    st.dataframe(tables["packages"][tables["packages"]["client_id"] == cid], use_container_width=True, hide_index=True)
    pz = tables["pauses"][tables["pauses"]["client_id"] == cid]
    if len(pz):
        st.dataframe(pz, use_container_width=True, hide_index=True)
    oc = tables["outcomes"][tables["outcomes"]["client_id"] == cid]
    if len(oc):
        st.dataframe(oc, use_container_width=True, hide_index=True)

    st.subheader("Coach note (not saved)")
    st.text_area("What will you say to this client this week?", height=100,
                 placeholder="The data suggests who to talk to; your coaching judgement decides what to say.")

    with st.expander("How is the risk score made, and how far can I trust it?"):
        st.markdown(f"""
**Model.** Logistic regression trained on past packages from 10 other synthetic cohorts, using only information available when
half the package was used: gym attendance (cancels, no-shows, pace, gaps) and, outside the gym, the 30 days of Fitbit-style activity before
that date (steps, active minutes, days with data, step trend). Maternity pauses and pending renewals are not counted as churn. Sample: {metrics['n_train']} packages.
Adding the activity inputs improved cross-validated accuracy only slightly, within the noise (see the analysis tab). If a Fitbit file is attached to made-up
clients, their activity has no real link to those clients' bookings, so in that mode the activity inputs only demonstrate the plumbing.
No hand-set rules: the weights are learned. Priority cut-offs ({CFG.model_high:.0%} High, {CFG.model_watch:.0%} Watch) are a choice
based on how many past packages each would have flagged (see `pt_config.py`).

**Accuracy (synthetic data).** Grouped cross-validation AUC {metrics['grouped_cv_auc_mean']} (+/- {metrics['grouped_cv_auc_sd']});
held-out 35-client cohort AUC {metrics['heldout_cohort_auc']} on {metrics['heldout_cohort_n']} packages (noisy). 0.5 is a coin flip, 1.0 perfect.

**Limits.** Synthetic data; the attendance-to-renewal link is an assumption in the generator. Real use needs real, consented data and a fresh evaluation.
""")

# ----------------------------------------------------------------------------- tab 2
with tab_why:
    st.subheader("What is associated with not renewing a package?")
    st.info("Associations on SYNTHETIC data. The generator makes renewal depend on attendance, not on pause length, so this tab "
            "shows that the method finds what exists and does not invent what does not. It says nothing about real clients.")
    pooled_csv = ANALYSIS_DIR / "association_pooled.csv"
    cols = ["family", "factor", "cases", "summary", "ci_low", "ci_high", "p_holm", "evidence"]
    names = {"family": "Family", "factor": "Factor", "cases": "Cases", "summary": "Effect", "ci_low": "CI low", "ci_high": "CI high",
             "p_holm": "p (Holm)", "evidence": "Verdict"}
    res_c = cohort_analysis(tables)
    st.markdown(f"**This 35-client cohort:** {res_c['n']} finished packages, {res_c['events']} did not renew.")
    st.dataframe(res_c["association"][cols].round(3).rename(columns=names), use_container_width=True, hide_index=True)
    if pooled_csv.exists():
        st.markdown("**Pooled across 11 synthetic cohorts** (enough packages to see smaller effects; run `python -m layer1_analytics.pt_analysis`).")
        pooled = pd.read_csv(pooled_csv)
        st.dataframe(pooled[cols].round(3).rename(columns=names), use_container_width=True, hide_index=True)
        g1, g2 = st.columns(2)
        for col, name in ((g1, "skip_rate_pooled.png"), (g2, "pause_length_pooled.png"),
                          (g1, "steps_30_pooled.png"), (g2, "checkins_30_pooled.png")):
            if (ANALYSIS_DIR / name).exists():
                col.image(str(ANALYSIS_DIR / name))
        if (ANALYSIS_DIR / "beyond_skip_rate_pooled.csv").exists():
            st.markdown("**Does it add anything beyond skipped sessions?** (exploratory; intervals are not corrected for 11 comparisons)")
            st.dataframe(pd.read_csv(ANALYSIS_DIR / "beyond_skip_rate_pooled.csv").round(2), use_container_width=True, hide_index=True)
        if (ANALYSIS_DIR / "model_comparison_activity.csv").exists():
            st.markdown("**Does activity improve the prediction model?**")
            st.dataframe(pd.read_csv(ANALYSIS_DIR / "model_comparison_activity.csv"), use_container_width=True, hide_index=True)
        if (ANALYSIS_DIR / "adjusted_pooled.png").exists():
            st.image(str(ANALYSIS_DIR / "adjusted_pooled.png"))
    else:
        st.caption("Run `python -m layer1_analytics.pt_analysis` to add the pooled results and charts.")
    if (ANALYSIS_DIR / "validation.csv").exists():
        st.markdown("**Can the method detect a pause effect at all?** Planting a known effect in synthetic data:")
        st.dataframe(pd.read_csv(ANALYSIS_DIR / "validation.csv"), use_container_width=True, hide_index=True)
    st.caption("Only finished packages with a renewal decision are analysed. Clients who stopped attending have no outcome "
               "(there is no lapse rule) and are not included. Associations are not causes.")
