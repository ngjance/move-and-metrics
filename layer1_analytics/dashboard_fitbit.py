"""Piece 1 (first version): Fitbit-only activity dashboard (Streamlit). Kept for reference.

Run from the repo root:  streamlit run layer1_analytics/dashboard_fitbit.py
Shows real Fitbit data if data/raw/dailyActivity_merged.csv exists, else synthetic clients.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from layer1_analytics.data_loading import load_activity
from layer1_analytics.features import client_status, prepare, weekly_summary

st.set_page_config(page_title="Move & Metrics: Coach Dashboard", layout="wide")


@st.cache_data
def get_data():
    return load_activity()


df, source = get_data()
st.title("Coach dashboard")
if source == "synthetic":
    st.warning("Showing SYNTHETIC clients. Add dailyActivity_merged.csv to data/raw/ to use the Fitbit dataset.")
else:
    st.info("Showing the public Kaggle FitBit Fitness Tracker dataset (2016, 30 users). Not client data.")

status = client_status(df)
counts = status["status"].value_counts()
c1, c2, c3 = st.columns(3)
c1.metric("High attention", int(counts.get("High", 0)))
c2.metric("Watch", int(counts.get("Watch", 0)))
c3.metric("On track", int(counts.get("OK", 0)))

st.subheader("Who needs a check-in first")
st.caption("Rule-based baseline using thresholds set in layer1_analytics/config.py. "
           "Not the machine-learning model.")
show = status.copy()
show["change_vs_prior"] = show["change_vs_prior"].map(lambda x: "" if pd.isna(x) else f"{x:+.0%}")
st.dataframe(show, use_container_width=True, hide_index=True)

st.subheader("Client detail")
client = st.selectbox("Client Id", status["Id"].tolist())
d = prepare(df)
d = d[d["Id"] == client].set_index("ActivityDate")
left, right = st.columns(2)
with left:
    st.caption("Daily steps with 7-day average")
    chart = pd.DataFrame({"steps": d["TotalSteps"],
                          "7-day avg": d["TotalSteps"].rolling(7, min_periods=1).mean()})
    st.line_chart(chart)
with right:
    st.caption("Weekly active minutes (very + fairly active)")
    w = weekly_summary(df)
    w = w[w["Id"] == client].set_index("Week")
    st.bar_chart(w["active_minutes"])

st.subheader("Coach note (not saved)")
st.text_area("What would you say to this client this week?", height=100,
             placeholder="The data suggests who to talk to; your coaching judgement decides what to say.")
st.caption("Informational coaching aid, not medical advice.")
