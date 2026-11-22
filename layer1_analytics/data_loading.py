"""Load real Fitbit data if present, otherwise fall back to synthetic clients."""
import pandas as pd

from . import config
from .synthetic import generate_clients

NEEDED = ["Id", "ActivityDate", "TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes",
          "LightlyActiveMinutes", "SedentaryMinutes", "Calories"]


def load_fitbit_daily(path=None) -> pd.DataFrame:
    path = path or config.FITBIT_DAILY_FILE
    df = pd.read_csv(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df[NEEDED].copy()
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], format="mixed")
    return df


def load_activity(prefer_real: bool = True):
    """Return (dataframe, source_label). source_label is 'real' or 'synthetic'."""
    if prefer_real and config.FITBIT_DAILY_FILE.exists():
        return load_fitbit_daily(), "real"
    df, _ = generate_clients()
    return df, "synthetic"
