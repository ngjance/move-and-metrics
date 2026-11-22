"""Central settings. Edit here rather than hunting through the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

# Kaggle "FitBit Fitness Tracker Data" (Mobius): put dailyActivity_merged.csv here
FITBIT_DAILY_FILE = RAW_DIR / "dailyActivity_merged.csv"

# Kaggle "Gym Customers Features and Churn": save the CSV here (or pass --csv)
GYM_CHURN_FILE = RAW_DIR / "gym_churn.csv"

# --- Coach-defined rules (YOUR domain judgement; change and justify these) ---
ACTIVE_STEP_THRESHOLD = 2000   # a day with fewer steps counts as "inactive / not worn"
STREAK_DAYS_HIGH = 3           # this many inactive days in a row at the end => High
DROP_PCT_HIGH = 0.30           # 14-day average steps down this much vs prior 14 days => High
LOW_ACTIVE_DAYS_WATCH = 3      # 3 or fewer active days in the last 7 => Watch
COMPARE_WINDOW_DAYS = 14
