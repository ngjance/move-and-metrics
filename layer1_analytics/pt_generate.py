"""Generate the synthetic personal-training database.

    python -m layer1_analytics.pt_generate                # CSVs + Excel workbook
    python -m layer1_analytics.pt_generate --fitbit data/raw/dailyActivity_merged.csv

Output goes to data/synthetic/ (safe to commit: all synthetic). The hidden-variable file is saved
as _truth_do_not_model.csv for validation only; never use it as model input.
"""
import argparse
from pathlib import Path

from . import config
from .pt_config import PTConfig
from .pt_synthetic import generate
from .pt_workbook import build_workbook
from .wearables import build_daily_activity


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=str(config.ROOT / "data" / "synthetic"))
    ap.add_argument("--fitbit", default=None, help="path to dailyActivity_merged.csv to use real Fitbit series")
    a = ap.parse_args()
    cfg = PTConfig()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = generate(cfg)
    for name, df in tables.items():
        fname = "_truth_do_not_model.csv" if name == "truth" else f"{name}.csv"
        df.to_csv(out / fname, index=False)
    wear = build_daily_activity(tables, cfg, fitbit_path=a.fitbit)
    wear.to_csv(out / "daily_activity.csv", index=False)
    xlsx = build_workbook(tables, cfg, out / "personal_training_synthetic.xlsx")
    print({k: len(v) for k, v in tables.items()}, "daily_activity:", len(wear))
    print("Wrote", out)
    print("Open the workbook in Excel (formulas calculate on open).")


if __name__ == "__main__":
    main()
