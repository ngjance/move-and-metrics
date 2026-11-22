# Data

Raw files are git-ignored. Download them yourself and respect each dataset's licence.

## 1. FitBit Fitness Tracker Data (dashboard)
1. Open the Kaggle dataset "FitBit Fitness Tracker Data" (author: Mobius / `arashnic/fitbit`).
   Many copies exist; use the original and credit it.
2. Download it and find `dailyActivity_merged.csv`
   (it is inside one of the two date-range folders in the download).
3. Copy it to `data/raw/dailyActivity_merged.csv`.

The dashboard detects the file automatically and shows "real data" instead of synthetic clients.

## Synthetic data
`layer1_analytics/synthetic.py` generates 20 clients over 60 days with four personas
(steady, declining, erratic, rebounding). Always label it as synthetic when showing it.

