# Data

Raw files are git-ignored. Download them yourself and respect each dataset's licence.

## 1. FitBit Fitness Tracker Data (dashboard)
1. Open the Kaggle dataset "FitBit Fitness Tracker Data" (author: Mobius / `arashnic/fitbit`).
   Many copies exist; use the original and credit it.
2. Download it and find `dailyActivity_merged.csv`
   (it is inside one of the two date-range folders in the download).
3. Copy it to `data/raw/dailyActivity_merged.csv`.

The dashboard detects the file automatically and shows "real data" instead of synthetic clients.

## 2. Gym Customers Features and Churn (churn model)
1. Find the dataset by name on Kaggle and read its licence and description.
2. Save the CSV as `data/raw/gym_churn.csv`.
3. Open it and note the **target column name** and any **ID columns**.
4. Run:
   ```bash
   python -m layer1_analytics.churn_model --csv data/raw/gym_churn.csv --target <TARGET> --drop <ID_COLUMNS>
   ```
   The script maps yes/no/true/false/0/1 targets automatically and one-hot encodes text columns.

## Synthetic data
`layer1_analytics/synthetic.py` generates 20 clients over 60 days with four personas
(steady, declining, erratic, rebounding). Always label it as synthetic when showing it.

## Never put here
Real client data from Virgin Active, Surge or anywhere else, in any form.

## Synthetic personal-training database (`data/synthetic/`)
Generated, safe to commit. Rebuild with `python -m layer1_analytics.pt_generate`
(add `--fitbit data/raw/dailyActivity_merged.csv` to use real Fitbit series for the last ~30 days).
`_truth_do_not_model.csv` holds the hidden engagement values for validation only. Never use it as model input.
