"""Synthetic clients so the project runs without any downloads.

Schema mirrors the columns we use from the Fitbit daily activity file. Four personas
let you check that the dashboard and rules catch what they should:
steady, declining (drop-off), erratic, rebounding.
This is SYNTHETIC data: label it as such anywhere you show it.
"""
import numpy as np
import pandas as pd

PERSONAS = ["steady", "declining", "erratic", "rebounding"]


def generate_clients(n_clients: int = 20, days: int = 60, start: str = "2026-09-01", seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    rows, truth = [], []
    for i in range(n_clients):
        persona = PERSONAS[i % len(PERSONAS)]
        client_id = 1_000_000_000 + i
        base = int(rng.integers(5000, 11000))
        truth.append({"Id": client_id, "persona": persona})
        for k, d in enumerate(dates):
            t = k / (days - 1)
            if persona == "steady":
                mult = 1.0
            elif persona == "declining":
                mult = 1.0 - 0.8 * max(0.0, (t - 0.5) / 0.5)
            elif persona == "erratic":
                mult = float(rng.choice([0.2, 1.0, 1.6], p=[0.3, 0.4, 0.3]))
            else:  # rebounding
                mult = 0.4 + 0.8 * t
            steps = max(0, int(rng.normal(base * mult, base * 0.2)))
            if persona == "declining" and t > 0.9 and rng.random() < 0.7:
                steps = 0  # tracker off / no activity
            if steps == 0:
                very = fairly = lightly = 0
                sedentary = 1440
                calories = 1500
            else:
                very = int(steps / 1000 * rng.uniform(1, 3))
                fairly = int(steps / 1000 * rng.uniform(0.5, 2))
                lightly = int(min(400, steps / 40 * rng.uniform(0.8, 1.2)))
                sedentary = max(0, 1440 - very - fairly - lightly - int(rng.integers(300, 500)))
                calories = int(1500 + steps * 0.04 + very * 8)
            rows.append({
                "Id": client_id, "ActivityDate": d, "TotalSteps": steps,
                "VeryActiveMinutes": very, "FairlyActiveMinutes": fairly,
                "LightlyActiveMinutes": lightly, "SedentaryMinutes": sedentary,
                "Calories": calories,
            })
    return pd.DataFrame(rows), pd.DataFrame(truth)
