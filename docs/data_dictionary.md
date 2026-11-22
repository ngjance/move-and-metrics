# Data dictionary

## Fitbit daily activity (`dailyActivity_merged.csv`, columns used)

| Column | Meaning | Notes |
|---|---|---|
| `Id` | Anonymous user identifier | |
| `ActivityDate` | Calendar date | Parsed to datetime |
| `TotalSteps` | Steps recorded that day | 0 can mean the tracker was not worn |
| `VeryActiveMinutes` | Minutes of vigorous activity | |
| `FairlyActiveMinutes` | Minutes of moderate activity | |
| `LightlyActiveMinutes` | Minutes of light activity | |
| `SedentaryMinutes` | Minutes sedentary | |
| `Calories` | Estimated calories burned | Device estimate |

## Derived columns (`features.py`)

| Column | Definition |
|---|---|
| `ActiveMinutes` | `VeryActiveMinutes + FairlyActiveMinutes` |
| `IsActiveDay` | `TotalSteps >= ACTIVE_STEP_THRESHOLD` (default 2000) |
| `inactive_streak_days` | Consecutive inactive days at the end of the data (missing dates count as inactive) |
| `change_vs_prior` | Mean steps of last 14 days vs the 14 days before |
| `status` | High if streak >= 3 or change <= -30%; Watch if change <= -15% or <= 3 active days in last 7; else OK |

Thresholds live in `layer1_analytics/config.py`. Document why you chose yours.

---

# Synthetic personal-training database (`data/synthetic/`)

All tables link by `client_id` and `package_id`. Sessions left, package used and pace are **derived**
(formulas in the workbook, `pt_features.py` in pandas), never typed.

| Table | One row per | Key columns |
|---|---|---|
| `clients` | client | client_id, age, age_band, goal (Fat loss / Strength / Prenatal / Postpartum / Longevity), join_date, referral_source, planned_per_week, weeks_pregnant_at_join, weeks_postpartum_at_join |
| `packages` | package | package_id, client_id, package_no, sessions_signed (20/50/100), price_per_session (195/165/155), purchase_date, start_date, status (Active / Completed / Paused; no expiry dates, no lapse rule), planned_per_week |
| `pauses` | planned break | pause_id, package_id, pause_start, pause_end (blank = ongoing), reason_category (Maternity, Travel, Work, Illness, Injury) |
| `sessions` | booking attempt | session_id, package_id, date, phase (Prenatal / Postpartum / Standard), outcome (Attended / Early cancel / Late cancel / No-show), cancel_notice_hours, reschedule_attempt |
| `outcomes` | finished package | outcome_id, package_id, end_date, outcome (Completed only; open packages have no outcome), renewed (Yes / No / Pending), reason_category, renewed_package_id |
| `measurements` | measure at a check-in | measurement_id, client_id, date, measure_name (Body weight kg, Body fat percentage %, Skeletal muscle mass kg, plus one goal-specific measure), value, unit |
| `daily_activity` | client-day with data | Fitbit-style columns, `Id` = client_id, `source` (synthetic = whole history; fitbit_shifted = real Fitbit series, last ~30 days only), `fitbit_id`. A missing day = not worn or not synced |
| `_truth_do_not_model` | client | hidden engagement and scenario tags, for validation only |

## Definitions
- **Early cancel:** cancelled with 24h+ notice. The session is not used and is rebooked (up to 2 times).
- **Late cancel / no-show:** under 24h notice, or did not attend. Both use up a session (placeholder policy; change on the Assumptions sheet).
- **Sessions left** = signed - attended - forfeited.
- **Status:** Completed if no sessions are left; Paused if a pause has no end date; otherwise Active. There are no expiry dates and no lapse rule; `days_since_last_session` shows who has gone quiet.
- **Weeks active excl. pause** = (last attended - first attended - completed pause days) / 7, minimum 1 day.
- **Renewed = Pending:** package finished fewer than 14 days before the as-of date. Exclude from model labels.
- **Maternity pause is not churn.** Keep it out of churn labels or tag it as a separate life-stage group.

---

# Renewal risk (`data/synthetic/renewal_risk.csv`, produced by `pt_risk.py`)

One row per package with sessions left (Active or Paused) or a renewal pending.

| Column | Meaning |
|---|---|
| `priority` | High / Watch / OK (from the model's estimated chance of not renewing), or Too early / Paused / Renewal pending (not scored) |
| `model_risk` | Estimated chance the client does not renew (blank when not scored) |
| `model_drivers` | The features pushing that estimate up, in plain English |
| `note` | Explanation for rows that are not scored |
| `pace_vs_plan_4wk` | Attended sessions per active week over the last 28 days divided by planned per week (paused days excluded) |
| `avg_steps_30`, `fitbit_days_30` | Average daily steps, and share of days with Fitbit data, over the 30 days before the as-of date |

## Model definition
- **Unit:** one past package with a decided outcome. Pending renewals and "Life stage change" non-renewals are excluded.
- **Label:** 1 = did not renew.
- **Snapshot:** features use only data up to the date half the package was used, or halfway through its planned schedule (sessions / planned per week), whichever is first (never earlier than day 14).
- **Inputs:** pace vs plan (overall and last 4 weeks), days since last session, late-cancel/no-show and early-cancel rates (overall and last 4 weeks), weeks to use the remaining sessions at the recent pace, had a pause, package number, size, planned sessions per week, age, goal, and activity over the 30 days before the snapshot: `steps_30`, `active_min_30`, `wear_30` (share of days with data), `steps_trend_30` (second half vs first half of the window). Missing values are filled with neutral defaults.
  `used_pct` and `days_since_start` are shown on the dashboard but are not model inputs.
- **Validation:** grouped cross-validation by client across 10 synthetic cohorts, plus the held-out 35-client cohort.

---

# Association analysis (`docs/analysis/`, produced by `pt_analysis.py`)

`analysis_frame_cohort.csv` has one row per **finished package with a renewal decision** (Yes/No), with whole-package exposures:

| Column | Meaning |
|---|---|
| `skip_rate` | (early cancels + late cancels + no-shows) / booking attempts |
| `forfeit_rate`, `early_rate` | Late cancels + no-shows, and early cancels only, over booking attempts |
| `had_pause`, `n_pauses`, `pause_days`, `longest_pause` | Completed pauses in the package (days; 0 if none) |
| `maternity_pause` | 1 if any pause had reason Maternity |
| `steps_30`, `active_min_30`, `wear_30`, `steps_trend_30` | Activity over the 30 days before the package finished (see the model definition) |
| `baseline_weight`, `baseline_bodyfat`, `baseline_smm` | Body weight, body fat % and skeletal muscle mass at (or just before) the start of the package |
| `delta_weight`, `delta_bodyfat`, `delta_smm` | Change from that baseline to the last check-in up to the end of the package (needs 28+ days between them) |
| `goal_progress` | 1 if body fat fell (fat loss, postpartum) or muscle rose (strength, longevity); blank otherwise |
| `not_renewed` | 1 = did not renew (outcome) |
| `cluster` | Client key used to resample whole clients in the bootstrap |

Other files: `association_*.csv` (effect, 95% CI, p, Holm-adjusted p, verdict), `dose_response_*.csv`, `adjusted_model_*.csv`,
`beyond_skip_rate_*.csv` (does a factor add anything after skipped sessions?), `model_comparison_activity.csv`, `validation.csv` (planted-effect test), `report.md` and charts. "cohort" = the 35-client data; "pooled" = it plus 10 other synthetic cohorts.
