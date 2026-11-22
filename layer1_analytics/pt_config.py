"""All settings for the personal-training synthetic data generator, in ONE place.

Edit values here and re-run:  python -m layer1_analytics.pt_generate
Each setting says whether it came from you or is a placeholder to replace with your real-world sense.
"""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class PTConfig:
    seed: int = 2026
    as_of: date = date(2026, 11, 15)            # "today" for the data; sessions stop here

    # ---- specified by you -------------------------------------------------------------
    n_clients: int = 35
    package_sizes: tuple = (20, 50, 100)
    price_per_session: dict = field(default_factory=lambda: {20: 195, 50: 165, 100: 155})
    p_early_cancel: float = 0.30                # cancelled with >= 24h notice; rebooked
    p_no_show: float = 0.01
    # fat loss is the clear majority (18 of 35 = 51%); longevity clients are all above 55
    goal_counts: dict = field(default_factory=lambda: {
        "Fat loss": 18, "Strength": 7, "Prenatal": 3, "Postpartum": 4, "Longevity": 3})

    # ---- my defaults from earlier in our chat (change to match your experience) --------
    p_late_cancel: float = 0.10                 # cancelled with < 24h notice
    share_random_pause: float = 0.08            # non-maternity pauses (travel, injury, work, illness)
    share_dropout: float = 0.12                 # stop attending mid-package (includes 1 planted case)
    renewal_rate: float = 0.55                  # of finished packages (average; varies by client)

    # ---- PLACEHOLDERS: not given by you, replace with your real policies ---------------
    package_size_weights: tuple = (0.40, 0.40, 0.20)
    late_cancel_forfeits: bool = True           # a late cancel uses up a session
    no_show_forfeits: bool = True
    max_reschedules: int = 2                    # an early cancel is rebooked up to this many times
    renewal_typical_attendance_share: float = 0.63   # attendance share that leaves renewal odds unchanged (tuned so the average renewal rate is about 55%)
    renewal_pause_penalty_per_30d: float = 0.0   # VALIDATION ONLY: plants a pause effect on renewal odds (0 = none). Used to test the analysis.
    renewal_behaviour_weight: float = 1.5      # how strongly in-package attendance moves renewal odds
    share_early_renewal: float = 0.30           # of renewers, buy the next package before finishing
    pending_window_days: int = 14               # package finished this recently => renewal "Pending"
    join_window: tuple = (date(2025, 9, 1), date(2026, 9, 30))
    age_ranges: dict = field(default_factory=lambda: {
        "Fat loss": (28, 54), "Strength": (22, 50), "Prenatal": (27, 37),
        "Postpartum": (28, 40), "Longevity": (56, 72)})   # Longevity: above 55
    planned_per_week_choices: dict = field(default_factory=lambda: {
        "Fat loss": (2, 3), "Strength": (2, 3), "Prenatal": (1, 2),
        "Postpartum": (1, 2), "Longevity": (1, 2)})
    referral_sources: tuple = ("Referral", "Walk-in", "Online", "Corporate", "Social media")
    non_renewal_reasons: dict = field(default_factory=lambda: {
        "Reached goal": 0.30, "Price": 0.20, "Schedule change": 0.25,
        "Relocated": 0.10, "No reply": 0.15})
    dropout_reasons: tuple = ("Lost motivation", "Schedule change", "Price", "No reply")
    pause_reasons: tuple = ("Travel", "Work", "Illness", "Injury")

    # ---- renewal-risk model settings ----
    min_days_to_score: int = 14             # packages younger than this are "too early"
    min_used_pct_to_score: float = 0.10
    # Model thresholds, checked on out-of-fold predictions for 291 past synthetic packages:
    #   0.60 flags about 1 in 4 packages and 74% of those did not renew
    #   0.50 flags about 2 in 5 packages and 70% of those did not renew (base rate 44%)
    model_high: float = 0.60                # estimated chance of not renewing
    model_watch: float = 0.50
    snapshot_used_pct: float = 0.50         # training snapshot: when half the package is used

