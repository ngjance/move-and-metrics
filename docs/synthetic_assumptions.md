# Synthetic data assumptions

Everything in `data/synthetic/` is made up. Settings live in `layer1_analytics/pt_config.py`.

## Specified by the author
35 clients; package sizes 20 / 50 / 100 priced at $195 / $165 / $155 per session; early cancel 30%;
no-show 1%; goal mix of fat loss (the majority, 18 of 35), strength, prenatal, postpartum and
longevity (all above 55).

## Earlier defaults
Late cancel 10%; about 8% of clients pause for non-maternity reasons; about 12% drop off mid-package;
average renewal rate 55% (the 35-client cohort lands near 49%; expect noise).

## No expiry dates and no lapse rule (confirmed by the author)
A package stays open until its sessions are used up. There is no expiry and no rule that turns inactivity into a "lapsed" status.
A client who stops attending stays Active with sessions left, and their package never gets an outcome row.

## Placeholders (replace with real policies and experience)
Late cancels and no-shows use up a session; an early cancel is
rebooked up to 2 times; 30% of renewers renew early; split of the non-fat-loss goals (7 / 3 / 4 / 3);
age ranges; planned sessions per week by goal; reasons for leaving; join-date window.

## How renewal is generated (an assumption, not a fact)
Renewal odds depend on three things: a hidden engagement level, how the client actually behaved in that package
(share of booking attempts attended, weight 1.5 around a typical share of 0.63, tuned so the average renewal rate
is about 55%), and random chance. Real attendance does tend to predict renewal, but how strongly is my assumption.
Any model trained on this data learns that assumption.

## What the analysis can and cannot show
Renewal odds depend on in-package attendance and a hidden engagement level, and **not on pause length** (a switch,
`renewal_pause_penalty_per_30d`, plants a pause effect for method validation only, default 0). So on this data the analysis
should find a skipped-session effect and no pause effect, and it does. It validates the method, not real retention drivers.

## Wearable activity and body measurements (assumptions)
- **Daily activity** exists for each client's whole history. Steps rise on session days, depend on goal and on the hidden engagement
  level, and fall about 25% after 3+ weeks without an attended session. A missing day means the tracker was not worn or not synced;
  engaged clients wear it more (wear probability about 0.55 + 0.4 x engagement).
- **Body weight, body fat % and skeletal muscle mass** are recorded at each check-in (every 8 weeks while attending). They drift with
  attended sessions by goal (for example fat loss: weight and body fat down, muscle slightly up; strength: muscle up), scaled by
  engagement, with noise. Prenatal weight rises in pregnancy and eases postpartum.
- **No direct effect on renewal.** Neither activity nor body measures feeds the renewal decision. They relate to renewal only because
  they follow the same hidden engagement. So "less active clients renew less" here is confounding through engagement, not a causal effect.

## How realism is added
- Each client has a hidden engagement level that raises or lowers cancellations, drop-off and renewal, plus
  random noise, so no single rule determines the outcome.
- Planted cases: 3 prenatal clients (pause with exactly 10 sessions left; one finishes and does not renew,
  one finishes and renews, one returns then drops off), 3 more drop-offs, 3 non-maternity pauses, early renewals.
- Packages finished fewer than 14 days before the as-of date have renewal "Pending".

## Known limits
- Observed rates differ slightly from the targets (see the workbook's Checks sheet). With 35 clients, expect noise.
- Cancellation rates are per booking attempt and include fade-out cancels before a drop-off.
- Measurements and wearable data are simple simulations.
- Models trained on this data demonstrate a method. They do not show real drivers of client retention.
- Fitbit data (if used) is from 2016 and covers about 30 users for about a month.
