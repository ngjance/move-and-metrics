# Layer 2: Coach Copilot (design)

**Goal:** prepare a weekly check-in brief for each client so a trainer spends minutes, not
hours, and never sends anything they have not read and approved.

## The loop

```
1. GATHER   tools: get_trends(client), get_status(client), get_risk(client), get_notes(client)
2. DRAFT    brief: wins, concerns, 2-3 talking points, one suggested adjustment
3. CHECK    score the draft against rubric.md (automatic)
4. REVISE   if any "must" fails, fix and re-check (max 2 revisions)
5. ESCALATE if a red flag is present, stop drafting and alert the trainer
6. APPROVE  trainer edits or approves in the app; nothing is sent automatically
7. LOG      save inputs, draft, checks, edits and the final text
```

Stop rules: rubric passes, or 2 revisions used (then flag "needs human rewrite"), or a red flag.

## The harness (what makes it trustworthy)

| Part | What it does |
|---|---|
| Tools | Read-only access to layer 1 outputs; no sending, no external calls |
| Guardrails | No medical advice, diagnosis or medication talk; red flags go to the trainer |
| Rubric check | Automatic pass/fail on the rules in `rubric.md` |
| Human approval | Trainer must approve every brief |
| Logging | Every run is recorded so errors can be traced |
| Test suite | Synthetic clients with planted problems; measure how many the agent catches |

## Build plan

1. Write `rubric.md` properly (draft provided). This is where your coaching expertise matters most.
2. Create 20 synthetic clients with notes and planted red flags (`tests/scenarios/`).
3. Build the loop in Python with a model API of your choice.
4. Add the approval screen (Streamlit) and logging.
5. Run the scenarios, record pass rates, fix failures, write up the results honestly, including failures.

## Metrics to report

- Red flags correctly escalated (target: all of them; report misses)
- Rubric pass rate on first draft and after revision
- Average revisions needed
- Trainer edit rate (how much of each brief a trainer changes)
- Time to prepare a brief versus doing it by hand

## Privacy

Public repo: synthetic data only. A pilot with real clients (for example at Surge) needs written
permission, client consent, de-identified data and PDPA compliance, and should run privately.
