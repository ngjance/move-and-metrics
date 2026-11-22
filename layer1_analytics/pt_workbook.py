"""Write the synthetic personal-training database to an Excel workbook with live formulas.

Values (inputs) are blue; formulas are black. Derived columns (sessions left, status, package used %,
pace excluding pauses, ...) are formulas, so pasting real data into the value columns updates them.
Packages have NO expiry date and there is no lapse rule: status is Completed, Paused or Active.
"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .pt_config import PTConfig

FONT = "Arial"
NAVY = "1F3864"
BLUE = Font(name=FONT, size=10, color="0000FF")
BLACK = Font(name=FONT, size=10, color="000000")
BOLD = Font(name=FONT, size=10, bold=True)
HDR_FONT = Font(name=FONT, size=10, bold=True, color="FFFFFF")
HDR_FILL = PatternFill("solid", fgColor=NAVY)
CALC_HDR_FILL = PatternFill("solid", fgColor="5B7DB1")
INPUT_FILL = PatternFill("solid", fgColor="FFFF99")
DATE = "yyyy-mm-dd"
MAXS, MAXP, MAXC = 6000, 500, 200   # formula ranges leave room for pasted rows


def _header(ws, headers, calc_from=None):
    for i, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=i, value=h)
        c.font, c.alignment = HDR_FONT, Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.fill = CALC_HDR_FILL if (calc_from and i >= calc_from) else HDR_FILL
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 32


def _val(v):
    return None if (v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NaT) else v


def _write_values(ws, df, cols, start_col=1):
    for r, row in enumerate(df[cols].itertuples(index=False), 2):
        for j, v in enumerate(row):
            v = _val(v)
            c = ws.cell(row=r, column=start_col + j, value=v)
            c.font = BLUE
            if hasattr(v, "year"):
                c.number_format = DATE


def _widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def build_workbook(tables, cfg: PTConfig, path):
    wb = Workbook()
    wb.calculation.fullCalcOnLoad = True

    # ------------------------------------------------------------------ Read me
    ws = wb.active
    ws.title = "Read me"
    lines = [
        ("Personal training: SYNTHETIC client database", True),
        ("Everything here is made up for a portfolio project. No real client data. "
         "Do not present results from it as real findings.", False),
        ("", False),
        ("How the sheets connect", True),
        ("Clients -> Packages -> Sessions / Pauses / Outcomes, linked by client_id and package_id. "
         "Measurements link by client_id.", False),
        ("", False),
        ("Packages have no expiry date and there is no lapse rule", True),
        ("A package stays open until its sessions are used up. Status is a formula: Completed (no sessions left), "
         "Paused (an ongoing pause), otherwise Active. A client who stops attending stays Active; use the "
         "days_since_last_session column to see them.", False),
        ("", False),
        ("Colour legend", True),
        ("Blue text = values (data you can overwrite).   Black text = formulas (do not overwrite).   "
         "Yellow fill = assumption cells you may edit.", False),
        ("Blue-grey column headers mark formula columns.", False),
        ("", False),
        ("Using it with your own (real, de-identified) data", True),
        ("1. Overwrite the blue value columns only, one row per client / package / session / pause / outcome.", False),
        ("2. Copy the formula columns down to cover any extra rows (formula ranges reach row 6000 for Sessions, "
         "500 for Packages, 200 for Pauses and Clients).", False),
        ("3. Update the as-of date on the Assumptions sheet, then check the Checks sheet.", False),
        ("4. Real client data needs Surge's written permission, client consent and PDPA compliance, and must stay "
         "off public repositories.", False),
        ("", False),
        ("How to record the pre/postnatal case (see client C001)", True),
        ("One package, one pause row (reason: Maternity), sessions before the pause tagged Prenatal and after it "
         "Postpartum, and an outcome row 'Completed, not renewed, Life stage change'. It is not a drop-off.", False),
        ("", False),
        ("Hidden variables", True),
        ("The generator uses a hidden engagement level per client. It is stored only in the repo's truth file, "
         "never in this workbook, and must never be given to a model.", False),
    ]
    for i, (t, b) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=t)
        c.font = Font(name=FONT, size=14 if i == 1 else 10, bold=b, color=NAVY if b else "000000")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 120

    # ------------------------------------------------------------------ Assumptions
    wa = wb.create_sheet("Assumptions")
    wa["A1"], wa["A1"].font = "Assumptions", Font(name=FONT, size=14, bold=True, color=NAVY)
    wa["A3"], wa["A3"].font = "Used by formulas (edit the yellow cells)", BOLD
    rows = [(4, "As-of date", cfg.as_of, "Data is generated up to this date. Update when you refresh."),
            (5, "Late cancel uses up a session (1 = yes, 0 = no)", int(cfg.late_cancel_forfeits),
             "PLACEHOLDER. Confirm Surge's policy."),
            (6, "No-show uses up a session (1 = yes, 0 = no)", int(cfg.no_show_forfeits),
             "PLACEHOLDER. Confirm Surge's policy.")]
    for r, a, b, c in rows:
        wa.cell(row=r, column=1, value=a).font = BLACK
        x = wa.cell(row=r, column=2, value=b)
        x.font, x.fill = BLUE, INPUT_FILL
        if r == 4:
            x.number_format = DATE
        wa.cell(row=r, column=3, value=c).font = BLACK
    wa["A8"], wa["A8"].font = "Package price list (from you)", BOLD
    for j, h in enumerate(["Sessions in package", "Price per session ($)", "Source"], 1):
        c = wa.cell(row=9, column=j, value=h)
        c.font, c.fill = HDR_FONT, HDR_FILL
    for k, size in enumerate(cfg.package_sizes):
        r = 10 + k
        wa.cell(row=r, column=1, value=size).font = BLUE
        p = wa.cell(row=r, column=2, value=cfg.price_per_session[size])
        p.font, p.fill, p.number_format = BLUE, INPUT_FILL, "$#,##0"
        wa.cell(row=r, column=3, value="You specified").font = BLACK

    wa["A14"], wa["A14"].font = ("Generator settings (documentation only: change them in pt_config.py and re-run "
                                 "python -m layer1_analytics.pt_generate)"), BOLD
    for j, h in enumerate(["Setting", "Value", "Source"], 1):
        c = wa.cell(row=15, column=j, value=h)
        c.font, c.fill = HDR_FONT, HDR_FILL
    gen = [("Number of clients", cfg.n_clients, "You specified", None),
           ("Fat loss clients (largest group, 51%)", cfg.goal_counts["Fat loss"], "You specified: majority", None),
           ("Strength clients", cfg.goal_counts["Strength"], "You specified: goal mix; counts are my split", None),
           ("Prenatal clients", cfg.goal_counts["Prenatal"], "You specified: goal mix; counts are my split", None),
           ("Postpartum clients", cfg.goal_counts["Postpartum"], "You specified: goal mix; counts are my split", None),
           ("Longevity clients (all above 55)", cfg.goal_counts["Longevity"], "You specified", None),
           ("Early cancel rate (24h+ notice, rebooked)", cfg.p_early_cancel, "You specified", "0%"),
           ("Late cancel rate (under 24h notice)", cfg.p_late_cancel, "Earlier default", "0%"),
           ("No-show rate", cfg.p_no_show, "You specified", "0%"),
           ("Share of clients with a non-maternity pause", cfg.share_random_pause, "Earlier default", "0%"),
           ("Share of clients who drop off mid-package", cfg.share_dropout, "Earlier default", "0%"),
           ("Renewal rate (average of finished packages)", cfg.renewal_rate, "Earlier default", "0%"),
           ("Share of renewers who renew early", cfg.share_early_renewal, "Placeholder", "0%"),
           ("Max rebooks of an early cancel", cfg.max_reschedules, "Placeholder", None),
           ("Days after finishing before renewal counts as Pending", cfg.pending_window_days, "Placeholder", None),
           ("Random seed", cfg.seed, "Makes the data reproducible", None)]
    addr = {}
    for k, (a, b, c, fmt) in enumerate(gen):
        r = 16 + k
        wa.cell(row=r, column=1, value=a).font = BLACK
        x = wa.cell(row=r, column=2, value=b)
        x.font = BLUE
        if fmt:
            x.number_format = fmt
        wa.cell(row=r, column=3, value=c).font = BLACK
        addr[a] = f"Assumptions!$B${r}"
    _widths(wa, [58, 22, 60])

    # ------------------------------------------------------------------ Clients
    wc = wb.create_sheet("Clients")
    ccols = ["client_id", "age", "age_band", "goal", "join_date", "referral_source", "planned_per_week",
             "weeks_pregnant_at_join", "weeks_postpartum_at_join"]
    _header(wc, ccols + ["tenure_days", "packages_count", "sessions_attended_total", "total_package_value"], calc_from=10)
    cl = tables["clients"]
    _write_values(wc, cl, ccols)
    for r in range(2, len(cl) + 2):
        wc[f"J{r}"] = f"=Assumptions!$B$4-E{r}"
        wc[f"K{r}"] = f"=COUNTIFS(Packages!$B$2:$B${MAXP},A{r})"
        wc[f"L{r}"] = f"=SUMIFS(Packages!$K$2:$K${MAXP},Packages!$B$2:$B${MAXP},A{r})"
        wc[f"M{r}"] = f"=SUMIFS(Packages!$F$2:$F${MAXP},Packages!$B$2:$B${MAXP},A{r})"
        wc[f"M{r}"].number_format = "$#,##0"
        for col in "JKLM":
            wc[f"{col}{r}"].font = BLACK
    _widths(wc, [11, 7, 10, 12, 12, 16, 12, 14, 14, 12, 12, 14, 14])

    # ------------------------------------------------------------------ Packages
    # A package_id, B client_id, C package_no, D sessions_signed, E price (formula), F value (formula),
    # G purchase_date, H start_date, I status (formula), J planned_per_week, K attended, L forfeited,
    # M left, N used %, O days paused, P first session, Q last session, R weeks active excl pause,
    # S sessions/week actual, T days since last session, U weeks to finish at pace
    wp = wb.create_sheet("Packages")
    heads = ["package_id", "client_id", "package_no", "sessions_signed", "price_per_session", "package_value",
             "purchase_date", "start_date", "status", "planned_per_week",
             "sessions_attended", "sessions_forfeited", "sessions_left", "package_used_pct", "days_paused",
             "first_session", "last_session", "weeks_active_excl_pause", "sessions_per_week_actual",
             "days_since_last_session", "weeks_to_finish_at_pace"]
    _header(wp, heads, calc_from=5)
    for c in (7, 8, 10):
        wp.cell(row=1, column=c).fill = HDR_FILL
    pk = tables["packages"]
    for r, row in enumerate(pk.itertuples(index=False), 2):
        d = row._asdict()
        for col, key in zip("ABCDGHJ", ["package_id", "client_id", "package_no", "sessions_signed",
                                       "purchase_date", "start_date", "planned_per_week"]):
            wp[f"{col}{r}"] = d[key]
            wp[f"{col}{r}"].font = BLUE
            if key.endswith("date"):
                wp[f"{col}{r}"].number_format = DATE
        S = f"Sessions!$B$2:$B${MAXS},$A{r}"
        G = f"Sessions!$G$2:$G${MAXS}"
        F = {
            "E": f"=INDEX(Assumptions!$B$10:$B$12,MATCH(D{r},Assumptions!$A$10:$A$12,0))",
            "F": f"=D{r}*E{r}",
            "I": (f'=IF(M{r}<=0,"Completed",IF(COUNTIFS(Pauses!$B$2:$B${MAXC},A{r},Pauses!$I$2:$I${MAXC},"Ongoing")>0,'
                  f'"Paused","Active"))'),
            "K": f'=COUNTIFS({S},{G},"Attended")',
            "L": (f'=Assumptions!$B$5*COUNTIFS({S},{G},"Late cancel")'
                  f'+Assumptions!$B$6*COUNTIFS({S},{G},"No-show")'),
            "M": f"=D{r}-K{r}-L{r}",
            "N": f"=IF(D{r}=0,0,(K{r}+L{r})/D{r})",
            "O": f"=SUMIFS(Pauses!$H$2:$H${MAXC},Pauses!$B$2:$B${MAXC},$A{r})",
            "P": f'=IF(K{r}=0,"",_xlfn.MINIFS(Sessions!$D$2:$D${MAXS},{S},{G},"Attended"))',
            "Q": f'=IF(K{r}=0,"",_xlfn.MAXIFS(Sessions!$D$2:$D${MAXS},{S},{G},"Attended"))',
            "R": f'=IF(K{r}=0,"",MAX(1,Q{r}-P{r}-O{r})/7)',
            "S": f'=IF(R{r}="","",K{r}/R{r})',
            "T": f'=IF(Q{r}="",Assumptions!$B$4-H{r},Assumptions!$B$4-Q{r})',
            "U": f'=IF(OR(S{r}="",S{r}=0),"",MAX(0,M{r})/S{r})',
        }
        for col, f in F.items():
            wp[f"{col}{r}"] = f
            wp[f"{col}{r}"].font = BLACK
        wp[f"E{r}"].number_format = wp[f"F{r}"].number_format = "$#,##0"
        wp[f"N{r}"].number_format = "0.0%"
        wp[f"P{r}"].number_format = wp[f"Q{r}"].number_format = DATE
        wp[f"R{r}"].number_format = wp[f"S{r}"].number_format = wp[f"U{r}"].number_format = "0.0"
    _widths(wp, [11, 10, 10, 13, 13, 13, 13, 12, 11, 13, 13, 13, 12, 13, 11, 13, 13, 16, 16, 16, 16])

    # ------------------------------------------------------------------ Pauses
    wz = wb.create_sheet("Pauses")
    zcols = ["pause_id", "package_id", "client_id", "pause_start", "pause_end", "reason_category"]
    _header(wz, zcols + ["sessions_left_at_pause", "days_paused", "pause_status"], calc_from=7)
    zz = tables["pauses"]
    _write_values(wz, zz, zcols)
    for r in range(2, len(zz) + 2):
        S = f"Sessions!$B$2:$B${MAXS},$B{r}"
        G = f"Sessions!$G$2:$G${MAXS}"
        D = f'Sessions!$D$2:$D${MAXS},"<"&$D{r}'
        wz[f"G{r}"] = (f"=INDEX(Packages!$D$2:$D${MAXP},MATCH(B{r},Packages!$A$2:$A${MAXP},0))"
                       f'-COUNTIFS({S},{D},{G},"Attended")'
                       f'-Assumptions!$B$5*COUNTIFS({S},{D},{G},"Late cancel")'
                       f'-Assumptions!$B$6*COUNTIFS({S},{D},{G},"No-show")')
        wz[f"H{r}"] = f'=IF(E{r}="",0,E{r}-D{r})'
        wz[f"I{r}"] = f'=IF(E{r}="","Ongoing","Ended")'
        for col in "GHI":
            wz[f"{col}{r}"].font = BLACK
    _widths(wz, [10, 11, 10, 13, 13, 16, 20, 12, 12])

    # ------------------------------------------------------------------ Sessions
    wsn = wb.create_sheet("Sessions")
    _header(wsn, ["session_id", "package_id", "client_id", "date", "weekday", "phase", "outcome",
                  "cancel_notice_hours", "reschedule_attempt"], calc_from=5)
    for c in (1, 2, 3, 4, 6, 7, 8, 9):
        wsn.cell(row=1, column=c).fill = HDR_FILL
    ss = tables["sessions"]
    for r, row in enumerate(ss.itertuples(index=False), 2):
        d = row._asdict()
        vals = {"A": d["session_id"], "B": d["package_id"], "C": d["client_id"], "D": d["date"],
                "F": d["phase"], "G": d["outcome"], "H": _val(d["cancel_notice_hours"]),
                "I": d["reschedule_attempt"]}
        for col, v in vals.items():
            c = wsn[f"{col}{r}"]
            c.value, c.font = v, BLUE
        wsn[f"D{r}"].number_format = DATE
        wsn[f"E{r}"] = f'=TEXT(D{r},"ddd")'
        wsn[f"E{r}"].font = BLACK
    _widths(wsn, [11, 11, 10, 12, 10, 12, 14, 18, 18])

    # ------------------------------------------------------------------ Outcomes
    wo = wb.create_sheet("Outcomes")
    ocols = ["outcome_id", "package_id", "client_id", "end_date", "outcome", "renewed", "reason_category",
             "renewed_package_id"]
    _header(wo, ocols)
    _write_values(wo, tables["outcomes"], ocols)
    _widths(wo, [11, 11, 10, 12, 26, 10, 20, 18])

    # ------------------------------------------------------------------ Measurements
    wm = wb.create_sheet("Measurements")
    mcols = ["measurement_id", "client_id", "date", "measure_name", "value", "unit"]
    _header(wm, mcols)
    _write_values(wm, tables["measurements"], mcols)
    _widths(wm, [15, 10, 12, 26, 9, 8])

    # ------------------------------------------------------------------ Checks
    wk = wb.create_sheet("Checks")
    wk["A1"], wk["A1"].font = "Checks (all formulas)", Font(name=FONT, size=14, bold=True, color=NAVY)
    for j, h in enumerate(["Check", "Result", "Target", "Status", "Note"], 1):
        c = wk.cell(row=3, column=j, value=h)
        c.font, c.fill = HDR_FONT, HDR_FILL
    SG = f"Sessions!$G$2:$G${MAXS}"
    CG = f"Clients!$D$2:$D${MAXC}"
    CA = f"Clients!$B$2:$B${MAXC}"
    PS = f"Packages!$I$2:$I${MAXP}"
    OF = f"Outcomes!$F$2:$F${MAXC}"
    chk = [
        ("Booking attempts (session rows)", f"=COUNTA(Sessions!$A$2:$A${MAXS})", None, None, "#,##0", "Includes rebooked attempts."),
        ("Attended share", f'=COUNTIF({SG},"Attended")/B4', None, None, "0.0%", "Remainder after the three rates below."),
        ("Early cancel rate", f'=COUNTIF({SG},"Early cancel")/B4', f"={addr['Early cancel rate (24h+ notice, rebooked)']}", "rate", "0.0%",
         "Per booking attempt; includes fade-out cancels before a drop-off."),
        ("Late cancel rate", f'=COUNTIF({SG},"Late cancel")/B4', f"={addr['Late cancel rate (under 24h notice)']}", "rate", "0.0%", ""),
        ("No-show rate", f'=COUNTIF({SG},"No-show")/B4', f"={addr['No-show rate']}", "rate", "0.0%", ""),
        ("Clients", f"=COUNTA(Clients!$A$2:$A${MAXC})", f"={addr['Number of clients']}", "eq", "0", ""),
        ("Fat loss clients", f'=COUNTIF({CG},"Fat loss")', f"={addr['Fat loss clients (largest group, 51%)']}", "eq", "0", ""),
        ("Strength clients", f'=COUNTIF({CG},"Strength")', f"={addr['Strength clients']}", "eq", "0", ""),
        ("Prenatal clients", f'=COUNTIF({CG},"Prenatal")', f"={addr['Prenatal clients']}", "eq", "0", ""),
        ("Postpartum clients", f'=COUNTIF({CG},"Postpartum")', f"={addr['Postpartum clients']}", "eq", "0", ""),
        ("Longevity clients", f'=COUNTIF({CG},"Longevity")', f"={addr['Longevity clients (all above 55)']}", "eq", "0", ""),
        ("Fat loss share of clients", "=B10/B9", 0.5, "gt", "0.0%", "Majority means above 50%."),
        ("Youngest Longevity client (age)", f'=_xlfn.MINIFS({CA},{CG},"Longevity")', 55, "gt", "0", "Must be above 55."),
        ("Oldest non-Longevity client (age)", f'=_xlfn.MAXIFS({CA},{CG},"<>Longevity")', 55, "lt", "0", "Should be 55 or below."),
        ("Packages where used sessions exceed signed",
         f"=SUMPRODUCT(--(Packages!$K$2:$K${MAXP}+Packages!$L$2:$L${MAXP}>Packages!$D$2:$D${MAXP}))", 0, "eq", "0", ""),
        ("Session rows dated after the as-of date", f'=COUNTIF(Sessions!$D$2:$D${MAXS},">"&Assumptions!$B$4)', 0, "eq", "0", ""),
        ("Packages: Active", f'=COUNTIF({PS},"Active")', None, None, "0", ""),
        ("Packages: Completed", f'=COUNTIF({PS},"Completed")', None, None, "0", ""),
        ("Packages: Paused", f'=COUNTIF({PS},"Paused")', None, None, "0", ""),
        ("Renewal rate (decided outcomes only)", f'=COUNTIF({OF},"Yes")/(COUNTIF({OF},"Yes")+COUNTIF({OF},"No"))',
         f"={addr['Renewal rate (average of finished packages)']}", "rate", "0.0%",
         "Small sample: expect noise. Only finished packages have a renewal decision."),
        ("Pauses recorded", f"=COUNTA(Pauses!$A$2:$A${MAXC})", None, None, "0", ""),
        ("Planted case: C001 sessions left at maternity pause",
         f'=INDEX(Pauses!$G$2:$G${MAXC},MATCH("C001",Pauses!$C$2:$C${MAXC},0))', 10, "eq", "0",
         "Prenatal client pauses with exactly 10 left."),
    ]
    for k, (name, f, tgt, kind, fmt, note) in enumerate(chk):
        r = 4 + k
        wk.cell(row=r, column=1, value=name).font = BLACK
        b = wk.cell(row=r, column=2, value=f)
        b.font, b.number_format = BLACK, fmt
        if tgt is not None:
            t = wk.cell(row=r, column=3, value=tgt)
            t.font, t.number_format = BLACK, fmt
            st = {"rate": f'=IF(ABS(B{r}-C{r})<=0.03,"OK","Review")', "eq": f'=IF(B{r}=C{r},"OK","Review")',
                  "gt": f'=IF(B{r}>C{r},"OK","Review")', "lt": f'=IF(B{r}<=C{r},"OK","Review")'}[kind]
            wk.cell(row=r, column=4, value=st).font = BLACK
        wk.cell(row=r, column=5, value=note).font = BLACK
    _widths(wk, [52, 14, 12, 10, 62])

    wb.save(path)
    return path
