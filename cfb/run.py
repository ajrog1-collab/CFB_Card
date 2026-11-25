"""
Builds docs/data.json for the site.

Run by GitHub Actions twice a week. Also safe to run by hand.

  1. Refresh data (history is cached; current season always refetched)
  2. Fit ratings and the preseason prior
  3. Backtest for context
  4. Find qualified wagers among upcoming games
  5. Append new ones to data/bet_log.csv, grade any that have finished
  6. Write docs/data.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model import (
    PriorModel, TIER_NAMES, add_adjusted_margin, add_situational, apply_debias,
    assign_tier, attach_epa,
    attach_lines, attach_turnovers, attach_venue_weather, blend_prior, blend_ratings,
    build_games, cfbd_get, col, confidence_features, confidence_score,
    fetch_venue_forecast, fetch_venue_weather, fit_calibration, fit_debias,
    fit_points_ratings, fit_ratings, fit_uncertainty, games_played_counts,
    api_usage, debias_like_board, infer_home_venues, parse_qb_value, parse_sp, parse_team_stats, parse_teams,
    parse_venues,
    predict_total, predict_uncertainty, rating_diff, regime_cuts,
    relative_tiers,
    returning_production, tier_cutpoints,
    season_ratings, situational_matrix, talent_composite, totals_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config.json").read_text())
BET_LOG = ROOT / "data" / "bet_log.csv"
OUT = ROOT / "docs" / "data.json"

HIST_YEARS = list(range(CONFIG["history_start"], CONFIG["current_season"]))
CURRENT = CONFIG["current_season"]
MIN_EDGE = float(CONFIG["min_edge"])
BLEND_K = float(CONFIG["blend_k"])
TEST_SEASONS = CONFIG["backtest_seasons"]
HOLDOUT = CONFIG.get("holdout_season")
TO_SHRINK = float(CONFIG.get("turnover_shrink", 0.7))
ALLOW_WEEK_BACKFILL = bool(CONFIG.get("allow_week_backfill", True))
RATING_TARGET = "adj_margin"
DEBIAS = bool(CONFIG.get("debias", True))
AUTO_THRESHOLD = bool(CONFIG.get("auto_threshold", True))
TARGET_QUALIFIED = float(CONFIG.get("target_qualified_share", 0.25))
MIN_EDGE_TOTAL = float(CONFIG.get("min_edge_total", 4.0))
TOP_PICK_COUNT = int(CONFIG.get("top_pick_count", 5))
CONF_TIERS = CONFIG.get("confidence_tiers", [0.2, 0.35, 0.5, 0.7])
LOOKAHEAD_DAYS = int(CONFIG["lookahead_days"])
PRICE = float(CONFIG["assumed_price"])  # e.g. -110


def american_to_profit(price: float) -> float:
    """Profit per 1 unit risked on a win."""
    return (100.0 / abs(price)) if price < 0 else (price / 100.0)


_YEAR_LEVEL_TEAM_STATS_WORKS = True


def fetch_team_stats(year: int) -> pd.DataFrame:
    """Team box scores for a season, used for turnover counts.

    /games/teams rejects a year-only query on current API versions, so we try it
    once and then stop wasting a call per season on it, falling back to a
    week-by-week loop (~16 calls).

    Completed seasons cache permanently — those box scores will never change.
    The current season must NOT: caching it freezes turnover data at whatever
    games had been played the first time this ran, so every rating built on
    turnover-adjusted margin silently stops learning from new results.
    """
    global _YEAR_LEVEL_TEAM_STATS_WORKS
    live = (year == CURRENT)

    if _YEAR_LEVEL_TEAM_STATS_WORKS:
        ts = cfbd_get("games/teams", {"year": year, "seasonType": "regular"},
                      force=live, required=False)
        if len(ts):
            return ts
        _YEAR_LEVEL_TEAM_STATS_WORKS = False
        print("    games/teams: year-level query rejected, using week-by-week from here")
    if not ALLOW_WEEK_BACKFILL:
        return pd.DataFrame()

    # For the live season, only the last few weeks can still change — earlier
    # weeks are final and stay cached, so this stays cheap.
    frames = []
    for wk in range(1, 17):
        w = cfbd_get("games/teams", {"year": year, "week": wk, "seasonType": "regular"},
                     force=(live and wk >= _current_week() - 1), required=False)
        if len(w):
            frames.append(w)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _current_week(default: int = 1) -> int:
    """Latest week with a completed game, from the schedule itself.

    An earlier version guessed this from the calendar with a hardcoded season
    start and 7-day weeks. That drifts against the real schedule, and it decides
    which weeks of box scores get refreshed — so drifting means turnover data
    silently stops updating.
    """
    if _LIVE_WEEK[0] is not None:
        return _LIVE_WEEK[0]
    return default


_LIVE_WEEK = [None]


def set_live_week(cur_games_frame):
    """Record the newest week that has a final score, for cache decisions."""
    try:
        played = cur_games_frame[cur_games_frame["home_pts"].notna()]
        if len(played):
            _LIVE_WEEK[0] = int(played["week"].max())
        else:
            _LIVE_WEEK[0] = 1
    except Exception:
        _LIVE_WEEK[0] = None
    return _LIVE_WEEK[0]


# ======================================================================
# 1. data
# ======================================================================

def load_everything():
    print("Fetching data...")
    games_raw, lines_raw, adv_raw = [], [], []
    recruit, returning, fbs = [], [], {}
    teams_raw = pd.DataFrame()
    sp_by_year, team_stats = {}, []

    for yr in HIST_YEARS:
        for st in ("regular", "postseason"):
            g = cfbd_get("games", {"year": yr, "seasonType": st})
            if len(g):
                games_raw.append(g)
        l = cfbd_get("lines", {"year": yr, "seasonType": "regular"})
        if len(l):
            lines_raw.append(l)
        a = cfbd_get("stats/game/advanced",
                     {"year": yr, "seasonType": "regular", "excludeGarbageTime": "true"},
                     required=False)
        if len(a):
            adv_raw.append(a)
        t = cfbd_get("teams/fbs", {"year": yr})
        if len(t):
            c = col(t, "school", "team")
            fbs[yr] = set(t[c]) if c else set()
        rc = cfbd_get("recruiting/teams", {"year": yr}, required=False)
        if len(rc):
            recruit.append(rc.assign(_year=yr))
        rp = cfbd_get("player/returning", {"year": yr}, required=False)
        if len(rp):
            returning.append(rp.assign(_year=yr))

        # SP+ (used only as a PRIOR-year feature, never within-season)
        sp = cfbd_get("ratings/sp", {"year": yr}, required=False)
        parsed = parse_sp(sp)
        if parsed:
            sp_by_year[yr] = parsed

        # team box scores, for turnovers
        ts = fetch_team_stats(yr)
        if len(ts):
            team_stats.append(ts)

        print(f"  {yr}")

    # current season: never cached, always fresh
    t = cfbd_get("teams/fbs", {"year": CURRENT}, cache_empty=False)
    # a cache written before logos were requested would have no logo column;
    # refetch once so team marks are never stuck on initials
    if len(t) and not any(c in t.columns for c in ("logos", "logo")):
        t = cfbd_get("teams/fbs", {"year": CURRENT}, force=True)
    teams_raw = t
    if len(t):
        c = col(t, "school", "team")
        fbs[CURRENT] = set(t[c]) if c else set()
    rc = cfbd_get("recruiting/teams", {"year": CURRENT}, required=False, cache_empty=False)
    if len(rc):
        recruit.append(rc.assign(_year=CURRENT))
    rp = cfbd_get("player/returning", {"year": CURRENT}, required=False, cache_empty=False)
    if len(rp):
        returning.append(rp.assign(_year=CURRENT))

    qb_raw = cfbd_get("ppa/players/season", {"year": CURRENT, "position": "QB"},
                      cache_empty=False,
                      force=True, required=False)
    if not len(qb_raw):
        # nothing played yet this season, so last year's usage is the best guide
        qb_raw = cfbd_get("ppa/players/season",
                          {"year": CURRENT - 1, "position": "QB"}, required=False)
        qb_season = CURRENT - 1
    else:
        qb_season = CURRENT

    cur_games = cfbd_get("games", {"year": CURRENT, "seasonType": "regular"}, force=True)
    cur_lines = cfbd_get("lines", {"year": CURRENT, "seasonType": "regular"}, force=True)
    print(f"  {CURRENT} (live)")

    venues_raw = cfbd_get("venues", {}, required=False)

    # recruiting classes before the window, for the rolling average
    for yr in range(CONFIG["history_start"] - CONFIG["recruit_window"], CONFIG["history_start"]):
        r = cfbd_get("recruiting/teams", {"year": yr}, required=False)
        if len(r):
            recruit.append(r.assign(_year=yr))

    cat = lambda fs: pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()
    return {
        "games": cat(games_raw), "lines": cat(lines_raw), "adv": cat(adv_raw),
        "recruit": cat(recruit), "returning": cat(returning), "fbs": fbs,
        "cur_games": cur_games, "cur_lines": cur_lines,
        "sp": sp_by_year, "team_stats": cat(team_stats),
        "venues": venues_raw, "teams_meta": teams_raw,
        "qb": qb_raw, "qb_season": qb_season,
    }


# ======================================================================
# 2. tiers
# ======================================================================

def tier_stats(frame, edge_col="edge", margin_col="margin", line_col="mkt"):
    """Break results into edge tiers and measure calibration.

    For each tier we report the win rate (with its standard error, because a
    120-bet tier has a ~4.5 point error bar) and — more usefully — the average
    points by which the pick actually beat the line, next to the average edge
    the model predicted. The ratio between them is the calibration factor: if
    the model claims 8 points of edge and delivers 2, its edges are inflated 4x.
    """
    edges = CONFIG.get("tier_edges", [3, 5, 7, 10, 15])
    profit = american_to_profit(PRICE)
    out = []

    f = frame.dropna(subset=[edge_col, margin_col, line_col]).copy()
    if not len(f):
        return out

    # points by which the picked side beat the line
    f["_realized"] = np.where(f[edge_col] > 0,
                              f[margin_col] - f[line_col],
                              f[line_col] - f[margin_col])
    f["_pred"] = f[edge_col].abs()
    f["_result"] = np.where(f["_realized"] > 0, "win",
                            np.where(f["_realized"] < 0, "loss", "push"))

    bounds = [(edges[i], edges[i + 1] if i + 1 < len(edges) else None)
              for i in range(len(edges))]

    for lo, hi in bounds:
        sub = f[(f._pred >= lo) & ((f._pred < hi) if hi else True)]
        decided = sub[sub._result != "push"]
        n = int(len(sub))
        if n < 10:
            continue
        wins = int((sub._result == "win").sum())
        losses = int((sub._result == "loss").sum())
        pushes = int((sub._result == "push").sum())
        wp = float((decided._result == "win").mean()) if len(decided) else float("nan")
        se = float(np.sqrt(wp * (1 - wp) / len(decided)) * 100) if len(decided) else float("nan")
        units = wins * profit - losses
        pred = float(sub._pred.mean())
        real = float(sub._realized.mean())

        out.append({
            "label": f"{lo}-{hi}" if hi else f"{lo}+",
            "bets": n, "wins": wins, "losses": losses, "pushes": pushes,
            "win_pct": round(wp * 100, 1) if wp == wp else None,
            "se": round(se, 1) if se == se else None,
            "units": round(units, 2),
            "roi_pct": round(units / n * 100, 1) if n else None,
            "pred_edge": round(pred, 1),
            "realized_edge": round(real, 1),
            "calibration": round(real / pred, 2) if pred > 0.01 else None,
        })
    return out





# ======================================================================
# confidence tiers (A = most sure)
# ======================================================================

def tier_season_trend(s, cuts, edge_col="edge", outcome_col="margin", line_col="mkt"):
    """Per-tier, per-season results, plus an overall row for each tier.

    Answers the question the tier list cannot: does a given tier hold up year to
    year, or did one season carry it?

    Rows are tiered with the cutpoints for their own regime, so a Week 2 pick is
    graded against other Week 2 picks rather than against mid-season ones.
    """
    if not cuts or not len(s):
        return {}, {}

    f = s.dropna(subset=[edge_col, outcome_col, line_col, "confidence"]).copy()
    if not len(f):
        return {}, {}

    f["_tier"] = [assign_tier(c, regime_cuts(cuts, w))
                  for c, w in zip(f["confidence"], f["week"])]
    realized = np.where(f[edge_col] > 0, f[outcome_col] - f[line_col],
                        f[line_col] - f[outcome_col])
    f["_res"] = np.where(realized > 0, "win", np.where(realized < 0, "loss", "push"))
    profit = american_to_profit(PRICE)

    def summarize(frame):
        dec = frame[frame._res != "push"]
        if not len(dec):
            return None
        wins = int((frame._res == "win").sum())
        losses = int((frame._res == "loss").sum())
        units = wins * profit - losses
        wp = wins / (wins + losses) * 100 if (wins + losses) else None
        return {
            "bets": int(len(frame)), "wins": wins, "losses": losses,
            "pushes": int((frame._res == "push").sum()),
            "win_pct": round(wp, 1) if wp is not None else None,
            "units": round(units, 2),
            "roi_pct": round(units / len(frame) * 100, 1),
            "avg_conf": round(float(frame["confidence"].mean()), 2),
            "avg_gap": round(float(frame[edge_col].abs().mean()), 1),
        }

    trend, summary = {}, {}
    for name in TIER_NAMES:
        sub = f[f._tier == name]
        if len(sub) < 40:
            continue
        summary[name] = summarize(sub)
        seasons = []
        for season in sorted(sub["season"].dropna().unique()):
            ss = sub[sub.season == season]
            if len(ss) < 15:
                continue
            row = summarize(ss)
            if row:
                row["season"] = int(season)
                seasons.append(row)
        if seasons:
            trend[name] = seasons

    overall = summarize(f)
    if overall:
        summary["ALL"] = overall
        seasons = []
        for season in sorted(f["season"].dropna().unique()):
            ss = f[f.season == season]
            if len(ss) < 25:
                continue
            row = summarize(ss)
            if row:
                row["season"] = int(season)
                seasons.append(row)
        if seasons:
            trend["ALL"] = seasons
    return trend, summary


def combined_gap_trend(spread_seasons, total_seasons):
    """Points behind the closing line, per season, for each market and combined.

    The combined figure is weighted by pick count. Spread error and total error
    are both measured in points against the book's own number, so averaging them
    is meaningful, but they are not the same quantity — read the individual lines
    first and the combined one as a summary.
    """
    by = {}
    for row in spread_seasons or []:
        if row.get("gap") is not None:
            by.setdefault(row["season"], {})["spread"] = (row["gap"], row.get("bets", 0))
        if row.get("gap_raw") is not None:
            by.setdefault(row["season"], {})["spread_raw"] = (row["gap_raw"], row.get("bets", 0))
    for row in total_seasons or []:
        if row.get("gap") is not None:
            by.setdefault(row["season"], {})["total"] = (row["gap"], row.get("bets", 0))

    out = []
    for season in sorted(by):
        e = by[season]
        sp = e.get("spread")
        to = e.get("total")
        parts = [v for v in (sp, to) if v]
        wsum = sum(g * max(n, 1) for g, n in parts)
        nsum = sum(max(n, 1) for _g, n in parts)
        raw = e.get("spread_raw")
        out.append({
            "season": int(season),
            "spread": sp[0] if sp else None,
            "spread_raw": raw[0] if raw else None,
            "total": to[0] if to else None,
            "all": round(wsum / nsum, 2) if nsum else None,
        })
    return out


def season_breakdown(s, edge_col="edge", outcome_col="margin", line_col="mkt",
                     min_edge=None):
    """Per-season results for qualified picks, newest last.

    The useful trend column is `gap`: how much worse than the closing line the
    model was that year. Win rate bounces around on a few hundred picks; the
    accuracy gap is a far steadier read on whether the model is improving.
    """
    if not len(s):
        return []
    thr = MIN_EDGE if min_edge is None else min_edge
    profit = american_to_profit(PRICE)
    out = []

    for season in sorted(s["season"].dropna().unique()):
        sub = s[s.season == season]
        q = sub[(sub[edge_col].abs() >= thr)].dropna(subset=[outcome_col, line_col])
        if len(q) < 25:
            continue
        realized = np.where(q[edge_col] > 0, q[outcome_col] - q[line_col],
                            q[line_col] - q[outcome_col])
        wins = int((realized > 0).sum())
        losses = int((realized < 0).sum())
        pushes = int((realized == 0).sum())
        decided = wins + losses
        units = wins * profit - losses
        row = {
            "season": int(season),
            "bets": int(len(q)),
            "wins": wins, "losses": losses, "pushes": pushes,
            "win_pct": round(wins / decided * 100, 1) if decided else None,
            "units": round(units, 2),
            "roi_pct": round(units / len(q) * 100, 1) if len(q) else None,
        }
        if "e_model" in sub.columns and "e_mkt" in sub.columns:
            row["model_mae"] = round(float(sub.e_model.mean()), 2)
            row["market_mae"] = round(float(sub.e_mkt.mean()), 2)
            row["gap"] = round(float(sub.e_model.mean() - sub.e_mkt.mean()), 2)
            if "e_raw" in sub.columns and sub.e_raw.notna().any():
                row["gap_raw"] = round(float(sub.e_raw.mean() - sub.e_mkt.mean()), 2)
        out.append(row)
    return out


def bias_diagnostics(s, thr=None):
    """Is the edge systematically pointed at one kind of team?

    A healthy model disagrees with the book in both directions regardless of how
    big the number is. Correlation near zero between the market number and the
    edge is what that looks like; a strong negative reading means the model is
    just taking points everywhere.
    """
    f = s.dropna(subset=["edge", "mkt"])
    if len(f) < 100:
        return {}
    x = f["mkt"].to_numpy(dtype=float)
    y = f["edge"].to_numpy(dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
    dog = float((((x > 0) != (y > 0))).mean())
    qual = f[f.edge.abs() >= (MIN_EDGE if thr is None else thr)]
    return {
        "edge_vs_line_corr": round(corr, 3),
        "pct_on_underdog": round(dog * 100, 1),
        "qualified_share": round(len(qual) / len(f) * 100, 1),
    }


def confidence_tier_stats(frame, conf_col="confidence", outcome_col="margin",
                         line_col="mkt", edge_col="edge"):
    """Results grouped by confidence, to test whether confidence sorts winners.

    This is the honest test of the whole idea: if the top confidence group does
    not beat the bottom one, confidence is not measuring anything useful.
    """
    profit = american_to_profit(PRICE)
    f = frame.dropna(subset=[conf_col, outcome_col, line_col, edge_col]).copy()
    if len(f) < 100:
        return []

    f["_realized"] = np.where(f[edge_col] > 0, f[outcome_col] - f[line_col],
                              f[line_col] - f[outcome_col])
    f["_res"] = np.where(f["_realized"] > 0, "win",
                         np.where(f["_realized"] < 0, "loss", "push"))

    qs = list(CONF_TIERS)
    cuts = [f[conf_col].quantile(q) for q in qs]
    bounds = [(None, cuts[0])] + [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)] \
             + [(cuts[-1], None)]
    labels = ([f"bottom {int(qs[0]*100)}%"]
              + [f"{int(qs[i]*100)}-{int(qs[i+1]*100)}%" for i in range(len(qs) - 1)]
              + [f"top {int((1-qs[-1])*100)}%"])

    out = []
    for (lo, hi), label in zip(bounds, labels):
        sub = f
        if lo is not None:
            sub = sub[sub[conf_col] >= lo]
        if hi is not None:
            sub = sub[sub[conf_col] < hi]
        dec = sub[sub._res != "push"]
        if len(sub) < 30:
            continue
        wp = float((dec._res == "win").mean()) if len(dec) else float("nan")
        se = float(np.sqrt(wp * (1 - wp) / len(dec)) * 100) if len(dec) else float("nan")
        units = int((sub._res == "win").sum()) * profit - int((sub._res == "loss").sum())
        out.append({
            "label": label,
            "bets": int(len(sub)),
            "win_pct": round(wp * 100, 1) if wp == wp else None,
            "se": round(se, 1) if se == se else None,
            "units": round(units, 2),
            "roi_pct": round(units / len(sub) * 100, 1),
            "avg_conf": round(float(sub[conf_col].mean()), 2),
            "avg_edge": round(float(sub[edge_col].abs().mean()), 1),
        })
    return out


# ======================================================================
# 3. backtest
# ======================================================================

def run_backtest(d, prior_model, season_hfa, use_epa):
    preds = []
    sit_report = {}
    for season in TEST_SEASONS:
        prior = prior_model.preseason(season)
        if prior is None:
            continue
        earlier = [h for y, h in season_hfa.items() if y < season]
        hfa_ref = float(np.mean(earlier)) if earlier else 2.6

        cal = d[d.season < season]
        cal_x = rating_diff(cal, prior_model.season_r.get(season - 1), hfa_ref)
        cal_S, sit_names = situational_matrix(cal, cal_x)
        w_cal, sit_report = fit_calibration(
            cal_x, cal_S, sit_names, cal["margin"].to_numpy(dtype=float))


        for wk in sorted(d.loc[d.season == season, "week"].unique()):
            so_far = d[(d.season == season) & (d.week < wk)]
            test = d[(d.season == season) & (d.week == wk)]
            if test.empty:
                continue
            r_in = None
            if len(so_far) >= 150:
                r_m, _ = fit_ratings(so_far, RATING_TARGET, ridge=14.0, cap=35.0)
                r_p = None
                if use_epa:
                    r_p, _ = fit_ratings(so_far, "ppa_margin", ridge=1.0, cap=1.5)
                r_in = blend_ratings(r_m, r_p)
            gp = games_played_counts(so_far)
            R = blend_prior(prior, r_in, gp, BLEND_K)
            out = test.copy()
            rd = rating_diff(test, R, hfa_ref)
            S, _ = situational_matrix(test, rd)
            A = np.column_stack([rd, S, np.ones(len(test))])
            out["pred_raw"] = np.nan_to_num(A, nan=0.0) @ w_cal

            # a second opinion from the prior alone, for the disagreement feature
            out["pred_alt"] = rating_diff(test, prior, hfa_ref) * w_cal[0] + w_cal[-1]
            out["_gp_min"] = [min(gp.get(h, 0), gp.get(a, 0))
                              for h, a in zip(test["home_team"], test["away_team"])]
            preds.append(out)

    if not preds:
        return pd.DataFrame(), {}

    bt = pd.concat(preds, ignore_index=True)

    # ---- second pass: remove the systematic tilt ----------------------------
    # Fit pred_raw = a + b*line on *earlier* seasons only, then subtract that
    # line. Doing it here rather than inside the weekly loop matters: the
    # correction has to be fitted on predictions built exactly the way the ones
    # being corrected were built, or it under-corrects and the model keeps
    # leaning on underdogs.
    deb_a, deb_b = 0.0, 1.0
    if DEBIAS:
        bt["pred"] = bt["pred_raw"]
        slate_used = 0
        for season in sorted(bt["season"].dropna().unique()):
            hist = bt[(bt.season < season) & bt.mkt.notna() & bt.pred_raw.notna()]
            a, b = fit_debias(hist["pred_raw"], hist["mkt"]) if len(hist) >= 300 else (0.0, 1.0)
            # One week at a time, because that is the unit the live board sees.
            # Correcting a whole season at once leaves much larger residual
            # edges than the board ever produces, and every threshold and star
            # band is cut from these numbers.
            for wk in sorted(bt.loc[bt.season == season, "week"].dropna().unique()):
                m = (bt.season == season) & (bt.week == wk)
                if not m.any():
                    continue
                vals, src = debias_like_board(bt.loc[m], "pred_raw", "mkt", a, b)
                bt.loc[m, "pred"] = vals
                slate_used += (src == "slate")
        print(f"    debias: {slate_used} weeks corrected on their own slate")
        full = bt.dropna(subset=["mkt", "pred_raw"])
        if len(full) >= 300:
            deb_a, deb_b = fit_debias(full["pred_raw"], full["mkt"])
    else:
        bt["pred"] = bt["pred_raw"]

    s = bt.dropna(subset=["mkt", "pred"]).copy()
    if not len(s):
        return bt, {}

    s["e_model"] = (s.pred - s.margin).abs()
    s["e_mkt"] = (s.mkt - s.margin).abs()
    # Debiasing anchors predictions to the line, which flatters the accuracy
    # figure without the model getting better. Keep the unanchored error too so
    # improvement can be judged on the model's own work.
    if "pred_raw" in s.columns:
        s["e_raw"] = (s.pred_raw - s.margin).abs()
    s["edge"] = s.pred - s.mkt
    s["cover"] = np.where(s.margin > s.mkt, 1.0, np.where(s.margin < s.mkt, 0.0, np.nan))

    # ---- confidence: edge relative to expected error on this game ----
    X, cnames = confidence_features(s, pred_col="pred", alt_pred_col="pred_alt")
    if "_gp_min" in s.columns:
        imm = 6.0 / np.clip(s["_gp_min"].to_numpy(dtype=float), 1.0, None)
        X = np.column_stack([X, imm]) if X.size else imm.reshape(-1, 1)
        cnames = cnames + ["immaturity"]

    conf_w, conf_mae = fit_uncertainty(X, (s.pred - s.margin).to_numpy(dtype=float))
    s["sigma"] = predict_uncertainty(X, conf_w, conf_mae)
    s["confidence"] = confidence_score(s["edge"].to_numpy(dtype=float),
                                       s["sigma"].to_numpy(dtype=float))
    # The qualifying bar has to be known before the star bands, because the bands
    # are cut over qualifying picks only. Cutting over every game instead put 72%
    # of a live board in the top band: qualified picks are already the high-edge
    # quarter, so they all inherited the top of a distribution built from games
    # that would never be shown.
    thr = MIN_EDGE
    if AUTO_THRESHOLD and len(s) > 300:
        thr = float(np.quantile(s["edge"].abs().dropna(), 1 - TARGET_QUALIFIED))
        thr = round(max(thr, 1.0), 1)

    shown = s[s.edge.abs() >= thr]

    # Preseason ratings are compressed harder than in-season ones, so their
    # confidence values sit on a different scale. One set of cutpoints applied to
    # both means a live 5-star pick need not match a backtest 5-star pick. Cut
    # each regime separately and let live picks use the one they belong to.
    cuts_early = tier_cutpoints(shown.loc[shown.week <= 4, "confidence"])
    cuts_late = tier_cutpoints(shown.loc[shown.week >= 5, "confidence"])
    cuts_all = tier_cutpoints(shown["confidence"])
    cuts = {"early": cuts_early or cuts_all, "late": cuts_late or cuts_all,
            "all": cuts_all}
    tier_trend, tier_summary = tier_season_trend(shown, cuts)

    conf_report = ({n: round(float(w), 3) for n, w in zip(cnames, conf_w[:-1])}
                   if conf_w is not None else {})
    if conf_w is not None:
        conf_report["baseline"] = round(float(conf_w[-1]), 2)

    def slice_stats(frame):
        if not len(frame):
            return None
        out = {
            "games": int(len(frame)),
            "model_mae": round(float(frame.e_model.mean()), 2),
            "market_mae": round(float(frame.e_mkt.mean()), 2),
        }
        if "e_raw" in frame.columns and frame.e_raw.notna().any():
            out["model_mae_raw"] = round(float(frame.e_raw.mean()), 2)
        return out

    qual = s[(s.edge.abs() >= thr) & s.cover.notna()]
    win = float(np.where(qual.edge > 0, qual.cover, 1 - qual.cover).mean()) if len(qual) > 20 else None

    thresholds = []
    for cut in (0, 2, 3, 4, 6, 8, 10):
        sub = s[(s.edge.abs() >= cut) & s.cover.notna()]
        if len(sub) < 40:
            continue
        thresholds.append({
            "edge": cut,
            "bets": int(len(sub)),
            "win_pct": round(float(np.where(sub.edge > 0, sub.cover, 1 - sub.cover).mean()) * 100, 1),
        })

    tuning = s[s.season != HOLDOUT] if HOLDOUT else s
    hold = s[s.season == HOLDOUT] if HOLDOUT else pd.DataFrame()

    def qual_pct(frame):
        q = frame[(frame.edge.abs() >= MIN_EDGE) & frame.cover.notna()]
        if len(q) < 20:
            return None, int(len(q))
        return round(float(np.where(q.edge > 0, q.cover, 1 - q.cover).mean()) * 100, 1), int(len(q))

    tune_pct, tune_n = qual_pct(tuning)
    hold_pct, hold_n = qual_pct(hold)

    return bt, {
        "overall": slice_stats(s),
        "early": slice_stats(s[s.week <= 4]),
        "late": slice_stats(s[s.week >= 5]),
        "situational": sit_report,
        "debias": {"intercept": round(deb_a, 3), "slope": round(deb_b, 3),
                   "active": bool(DEBIAS)},
        "bias_check": bias_diagnostics(s, thr),
        "by_season": season_breakdown(s, min_edge=thr),
        "min_edge": thr,
        "tier_cuts": cuts,
        "tier_trend": tier_trend,
        "tier_summary": tier_summary,
        "uncertainty": conf_report,
        "confidence_tiers": confidence_tier_stats(s),
        "confidence_tiers_holdout": (confidence_tier_stats(hold)
                                     if HOLDOUT and len(hold) else []),
        "conf_weights": ([float(x) for x in conf_w] if conf_w is not None else None),
        "conf_names": cnames,
        "conf_mae": round(conf_mae, 2) if conf_mae == conf_mae else None,
        "tiers": tier_stats(s),
        "tiers_holdout": tier_stats(hold) if HOLDOUT and len(hold) else [],
        "tuning": {"seasons": [x for x in TEST_SEASONS if x != HOLDOUT],
                   "win_pct": tune_pct, "bets": tune_n,
                   **(slice_stats(tuning) or {})},
        "holdout": ({"season": HOLDOUT, "win_pct": hold_pct, "bets": hold_n,
                     **(slice_stats(hold) or {})} if HOLDOUT and len(hold) else None),
        "qualified_win_pct": round(win * 100, 1) if win is not None else None,
        "qualified_bets": int(len(qual)),
        "thresholds": thresholds,
        "seasons": TEST_SEASONS,
    }


# ======================================================================
# 3b. totals
# ======================================================================

def run_totals_backtest(d, use_wx):
    """Walk-forward totals backtest.

    Offense/defense points ratings are refit each week from that season's games
    so far, then calibrated onto the totals scale with weather and rest context.
    Weeks 1-4 are skipped: there is no usable preseason prior for totals, and
    pretending otherwise would only add noise.
    """
    preds = []
    for season in TEST_SEASONS:
        for wk in sorted(d.loc[(d.season == season) & (d.week >= 5), "week"].unique()):
            so_far = d[(d.season == season) & (d.week < wk)]
            test = d[(d.season == season) & (d.week == wk)]
            if len(so_far) < 200 or test.empty:
                continue

            off, dfn, hfa_off, base = fit_points_ratings(so_far, ridge=14.0, cap=56.0)
            if off is None:
                continue

            raw_prev = predict_total(so_far, off, dfn, hfa_off, base)
            S_prev, names = totals_matrix(so_far)
            ok = ~np.isnan(raw_prev) & so_far["actual_total"].notna().to_numpy()
            if ok.sum() < 200:
                continue
            A = np.column_stack([raw_prev[ok], S_prev[ok], np.ones(int(ok.sum()))])
            w, *_ = np.linalg.lstsq(
                A, so_far.loc[ok, "actual_total"].to_numpy(dtype=float), rcond=None)

            raw = predict_total(test, off, dfn, hfa_off, base)
            S, _ = totals_matrix(test)
            At = np.nan_to_num(np.column_stack([raw, S, np.ones(len(test))]), nan=0.0)
            out = test.copy()
            out["pred_total_raw"] = At @ w
            preds.append(out)

    if not preds:
        return pd.DataFrame(), {}

    bt = pd.concat(preds, ignore_index=True)

    # Same tilt correction as spreads. Without it the totals model's own scale
    # decides whether the board is all overs or all unders.
    t_deb_a, t_deb_b = 0.0, 1.0
    if DEBIAS:
        bt["pred_total"] = bt["pred_total_raw"]
        slate_used = 0
        for season in sorted(bt["season"].dropna().unique()):
            hist = bt[(bt.season < season) & bt.mkt_total.notna() & bt.pred_total_raw.notna()]
            a, b = (fit_debias(hist["pred_total_raw"], hist["mkt_total"])
                    if len(hist) >= 300 else (0.0, 1.0))
            for wk in sorted(bt.loc[bt.season == season, "week"].dropna().unique()):
                m = (bt.season == season) & (bt.week == wk)
                if not m.any():
                    continue
                vals, src = debias_like_board(bt.loc[m], "pred_total_raw", "mkt_total", a, b)
                bt.loc[m, "pred_total"] = vals
                slate_used += (src == "slate")
        print(f"    totals debias: {slate_used} weeks corrected on their own slate")
        full = bt.dropna(subset=["mkt_total", "pred_total_raw"])
        if len(full) >= 300:
            t_deb_a, t_deb_b = fit_debias(full["pred_total_raw"], full["mkt_total"])
    else:
        bt["pred_total"] = bt["pred_total_raw"]

    s = bt.dropna(subset=["mkt_total", "pred_total", "actual_total"]).copy()
    if not len(s):
        return bt, {}

    s["e_model"] = (s.pred_total - s.actual_total).abs()
    s["e_mkt"] = (s.mkt_total - s.actual_total).abs()
    s["edge"] = s.pred_total - s.mkt_total
    # "cover" means the over hit
    s["cover"] = np.where(s.actual_total > s.mkt_total, 1.0,
                          np.where(s.actual_total < s.mkt_total, 0.0, np.nan))

    t_thr = MIN_EDGE_TOTAL
    if AUTO_THRESHOLD and len(s) > 300:
        t_thr = round(max(float(np.quantile(s["edge"].abs().dropna(),
                                            1 - TARGET_QUALIFIED)), 1.0), 1)

    qual = s[(s.edge.abs() >= t_thr) & s.cover.notna()]
    win = (float(np.where(qual.edge > 0, qual.cover, 1 - qual.cover).mean())
           if len(qual) > 20 else None)

    hold = s[s.season == HOLDOUT] if HOLDOUT else pd.DataFrame()
    hq = (hold[(hold.edge.abs() >= t_thr) & hold.cover.notna()]
          if len(hold) else pd.DataFrame())
    hwin = (float(np.where(hq.edge > 0, hq.cover, 1 - hq.cover).mean())
            if len(hq) > 20 else None)

    return bt, {
        "games": int(len(s)),
        "model_mae": round(float(s.e_model.mean()), 2),
        "market_mae": round(float(s.e_mkt.mean()), 2),
        "qualified_win_pct": round(win * 100, 1) if win is not None else None,
        "qualified_bets": int(len(qual)),
        "holdout": ({"season": HOLDOUT,
                     "win_pct": round(hwin * 100, 1) if hwin is not None else None,
                     "bets": int(len(hq))} if HOLDOUT and len(hq) else None),
        "tiers": tier_stats(s, edge_col="edge", margin_col="actual_total",
                            line_col="mkt_total"),
        "by_season": season_breakdown(s, outcome_col="actual_total",
                                      line_col="mkt_total", min_edge=t_thr),
        "seasons": TEST_SEASONS,
        "min_edge": t_thr,
        "uses_weather": bool(use_wx),
        "debias": {"intercept": round(t_deb_a, 3), "slope": round(t_deb_b, 3),
                   "active": bool(DEBIAS)},
        "bias_check": {
            "pct_over": round(float((s.edge > 0).mean()) * 100, 1),
            "edge_vs_line_corr": (round(float(np.corrcoef(
                s.mkt_total, s.edge)[0, 1]), 3) if len(s) > 100 else None),
            "qualified_share": round(float((s.edge.abs() >= t_thr).mean()) * 100, 1),
        },
    }


def fit_totals_calibration(d, use_wx):
    """Fit current offense/defense ratings plus the totals calibration."""
    off, dfn, hfa_off, base = fit_points_ratings(d, ridge=14.0, cap=56.0)
    if off is None:
        return None, None, 0.0, 0.0, None, []
    raw = predict_total(d, off, dfn, hfa_off, base)
    S, names = totals_matrix(d)
    ok = ~np.isnan(raw) & d["actual_total"].notna().to_numpy()
    if ok.sum() < 200:
        return off, dfn, hfa_off, base, None, names
    A = np.column_stack([raw[ok], S[ok], np.ones(int(ok.sum()))])
    w, *_ = np.linalg.lstsq(A, d.loc[ok, "actual_total"].to_numpy(dtype=float), rcond=None)
    return off, dfn, hfa_off, base, w, names


def find_total_picks(cur, off, dfn, hfa_off, base, cal_w, cal_names, debias=None,
                     min_edge=None):
    """Upcoming games with a posted total, ranked by disagreement."""
    now = datetime.now(timezone.utc)
    up = cur[
        cur.home_pts.isna() & cur.kickoff.notna()
        & (cur.kickoff > now) & (cur.kickoff < now + timedelta(days=LOOKAHEAD_DAYS))
    ].copy()
    up = up[(up.home_team != "NON_FBS") & (up.away_team != "NON_FBS")]
    if not len(up) or off is None or cal_w is None:
        return pd.DataFrame()

    raw = predict_total(up, off, dfn, hfa_off, base)
    S, names = totals_matrix(up)
    if list(names) != list(cal_names) or S.shape[1] != len(cal_names):
        S = np.zeros((len(up), len(cal_names)))
    A = np.nan_to_num(np.column_stack([raw, S, np.ones(len(up))]), nan=0.0)
    if A.shape[1] != len(cal_w):
        return pd.DataFrame()

    up["pred_total_raw"] = A @ cal_w
    up["pred_total"] = up["pred_total_raw"]
    debias_source = "none"
    if DEBIAS:
        # one implementation, shared with the backtest
        up["pred_total"], debias_source = debias_like_board(
            up, "pred_total_raw", "mkt_total",
            float((debias or {}).get("intercept", 0.0)),
            float((debias or {}).get("slope", 1.0)))
    up.attrs["debias_source"] = debias_source
    up["edge"] = up["pred_total"] - up["mkt_total"]
    up = up.dropna(subset=["edge"])
    if not len(up):
        return pd.DataFrame()

    up["side"] = np.where(up.edge > 0, "over", "under")
    up["pick_team"] = np.where(up.edge > 0, "Over", "Under")
    up["pick_spread"] = up["mkt_total"]
    up["qualified"] = up.edge.abs() >= (MIN_EDGE_TOTAL if min_edge is None else min_edge)
    return up.sort_values("edge", key=lambda c: c.abs(), ascending=False).reset_index(drop=True)


# ======================================================================
# edge drivers
# ======================================================================

def _fmt_team(name, n=18):
    return name if len(name) <= n else name[: n - 1] + "\u2026"


def market_ratings(games):
    """Power ratings fitted to the book's closing numbers instead of results.

    This is the one thing that makes a driver worth reading. Comparing our rating
    of a team against the market's implied rating says *where* the disagreement
    sits — which team the book is pricing differently, and by how much. Without
    it every explanation is a restatement of the edge the card already prints.
    """
    sub = games.dropna(subset=["mkt"])
    if len(sub) < 150:
        return {}, 0.0
    r, hfa = fit_ratings(sub, "mkt", ridge=6.0, cap=40.0, min_games=150)
    return (r or {}), hfa


def _disagreement(pick_key, fade_key, pick_name, fade_name, ratings, mkt_r):
    """Per-team gap between our rating and the market's implied one."""
    out = []
    for key, name in ((pick_key, pick_name), (fade_key, fade_name)):
        ours, theirs = ratings.get(key), mkt_r.get(key)
        if ours is None or theirs is None:
            continue
        out.append((name, ours - theirs))
    return out


def spread_drivers(row, ratings, ranks, off, dfn, hfa, games_played, in_season_weight,
                   sit_weights=None, mkt_r=None):
    """Why this pick, in numbers the card doesn't already show.

    The gap is printed on the card, so restating it is not a driver. Each entry
    here has to add something: a schedule mismatch, a unit mismatch, where the
    teams actually rank, or a reason for doubt. Anything that would read the same
    on half the slate is dropped.
    """
    out = []
    # A feature the plausibility gate rejected contributes exactly zero to the
    # number. Naming it as a reason implies it moved the line when it did not.
    sw = sit_weights or {}
    def uses(name):
        v = sw.get(name)
        return isinstance(v, (int, float)) and abs(v) > 1e-9

    home, away = row.get("home_name"), row.get("away_name")
    ht, at = row.get("home_team"), row.get("away_team")
    pick_home = row.get("edge", 0) > 0
    pick_name = home if pick_home else away
    fade_name = away if pick_home else home
    pick_key, fade_key = (ht, at) if pick_home else (at, ht)

    # 1. schedule: only genuine mismatches, not a routine 7-vs-7
    rest = float(row.get("rest_diff") or 0)
    if abs(rest) >= 5 and uses("rest_diff"):
        rested = home if rest > 0 else away
        out.append({"kind": "rest", "weight": 2.0 + abs(rest) / 7.0,
                    "text": f"{_fmt_team(rested)} has {abs(rest):.0f} more days of rest"})

    eb = float(row.get("east_body") or 0)
    tk = float(row.get("travel_diff_k") or 0)
    if eb > 0 and uses("east_body"):
        out.append({"kind": "travel", "weight": 2.0 + eb / 3.0,
                    "text": f"{_fmt_team(away)} crosses {eb:.0f} time zones for an early kick"})
    elif tk >= 1.6 and uses("travel_diff_k"):
        out.append({"kind": "travel", "weight": 1.6 + tk / 3.0,
                    "text": f"{_fmt_team(away)} flies {tk * 1000:,.0f} miles"})

    # 2. unit mismatch: which side of the ball actually creates the edge
    if off and dfn:
        o_pick, d_fade = off.get(pick_key), dfn.get(fade_key)
        d_pick, o_fade = dfn.get(pick_key), off.get(fade_key)
        cands = []
        # our strong offense against their weak defense
        if o_pick is not None and d_fade is not None and o_pick >= 3 and d_fade <= -4:
            cands.append((o_pick - d_fade, "offense",
                          f"{_fmt_team(pick_name)} offense (+{o_pick:.0f}) meets a defense "
                          f"{abs(d_fade):.0f} pts below average"))
        # our strong defense against their weak offense
        if d_pick is not None and o_fade is not None and d_pick >= 3 and o_fade <= -4:
            cands.append((d_pick - o_fade, "defense",
                          f"{_fmt_team(pick_name)} defense (+{d_pick:.0f}) meets an offense "
                          f"{abs(o_fade):.0f} pts below average"))
        if cands:
            cands.sort(key=lambda c: -c[0])
            _score, kind, text = cands[0]
            out.append({"kind": kind, "weight": 1.9, "text": text})

    # 3. where the disagreement actually sits — the only genuinely new thing
    for name, gap in _disagreement(pick_key, fade_key, pick_name, fade_name,
                                   ratings, mkt_r or {}):
        if abs(gap) < 3.0:
            continue
        verb = "higher" if gap > 0 else "lower"
        out.append({"kind": "rating", "weight": 2.2 + abs(gap) / 6.0,
                    "text": f"We rate {_fmt_team(name)} {abs(gap):.1f} pts {verb} "
                            f"than the book's number implies"})

    # fall back to standings only when the disagreement is too small to name
    if not any(x["kind"] == "rating" for x in out):
        rp, rf = ranks.get(pick_key), ranks.get(fade_key)
        if rp and rf and abs(rp - rf) >= 25:
            out.append({"kind": "rating", "weight": 1.2,
                        "text": f"Model ranks {_fmt_team(pick_name)} #{rp}, "
                                f"{_fmt_team(fade_name)} #{rf}"})

    # 4. reasons to doubt it
    gp = min(games_played.get(ht, 0), games_played.get(at, 0))
    if gp == 0:
        out.append({"kind": "caution", "weight": 3.0,
                    "text": "Preseason ratings only, no games played"})
    elif gp < 4:
        out.append({"kind": "caution", "weight": 3.0,
                    "text": f"Thin evidence: one side has played {gp} game{'s' if gp != 1 else ''}"})


    if row.get("rivalry") and uses("rivalry_compress"):
        out.append({"kind": "rivalry", "weight": 1.5,
                    "text": "Rivalry game, usually closer than ratings say"})

    mkt = float(row.get("mkt") or 0)
    if abs(mkt) >= 17 and ((mkt > 0) != pick_home):
        out.append({"kind": "contrarian", "weight": 1.7,
                    "text": f"Backing a {abs(mkt):.0f}-point underdog"})

    # Never return nothing. A quarter of the board had no explanation at all,
    # which reads as a broken button rather than a quiet game.
    if not out:
        gap = abs(float(row.get("edge") or 0))
        sig = float(row.get("sigma") or 0)
        if sig > 0:
            out.append({"kind": "rating", "weight": 0.5,
                        "text": f"{gap:.1f} pts of disagreement against a typical "
                                f"miss of {sig:.0f} on games like this"})
        else:
            out.append({"kind": "rating", "weight": 0.5,
                        "text": "No single factor drives this one; the ratings "
                                "and the number simply differ"})

    out.sort(key=lambda d: -d["weight"])
    return [{"kind": d["kind"], "text": d["text"]} for d in out[:2]]


def total_drivers(row, off, dfn, use_wx=True, mkt_total_r=None):
    """Why this total, in concrete numbers."""
    out = []
    # name the side of the ball the number turns on, rather than reciting both
    ht, at = row.get("home_team"), row.get("away_team")
    o_vals = [off.get(ht), off.get(at)] if off else []
    d_vals = [dfn.get(ht), dfn.get(at)] if dfn else []
    o_sum = sum(v for v in o_vals if v is not None) if all(v is not None for v in o_vals) and o_vals else None
    d_sum = sum(v for v in d_vals if v is not None) if all(v is not None for v in d_vals) and d_vals else None
    if o_sum is not None and d_sum is not None and abs(o_sum) + abs(d_sum) > 6:
        if abs(o_sum) >= abs(d_sum):
            word = "above" if o_sum > 0 else "below"
            out.append({"kind": "offense", "weight": 2.6,
                        "text": f"Scoring drives this one: the two offences rate "
                                f"{abs(o_sum):.0f} pts {word} average combined"})
        else:
            word = "above" if d_sum > 0 else "below"
            out.append({"kind": "defense", "weight": 2.6,
                        "text": f"Defence drives this one: the two rate "
                                f"{abs(d_sum):.0f} pts {word} average combined"})
    home, away = row.get("home_name"), row.get("away_name")
    over = row.get("edge", 0) > 0

    wind = row.get("wind") if use_wx else None
    if row.get("indoor"):
        out.append({"kind": "weather", "text": "Indoor, weather removed", "weight": 0.4})
    elif wind is not None and wind == wind and float(wind) >= 18:
        out.append({"kind": "weather",
                    "text": f"{float(wind):.0f} mph wind, the biggest drag on scoring",
                    "weight": float(wind) / 22.0})

    temp = row.get("temp")
    if temp is not None and temp == temp and float(temp) <= 28:
        out.append({"kind": "weather", "text": f"{float(temp):.0f}\u00b0F at kickoff",
                    "weight": 0.6})

    rest = float(row.get("rest_diff") or 0)
    if abs(rest) >= 7:
        out.append({"kind": "rest", "text": f"{abs(rest):.0f}-day rest gap", "weight": 0.5})

    out.sort(key=lambda d: -d["weight"])
    return [{"kind": d["kind"], "text": d["text"]} for d in out[:2]]


# ======================================================================
# 3. picks
# ======================================================================

def find_picks(cur, ratings, hfa, sit_weights=None, prior=None,
               conf_w=None, conf_names=None, conf_mae=None, games_played=None,
               debias=None, min_edge=None):
    """Upcoming FBS-vs-FBS games with a posted line, ranked by disagreement."""
    now = datetime.now(timezone.utc)
    up = cur[
        cur.home_pts.isna()
        & cur.kickoff.notna()
        & (cur.kickoff > now)
        & (cur.kickoff < now + timedelta(days=LOOKAHEAD_DAYS))
    ].copy()

    # Non-FBS opponents are excluded: the pooled rating is too crude to bet.
    up = up[(up.home_team != "NON_FBS") & (up.away_team != "NON_FBS")]
    if not len(up):
        return pd.DataFrame()

    rd = rating_diff(up, ratings, hfa)
    scale = 1.0
    if sit_weights:
        S, names = situational_matrix(up, rd)
        adj = np.zeros(len(up))
        for j, nm in enumerate(names):
            v = sit_weights.get(nm, 0.0)
            adj = adj + S[:, j] * (v if isinstance(v, (int, float)) else 0.0)
        scale = sit_weights.get("rating_scale", 1.0) or 1.0
        up["pred"] = rd * scale + adj
    else:
        up["pred"] = rd

    up["pred_raw"] = up["pred"]
    debias_source = "none"
    if DEBIAS:
        # Parameters fitted on history under-correct a board, because the board
        # is one slate of games. Shared with the backtest so both produce the
        # same distribution of edges — see debias_like_board in model.py.
        up["pred"], debias_source = debias_like_board(
            up, "pred_raw", "mkt",
            float((debias or {}).get("intercept", 0.0)),
            float((debias or {}).get("slope", 1.0)))
    up.attrs["debias_source"] = debias_source
    up["edge"] = up["pred"] - up["mkt"]
    up = up.dropna(subset=["edge"])
    if not len(up):
        return pd.DataFrame()

    up["side"] = np.where(up.edge > 0, "home", "away")
    up["pick_team"] = np.where(up.edge > 0, up.home_name, up.away_name)
    # Spread as it would be quoted for the side we like.
    up["pick_spread"] = np.where(up.edge > 0, -up["mkt"], up["mkt"])
    up["qualified"] = up.edge.abs() >= (MIN_EDGE if min_edge is None else min_edge)

    # ---- confidence ----
    up["pred_alt"] = (rating_diff(up, prior, hfa) * scale
                      if prior else up["pred"])
    X, names = confidence_features(up, pred_col="pred", alt_pred_col="pred_alt")
    if games_played is not None and conf_names and "immaturity" in conf_names:
        imm = np.array([6.0 / max(min(games_played.get(h, 0), games_played.get(a, 0)), 1.0)
                        for h, a in zip(up["home_team"], up["away_team"])])
        X = np.column_stack([X, imm]) if X.size else imm.reshape(-1, 1)
        names = names + ["immaturity"]

    if conf_w is not None and list(names) == list(conf_names):
        up["sigma"] = predict_uncertainty(X, np.asarray(conf_w), conf_mae)
    else:
        up["sigma"] = conf_mae if conf_mae else 13.0
    up["confidence"] = confidence_score(up["edge"].to_numpy(dtype=float),
                                        up["sigma"].to_numpy(dtype=float))

    up = up.sort_values("confidence", ascending=False).reset_index(drop=True)
    up["top_pick"] = False
    qual_idx = up.index[up.qualified][:TOP_PICK_COUNT]
    up.loc[qual_idx, "top_pick"] = True
    return up


# ======================================================================
# 4. bet log
# ======================================================================

def update_bet_log(picks, finished, total_picks=None, bt_cuts=None):
    """Append newly qualified wagers, then grade any that have completed.

    Handles both markets. `market` is 'spread' or 'total'; rows written before
    that column existed are treated as spreads.
    """
    cols = ["game_id", "market", "season", "week", "logged_at", "away_name",
            "home_name", "pick_team", "side", "pred", "mkt_at_log",
            "pick_spread", "edge", "confidence", "top_pick"]

    log = pd.read_csv(BET_LOG) if BET_LOG.exists() else pd.DataFrame(columns=cols)
    if "market" not in log.columns:
        log["market"] = "spread"
    log["market"] = log["market"].fillna("spread")

    def add(frame, market, pred_col, line_col):
        nonlocal log
        if frame is None or not len(frame):
            return
        # Only the current week is logged. Lines two weeks out are softer, which
        # would flatter closing-line value, and the record is meant to describe
        # wagers that would actually have been placed.
        keep = frame.qualified
        if "current_week" in frame.columns:
            keep = keep & frame["current_week"]
        new = frame[keep].copy()
        if not len(new):
            return
        new["market"] = market
        new["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        new = new.rename(columns={line_col: "mkt_at_log", pred_col: "pred"})
        new = new[[c for c in cols if c in new.columns]]
        log = pd.concat([log, new], ignore_index=True)

    add(picks, "spread", "pred", "mkt")
    add(total_picks, "total", "pred_total", "mkt_total")

    if not len(log):
        return log, {}

    # One row per game: the first time it qualified is the number we'd have taken.
    log["game_id"] = pd.to_numeric(log["game_id"], errors="coerce")
    log = (log.dropna(subset=["game_id"])
              .sort_values("logged_at")
              .drop_duplicates(["game_id", "market"], keep="first")
              .reset_index(drop=True))
    BET_LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(BET_LOG, index=False)

    # ---- grade ----
    res = finished[["game_id", "margin", "mkt", "actual_total", "mkt_total"]].rename(
        columns={"mkt": "mkt_close_spread", "mkt_total": "mkt_close_total"})
    g = log.merge(res, on="game_id", how="left")

    is_total = g["market"] == "total"
    # the realized number each wager is graded against
    g["outcome"] = np.where(is_total, g["actual_total"], g["margin"])
    g["mkt_close"] = np.where(is_total, g["mkt_close_total"], g["mkt_close_spread"])

    graded = g.dropna(subset=["outcome"]).copy()
    if not len(graded):
        return log, {"pending": int(len(g)), "settled": 0}

    # a wager wins if the outcome landed on the side taken
    over_or_home = graded["side"].isin(["home", "over"])
    graded["result"] = np.select(
        [graded.outcome == graded.mkt_at_log,
         over_or_home & (graded.outcome > graded.mkt_at_log),
         (~over_or_home) & (graded.outcome < graded.mkt_at_log)],
        ["push", "win", "win"], default="loss")

    profit = american_to_profit(PRICE)
    graded["units"] = graded.result.map({"win": profit, "loss": -1.0, "push": 0.0})
    graded["clv"] = np.where(over_or_home,
                             graded.mkt_close - graded.mkt_at_log,
                             graded.mkt_at_log - graded.mkt_close)

    decided = graded[graded.result != "push"]
    clv = graded.dropna(subset=["clv"])

    summary = {
        "settled": int(len(graded)),
        "pending": int(len(g) - len(graded)),
        "wins": int((graded.result == "win").sum()),
        "losses": int((graded.result == "loss").sum()),
        "pushes": int((graded.result == "push").sum()),
        "win_pct": round(float((decided.result == "win").mean()) * 100, 1) if len(decided) else None,
        "units": round(float(graded.units.sum()), 2),
        "roi_pct": round(float(graded.units.sum() / len(graded)) * 100, 1) if len(graded) else None,
        "avg_clv": round(float(clv.clv.mean()), 2) if len(clv) >= 5 else None,
        "beat_close_pct": round(float((clv.clv > 0).mean()) * 100, 1) if len(clv) >= 5 else None,
        "breakeven_pct": round(100.0 / (1.0 + profit), 1),
    }

    graded = graded.sort_values("logged_at", ascending=False)

    def market_summary(frame):
        if not len(frame):
            return None
        dec = frame[frame.result != "push"]
        return {
            "settled": int(len(frame)),
            "wins": int((frame.result == "win").sum()),
            "losses": int((frame.result == "loss").sum()),
            "pushes": int((frame.result == "push").sum()),
            "win_pct": round(float((dec.result == "win").mean()) * 100, 1) if len(dec) else None,
            "units": round(float(frame.units.sum()), 2),
            "roi_pct": round(float(frame.units.sum() / len(frame)) * 100, 1),
        }

    # live tier breakdowns, using the line recorded at log time
    def tiers_for(frame):
        if not len(frame):
            return []
        # drop the originals first: renaming onto existing names would create
        # duplicate columns and break the arithmetic in tier_stats
        lt = (frame.drop(columns=["margin", "mkt"], errors="ignore")
                   .rename(columns={"mkt_at_log": "mkt", "outcome": "margin"}))
        return tier_stats(lt, edge_col="edge", margin_col="margin", line_col="mkt")

    sp = graded[graded.market != "total"]
    to = graded[graded.market == "total"]
    cur_season = graded[graded.get("season", pd.Series(dtype=float)) == CURRENT] \
        if "season" in graded.columns else graded.iloc[0:0]
    top = graded[graded.get("top_pick", pd.Series(False, index=graded.index)) == True]

    def clv_stats(frame):
        """Closing line value only. This is the number that moves first when a
        model is genuinely beating the number — win rate needs hundreds of bets,
        CLV shows up in dozens."""
        c = frame.dropna(subset=["clv"]) if "clv" in frame.columns else frame.iloc[0:0]
        if len(c) < 3:
            return None
        return {
            "n": int(len(c)),
            "avg": round(float(c.clv.mean()), 2),
            "beat_pct": round(float((c.clv > 0).mean()) * 100, 1),
            "median": round(float(c.clv.median()), 2),
        }

    clv_by_market = {"spread": clv_stats(sp), "total": clv_stats(to),
                     "all": clv_stats(graded)}
    clv_by_tier = {}
    if "confidence" in graded.columns and bt_cuts:
        tier_of = [assign_tier(c, regime_cuts(bt_cuts, week=w))
                   for c, w in zip(graded["confidence"],
                                   graded.get("week", pd.Series([9] * len(graded),
                                                                index=graded.index)))]
        graded = graded.assign(_tier=tier_of)
        for name in TIER_NAMES:
            st = clv_stats(graded[graded._tier == name])
            if st:
                clv_by_tier[name] = st

    return log, {
        "summary": summary,
        "rows": graded,
        "tiers": tiers_for(sp),
        "tiers_total": tiers_for(to),
        "by_market": {"spread": market_summary(sp), "total": market_summary(to),
                      "top": market_summary(top)},
        "clv_by_market": clv_by_market,
        "clv_by_tier": clv_by_tier,
        "live_season": ({"season": CURRENT, **market_summary(cur_season)}
                        if len(cur_season) else None),
    }


# ======================================================================
# main
# ======================================================================

def main():
    data = load_everything()

    d = build_games(data["games"], data["fbs"])
    d, sign = attach_lines(d, data["lines"])
    d, use_epa = attach_epa(d, data["adv"])

    # turnover adjustment: strip the luck portion of turnover margin
    to_df = parse_team_stats(data["team_stats"])
    d, use_to = attach_turnovers(d, to_df)
    d, to_points = add_adjusted_margin(d, shrink=TO_SHRINK)

    # situational context: travel, body clock, rest, rivalry
    venues = parse_venues(data["venues"])
    home_venue = infer_home_venues(d)
    d = add_situational(d, venues, home_venue)

    # weather from Open-Meteo: one archive call per outdoor venue covers every
    # season at once, so this costs ~130 calls once rather than one per game
    wx_end = datetime.now(timezone.utc).date().isoformat()
    # rank venues by how many games they actually host, so the backfill spends
    # its calls where they buy the most coverage
    wx_start = CONFIG.get("weather_start", "2018-08-01")
    used = d.loc[d["season"] >= int(wx_start[:4]), "venue_id"].dropna()
    venue_order = [int(v) for v in used.value_counts().index]
    print(f"Venues hosting games since {wx_start[:4]}: {len(venue_order)} "
          f"(of {len(venues)} known)")

    wx_daily = fetch_venue_weather(
        venues, wx_start, wx_end,
        budget=int(CONFIG.get("weather_venue_budget", 25)),
        seconds_budget=int(CONFIG.get("weather_seconds_budget", 300)),
        venue_order=venue_order)
    d, use_wx = attach_venue_weather(d, wx_daily, venues)
    n_dome = sum(1 for v in venues.values() if v.get("dome"))
    print(f"Weather: {int(d['wind'].notna().sum()):,} games covered "
          f"| {n_dome} domes | active {use_wx}")

    d = d.sort_values(["season", "week"]).reset_index(drop=True)
    print(f"History: {len(d):,} games | lines {int(d.spread.notna().sum()):,} "
          f"| EPA {use_epa} | turnovers {use_to} | venues {len(venues)}")
    print(f"Rivalry games flagged: {int(d['rivalry'].sum())} | "
          f"mean away travel: {d['travel_diff_k'].mean()*1000:.0f} mi | "
          f"body-clock spots: {int((d['east_body'] > 0).sum())}")
    if to_points is not None:
        print(f"Turnover value: {to_points:.2f} pts each, removing {TO_SHRINK:.0%} "
              f"({TO_SHRINK * to_points:.2f} pts per net turnover)")

    season_r, season_hfa = season_ratings(d, HIST_YEARS, use_epa, target=RATING_TARGET)
    talent = talent_composite(data["recruit"], HIST_YEARS + [CURRENT], CONFIG["recruit_window"])
    ret, ret_field = returning_production(data["returning"])
    sp = data["sp"]
    prior_model = PriorModel(season_r, ret, talent, sp=sp)
    print(f"Prior model: {prior_model.n_train} team-seasons, R2 {prior_model.r2:.3f}, "
          f"features {prior_model.feats}")

    hfa = float(np.mean(list(season_hfa.values()))) if season_hfa else 2.6
    prior = prior_model.preseason(CURRENT)

    # ---- current season ----
    cur = pd.DataFrame()
    ratings_now = prior
    played = 0
    in_season_weight = 0.0

    if len(data["cur_games"]):
        cur = build_games(data["cur_games"], data["fbs"], keep_unplayed=True)
        cur, _ = attach_lines(cur, data["cur_lines"], sign=sign)
        # only worth a call once games have actually been played
        has_results = bool(cur["home_pts"].notna().sum())
        set_live_week(cur)
        cur_to = fetch_team_stats(CURRENT) if has_results else pd.DataFrame()
        cur, _ = attach_turnovers(cur, parse_team_stats(cur_to))
        cur, _ = add_adjusted_margin(cur, shrink=TO_SHRINK)
        cur = add_situational(cur, venues, home_venue)
        up_ids = cur.loc[cur.home_pts.isna(), "venue_id"].dropna().unique()
        fc = fetch_venue_forecast(
            venues, up_ids, days=LOOKAHEAD_DAYS + 2,
            seconds_budget=int(CONFIG.get("forecast_seconds_budget", 90)))
        cur_wx = pd.concat([wx_daily, fc], ignore_index=True) if len(fc) else wx_daily
        cur, _ = attach_venue_weather(cur, cur_wx, venues)
        done = cur[cur.home_pts.notna()]
        played = int(len(done))
        r_in = None
        if played >= 150:
            r_m, _ = fit_ratings(done, RATING_TARGET, ridge=14.0, cap=35.0)
            r_in = blend_ratings(r_m, None)
        gp = games_played_counts(done)
        ratings_now = blend_prior(prior, r_in, gp, BLEND_K)
        if prior:
            # Only teams that have played. The prior carries every team in the
            # rating universe — roughly 250, including the pooled non-FBS side —
            # and the ~115 that never take the field contributed a weight of
            # zero, holding this average near 0.3 deep into November and keeping
            # the model on early-season star bands all year.
            ws = [gp[t] / (gp[t] + BLEND_K) for t in prior
                  if t != "NON_FBS" and gp.get(t, 0) > 0]
            in_season_weight = float(np.mean(ws)) if ws else 0.0

    # ---- backtests ----
    bt, bt_summary = run_backtest(d, prior_model, season_hfa, use_epa)
    tbt, t_summary = run_totals_backtest(d, use_wx)
    if t_summary:
        print(f"\nTotals: model MAE {t_summary['model_mae']} vs market "
              f"{t_summary['market_mae']} on {t_summary['games']:,} games "
              f"| weather {use_wx}")

    # ---- picks + log ----
    gp_now = games_played_counts(cur[cur.home_pts.notna()]) if len(cur) else {}
    picks = (find_picks(cur, ratings_now, hfa, bt_summary.get("situational"),
                       prior=prior,
                       conf_w=bt_summary.get("conf_weights"),
                       conf_names=bt_summary.get("conf_names"),
                       conf_mae=bt_summary.get("conf_mae"),
                       games_played=gp_now,
                       debias=bt_summary.get("debias"),
                       min_edge=bt_summary.get("min_edge"))
             if len(cur) else pd.DataFrame())

    # ---- totals picks ----
    t_off, t_dfn, t_hfa, t_base, t_w, t_names = fit_totals_calibration(d, use_wx)
    total_picks = (find_total_picks(cur, t_off, t_dfn, t_hfa, t_base, t_w, t_names,
                                    debias=t_summary.get("debias"),
                                    min_edge=t_summary.get("min_edge"))
                   if len(cur) else pd.DataFrame())
    if len(total_picks):
        t_mae = t_summary.get("model_mae") or 13.0
        total_picks["sigma"] = t_mae
        total_picks["confidence"] = confidence_score(
            total_picks["edge"].to_numpy(dtype=float), np.full(len(total_picks), t_mae))
        total_picks = total_picks.sort_values("confidence", ascending=False).reset_index(drop=True)
        total_picks["top_pick"] = False
        tq = total_picks.index[total_picks.qualified][:TOP_PICK_COUNT]
        total_picks.loc[tq, "top_pick"] = True

    # Stars rank the picks that cleared the bar, per market, so spreads are not
    # judged against the much larger edges totals produce. Qualification stays
    # absolute; only the ranking is relative. See relative_tiers in model.py.
    _abs_cuts = regime_cuts(bt_summary.get("tier_cuts"),
                            in_season_weight=in_season_weight)
    for _f in (picks, total_picks):
        if len(_f) and "confidence" in _f.columns:
            cw = _f["current_week"] if "current_week" in _f.columns else True
            rel, absol = relative_tiers(_f["confidence"],
                                        _f["qualified"] & cw,
                                        _abs_cuts)
            _f["tier_rel"] = rel
            _f["tier_abs"] = absol

    # The board's current week is the earliest week still unplayed across both
    # markets, so spreads and totals agree on what "this week" means.
    _wks = [int(f["week"].min()) for f in (picks, total_picks)
            if len(f) and "week" in f.columns and f["week"].notna().any()]
    board_week = min(_wks) if _wks else None
    for _f in (picks, total_picks):
        if len(_f) and "week" in _f.columns:
            _f["current_week"] = ((_f["week"] == board_week)
                                  if board_week is not None else True)

    fin_cols = ["game_id", "margin", "mkt", "actual_total", "mkt_total"]
    finished = d[[c for c in fin_cols if c in d.columns]].copy()
    if len(cur):
        cf = cur[cur.home_pts.notna()][[c for c in fin_cols if c in cur.columns]]
        finished = pd.concat([finished, cf], ignore_index=True).drop_duplicates(
            "game_id", keep="last")
    for c in fin_cols:
        if c not in finished.columns:
            finished[c] = np.nan

    log, log_out = update_bet_log(picks, finished, total_picks,
                                  bt_cuts=bt_summary.get("tier_cuts"))

    # Ratings fitted to the book's own numbers, so a driver can say which team
    # the market is pricing differently instead of restating the edge.
    mkt_r, _mkt_hfa = market_ratings(d[d.season >= CURRENT - 1])
    if mkt_r:
        print(f"  market-implied ratings: {len(mkt_r)} teams")

    # ---- ratings table ----
    rank_map = {}
    if ratings_now:
        for i, (tm, _v) in enumerate(sorted(
                ((k, v) for k, v in ratings_now.items() if k != "NON_FBS"),
                key=lambda kv: -kv[1]), start=1):
            rank_map[tm] = i

    ratings_rows = []
    if ratings_now:
        srt = sorted(((t, v) for t, v in ratings_now.items() if t != "NON_FBS"),
                     key=lambda kv: -kv[1])
        ratings_rows = [{"rank": i, "team": t, "rating": round(v, 2)}
                        for i, (t, v) in enumerate(srt, start=1)]

    def total_pick_rows(frame):
        rows = []
        for _, r in frame.iterrows():
            rows.append({
                "game_id": int(r.game_id),
                "drivers": total_drivers(r, t_off or {}, t_dfn or {}, use_wx),
                "week": int(r.week) if pd.notna(r.week) else None,
                "kickoff": r.kickoff.isoformat() if pd.notna(r.kickoff) else None,
                "away": r.away_name, "home": r.home_name,
                "pick": r.pick_team,
                "line": round(float(r.mkt_total), 1),
                "model_total": round(float(r.pred_total), 1),
                "edge": round(float(r.edge), 1),
                "wind": (round(float(r.wind), 1)
                         if "wind" in frame.columns and pd.notna(r.get("wind")) else None),
                "indoor": bool(r.get("indoor", 0)),
                "confidence": (round(float(r.confidence), 2)
                               if "confidence" in frame.columns and pd.notna(r.get("confidence"))
                               else None),
                "top_pick": bool(r.get("top_pick", False)),
                "tier": r.get("tier_rel"),
                "tier_absolute": r.get("tier_abs"),
                "qualified": bool(r.qualified),
                "current_week": bool(r.get("current_week", True)),
            })
        return rows

    def totals_rating_rows(off, dfn):
        if not off:
            return []
        rows = []
        for t_ in sorted(off, key=lambda k: -(off.get(k, 0) - dfn.get(k, 0))):
            if t_ == "NON_FBS":
                continue
            rows.append({"team": t_, "offense": round(off.get(t_, 0.0), 1),
                         "defense": round(dfn.get(t_, 0.0), 1),
                         "pace": round(off.get(t_, 0.0) - dfn.get(t_, 0.0), 1)})
        return rows

    def pick_rows(frame):
        rows = []
        for _, r in frame.iterrows():
            rows.append({
                "game_id": int(r.game_id),
                "drivers": spread_drivers(r, ratings_now or {}, rank_map,
                                          t_off or {}, t_dfn or {}, hfa, gp_now,
                                          in_season_weight,
                                          bt_summary.get("situational"), mkt_r=mkt_r),
                "home_rating": (round(float((ratings_now or {}).get(r.home_team)), 1)
                                if (ratings_now or {}).get(r.home_team) is not None else None),
                "away_rating": (round(float((ratings_now or {}).get(r.away_team)), 1)
                                if (ratings_now or {}).get(r.away_team) is not None else None),
                "week": int(r.week) if pd.notna(r.week) else None,
                "kickoff": r.kickoff.isoformat() if pd.notna(r.kickoff) else None,
                "away": r.away_name, "home": r.home_name,
                "pick": r.pick_team,
                "pick_spread": round(float(r.pick_spread), 1),
                "model_margin": round(float(r.pred), 1),
                "market_margin": round(float(r.mkt), 1),
                "edge": round(float(r.edge), 1),
                "confidence": (round(float(r.confidence), 2)
                               if "confidence" in frame.columns and pd.notna(r.get("confidence"))
                               else None),
                "sigma": (round(float(r.sigma), 1)
                          if "sigma" in frame.columns and pd.notna(r.get("sigma")) else None),
                "top_pick": bool(r.get("top_pick", False)),
                "tier": r.get("tier_rel"),
                "tier_absolute": r.get("tier_abs"),
                "qualified": bool(r.qualified),
                "current_week": bool(r.get("current_week", True)),
            })
        return rows

    equity = []
    if "rows" in log_out and len(log_out["rows"]):
        eq = log_out["rows"].sort_values("logged_at")
        run_units = 0.0
        for i, (_, r) in enumerate(eq.iterrows(), start=1):
            run_units += float(r.units)
            equity.append({"n": i, "units": round(run_units, 2),
                           "market": ("total" if r.market == "total" else "spread")})

    history_rows = []
    if "rows" in log_out:
        for _, r in log_out["rows"].iterrows():
            history_rows.append({
                "week": int(r.week) if pd.notna(r.week) else None,
                "away": r.away_name, "home": r.home_name,
                "pick": r.pick_team,
                "pick_spread": round(float(r.pick_spread), 1) if pd.notna(r.pick_spread) else None,
                "edge": round(float(r.edge), 1) if pd.notna(r.edge) else None,
                "margin": int(r.margin) if pd.notna(r.margin) else None,
                "result": r.result,
                "units": round(float(r.units), 2),
                "clv": round(float(r.clv), 2) if pd.notna(r.clv) else None,
            })

    all_meta = parse_teams(data.get("teams_meta", pd.DataFrame()))
    on_board = set()
    for frame in (picks, total_picks):
        if len(frame):
            on_board |= set(frame["home_name"]) | set(frame["away_name"])
    team_meta = {k: v for k, v in all_meta.items() if k in on_board} or all_meta

    def board_balance(frame, kind):
        """How lopsided is this week's board? A healthy slate is near 50/50."""
        if not len(frame):
            return None
        e = frame["edge"].dropna()
        if not len(e):
            return None
        if kind == "spread":
            side = ((frame["mkt"] > 0) != (frame["edge"] > 0)).mean()
            label = "underdog"
        else:
            side = (frame["edge"] > 0).mean()
            label = "over"
        out = {"n": int(len(e)), "pct": round(float(side) * 100, 1), "side": label}
        if len(e) > 10 and frame["edge"].std() > 0:
            line = frame["mkt"] if kind == "spread" else frame["mkt_total"]
            if line.std() > 0:
                out["corr"] = round(float(np.corrcoef(line, frame["edge"])[0, 1]), 3)
        return out

    balance = {"spread": board_balance(picks, "spread"),
               "total": board_balance(total_picks, "total")}
    for key, frame in (("spread", picks), ("total", total_picks)):
        if balance.get(key) and len(frame):
            balance[key]["debias"] = frame.attrs.get("debias_source", "none")

    qb_value = parse_qb_value(data.get("qb", pd.DataFrame()))
    if qb_value and on_board:
        qb_value = {k: v for k, v in qb_value.items() if k in on_board} or qb_value

    # A relative top band on a quiet week is still the best of a weak board.
    # Say so, rather than letting five stars imply more than it does.
    def week_strength(frame, label):
        if not len(frame) or "tier_rel" not in frame.columns:
            return None
        top = frame[(frame.tier_rel == "A")]
        if not len(top):
            return None
        weak = int((top["tier_abs"].isin(["D", "E"]) | top["tier_abs"].isna()).sum())
        return {"market": label, "top_n": int(len(top)), "weak": weak,
                "top_edge": round(float(frame.loc[frame.qualified, "edge"].abs().max()), 1)
                             if frame.qualified.any() else None}

    strength = [s for s in (week_strength(picks, "spreads"),
                            week_strength(total_picks, "totals")) if s]

    notes = []
    for s in strength:
        if s["weak"] and s["weak"] >= s["top_n"] * 0.5:
            notes.append(
                f"Thin week for {s['market']}: the top-rated picks are small by "
                f"historical standards. Stars rank this week's board, they do not "
                f"promise a strong edge.")

    for kind, b in balance.items():
        if b and b["n"] >= 12 and (b["pct"] >= 75 or b["pct"] <= 25):
            notes.append(
                f"{b['pct']}% of {kind} picks are on the {b['side']}. A balanced board "
                f"sits near 50%, so treat this week's lean as a quirk of the model "
                f"rather than a read on the slate.")
    if in_season_weight < 0.30:
        notes.append(
            "Early season: ratings are mostly last year's, adjusted for returning "
            "production and recruiting. The model does not know about incoming "
            "transfers, so treat these numbers as weak."
        )
    if not use_epa:
        notes.append("EPA data unavailable this run; ratings are margin-only.")
    if not use_to:
        notes.append("Turnover data unavailable this run; ratings use raw scoring margin.")
    if not sp:
        notes.append("SP+ ratings unavailable this run; the prior does not include them.")
    if not use_wx:
        notes.append("Weather data unavailable, so the totals model runs without wind. "
                     "Wind is the largest external factor on scoring, so totals numbers "
                     "are weaker than they would otherwise be.")
    if prior_model.weights is None:
        notes.append("Preseason prior could not be fit. Check the data pull logs.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": CURRENT,
        "min_edge": bt_summary.get("min_edge", MIN_EDGE),
        "board_week": board_week,
        "assumed_price": PRICE,
        "state": {
            "games_played": played,
            "in_season_weight": round(in_season_weight, 3),
            "home_field": round(hfa, 2),
            "uses_epa": bool(use_epa),
            "uses_turnovers": bool(use_to),
            "uses_weather": bool(use_wx),
            "weather_games": int(d["wind"].notna().sum()) if "wind" in d.columns else 0,
            "domes": sum(1 for v in venues.values() if v.get("dome")),
            "uses_sp": bool(sp),
            "turnover_points": round(to_points, 2) if to_points is not None else None,
            "turnover_shrink": TO_SHRINK,
            "returning_field": ret_field,
        },
        "situational": bt_summary.get("situational", {}),
        "debias": bt_summary.get("debias", {}),
        "bias_check": bt_summary.get("bias_check", {}),
        "totals_bias_check": t_summary.get("bias_check", {}),
        "board_balance": balance,
        "week_strength": strength,
        "uncertainty": bt_summary.get("uncertainty", {}),
        "confidence_tiers": bt_summary.get("confidence_tiers", []),
        "tier_trend": bt_summary.get("tier_trend", {}),
        "tier_summary": bt_summary.get("tier_summary", {}),
        "tier_cuts": bt_summary.get("tier_cuts"),
        "tier_regime": ("early" if in_season_weight < 0.35 else "late"),
        "confidence_tiers_holdout": bt_summary.get("confidence_tiers_holdout", []),
        "top_pick_count": TOP_PICK_COUNT,
        "prior_model": {
            "r2": round(prior_model.r2, 3) if prior_model.r2 == prior_model.r2 else None,
            "team_seasons": prior_model.n_train,
            "coefficients": {k: round(v, 3) for k, v in prior_model.coefficients().items()},
        },
        "backtest": bt_summary,
        "teams": team_meta,
        "conf_label": "Conf",
        "conf_groups": [{"value": "p4", "label": "Power 4",
                         "members": ["SEC", "Big Ten", "ACC", "Big 12"]}],
        "qb_value": qb_value,
        "qb_season": data.get("qb_season"),
        "picks": pick_rows(picks) if len(picks) else [],
        "total_picks": total_pick_rows(total_picks) if len(total_picks) else [],
        "totals": t_summary,
        "gap_trend": combined_gap_trend(bt_summary.get("by_season", []),
                                        t_summary.get("by_season", [])),
        "totals_ratings": totals_rating_rows(t_off, t_dfn),
        "history": history_rows,
        "record": log_out.get("summary", {}),
        "equity": equity,
        "record_tiers": log_out.get("tiers", []),
        "record_tiers_total": log_out.get("tiers_total", []),
        "record_by_market": log_out.get("by_market", {}),
        "clv_by_market": log_out.get("clv_by_market", {}),
        "clv_by_tier": log_out.get("clv_by_tier", {}),
        "record_live_season": log_out.get("live_season"),
        "ratings": ratings_rows,
        "notes": notes,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT}")
    print(f"  picks: {len(payload['picks'])} "
          f"({sum(1 for p in payload['picks'] if p['qualified'])} qualified)")
    print(f"  total picks: {len(payload['total_picks'])} "
          f"({sum(1 for p in payload['total_picks'] if p['qualified'])} qualified)")
    print(f"  logged bets: {len(log)}  |  settled: {payload['record'].get('settled', 0)}")
    print(f"  {api_usage()}")
    if qb_value:
        top = sorted(qb_value.items(), key=lambda kv: -kv[1]["points"])[:3]
        print(f"  QB value ({data.get('qb_season')}): " +
              ", ".join(f"{k} {v['points']}" for k, v in top))

    tiers = payload["backtest"].get("tiers") or []
    if tiers:
        print(f"\n{'tier':>8} {'bets':>6} {'win%':>7} {'+/-':>5} "
              f"{'pred':>6} {'real':>6} {'calib':>6}")
        for r in tiers:
            print(f"{r['label']:>8} {r['bets']:>6} {r['win_pct']:>6}% "
                  f"{r['se']:>4} {r['pred_edge']:>6} {r['realized_edge']:>6} "
                  f"{r['calibration'] if r['calibration'] is not None else '--':>6}")
        print("\ncalib = realized edge / predicted edge. Near 1.0 means the model's")
        print("edges are honest. Near 0 means they carry no information.")


if __name__ == "__main__":
    main()
