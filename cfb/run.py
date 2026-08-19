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
    PriorModel, attach_epa, attach_lines, blend_prior, blend_ratings,
    build_games, cfbd_get, col, fit_ratings, games_played_counts,
    rating_diff, season_ratings, talent_composite, returning_production,
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
LOOKAHEAD_DAYS = int(CONFIG["lookahead_days"])
PRICE = float(CONFIG["assumed_price"])  # e.g. -110


def american_to_profit(price: float) -> float:
    """Profit per 1 unit risked on a win."""
    return (100.0 / abs(price)) if price < 0 else (price / 100.0)


# ======================================================================
# 1. data
# ======================================================================

def load_everything():
    print("Fetching data...")
    games_raw, lines_raw, adv_raw = [], [], []
    recruit, returning, fbs = [], [], {}

    for yr in HIST_YEARS:
        for st in ("regular", "postseason"):
            g = cfbd_get("games", {"year": yr, "seasonType": st})
            if len(g):
                games_raw.append(g)
        l = cfbd_get("lines", {"year": yr, "seasonType": "regular"})
        if len(l):
            lines_raw.append(l)
        a = cfbd_get("stats/game/advanced",
                     {"year": yr, "seasonType": "regular", "excludeGarbageTime": "true"})
        if len(a):
            adv_raw.append(a)
        t = cfbd_get("teams/fbs", {"year": yr})
        if len(t):
            c = col(t, "school", "team")
            fbs[yr] = set(t[c]) if c else set()
        rc = cfbd_get("recruiting/teams", {"year": yr})
        if len(rc):
            recruit.append(rc.assign(_year=yr))
        rp = cfbd_get("player/returning", {"year": yr})
        if len(rp):
            returning.append(rp.assign(_year=yr))
        print(f"  {yr}")

    # current season: never cached, always fresh
    t = cfbd_get("teams/fbs", {"year": CURRENT})
    if len(t):
        c = col(t, "school", "team")
        fbs[CURRENT] = set(t[c]) if c else set()
    rc = cfbd_get("recruiting/teams", {"year": CURRENT})
    if len(rc):
        recruit.append(rc.assign(_year=CURRENT))
    rp = cfbd_get("player/returning", {"year": CURRENT})
    if len(rp):
        returning.append(rp.assign(_year=CURRENT))

    cur_games = cfbd_get("games", {"year": CURRENT, "seasonType": "regular"}, force=True)
    cur_lines = cfbd_get("lines", {"year": CURRENT, "seasonType": "regular"}, force=True)
    print(f"  {CURRENT} (live)")

    # recruiting classes before the window, for the rolling average
    for yr in range(CONFIG["history_start"] - CONFIG["recruit_window"], CONFIG["history_start"]):
        r = cfbd_get("recruiting/teams", {"year": yr})
        if len(r):
            recruit.append(r.assign(_year=yr))

    cat = lambda fs: pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()
    return {
        "games": cat(games_raw), "lines": cat(lines_raw), "adv": cat(adv_raw),
        "recruit": cat(recruit), "returning": cat(returning), "fbs": fbs,
        "cur_games": cur_games, "cur_lines": cur_lines,
    }


# ======================================================================
# 2. backtest
# ======================================================================

def run_backtest(d, prior_model, season_hfa, use_epa):
    preds = []
    for season in TEST_SEASONS:
        prior = prior_model.preseason(season)
        if prior is None:
            continue
        earlier = [h for y, h in season_hfa.items() if y < season]
        hfa_ref = float(np.mean(earlier)) if earlier else 2.6

        cal = d[d.season < season]
        cal_x = rating_diff(cal, prior_model.season_r.get(season - 1), hfa_ref)
        ok = ~np.isnan(cal_x)
        if ok.sum() > 300:
            Ac = np.column_stack([cal_x[ok], np.ones(int(ok.sum()))])
            w_cal, *_ = np.linalg.lstsq(Ac, cal.loc[ok, "margin"].to_numpy(dtype=float), rcond=None)
        else:
            w_cal = np.array([1.0, 0.0])

        for wk in sorted(d.loc[d.season == season, "week"].unique()):
            so_far = d[(d.season == season) & (d.week < wk)]
            test = d[(d.season == season) & (d.week == wk)]
            if test.empty:
                continue
            r_in = None
            if len(so_far) >= 150:
                r_m, _ = fit_ratings(so_far, "margin", ridge=14.0, cap=35.0)
                r_p = None
                if use_epa:
                    r_p, _ = fit_ratings(so_far, "ppa_margin", ridge=1.0, cap=1.5)
                r_in = blend_ratings(r_m, r_p)
            R = blend_prior(prior, r_in, games_played_counts(so_far), BLEND_K)
            out = test.copy()
            out["pred"] = rating_diff(test, R, hfa_ref) * w_cal[0] + w_cal[1]
            preds.append(out)

    if not preds:
        return pd.DataFrame(), {}

    bt = pd.concat(preds, ignore_index=True)
    s = bt.dropna(subset=["mkt", "pred"]).copy()
    if not len(s):
        return bt, {}

    s["e_model"] = (s.pred - s.margin).abs()
    s["e_mkt"] = (s.mkt - s.margin).abs()
    s["edge"] = s.pred - s.mkt
    s["cover"] = np.where(s.margin > s.mkt, 1.0, np.where(s.margin < s.mkt, 0.0, np.nan))

    def slice_stats(frame):
        if not len(frame):
            return None
        return {
            "games": int(len(frame)),
            "model_mae": round(float(frame.e_model.mean()), 2),
            "market_mae": round(float(frame.e_mkt.mean()), 2),
        }

    qual = s[(s.edge.abs() >= MIN_EDGE) & s.cover.notna()]
    win = float(np.where(qual.edge > 0, qual.cover, 1 - qual.cover).mean()) if len(qual) > 20 else None

    thresholds = []
    for thr in (0, 2, 3, 4, 6, 8, 10):
        sub = s[(s.edge.abs() >= thr) & s.cover.notna()]
        if len(sub) < 40:
            continue
        thresholds.append({
            "edge": thr,
            "bets": int(len(sub)),
            "win_pct": round(float(np.where(sub.edge > 0, sub.cover, 1 - sub.cover).mean()) * 100, 1),
        })

    return bt, {
        "overall": slice_stats(s),
        "early": slice_stats(s[s.week <= 4]),
        "late": slice_stats(s[s.week >= 5]),
        "qualified_win_pct": round(win * 100, 1) if win is not None else None,
        "qualified_bets": int(len(qual)),
        "thresholds": thresholds,
        "seasons": TEST_SEASONS,
    }


# ======================================================================
# 3. picks
# ======================================================================

def find_picks(cur, ratings, hfa):
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

    up["pred"] = rating_diff(up, ratings, hfa)
    up["edge"] = up["pred"] - up["mkt"]
    up = up.dropna(subset=["edge"])
    if not len(up):
        return pd.DataFrame()

    up["side"] = np.where(up.edge > 0, "home", "away")
    up["pick_team"] = np.where(up.edge > 0, up.home_name, up.away_name)
    # Spread as it would be quoted for the side we like.
    up["pick_spread"] = np.where(up.edge > 0, -up["mkt"], up["mkt"])
    up["qualified"] = up.edge.abs() >= MIN_EDGE
    return up.sort_values("edge", key=lambda c: c.abs(), ascending=False).reset_index(drop=True)


# ======================================================================
# 4. bet log
# ======================================================================

def update_bet_log(picks, finished):
    """Append newly qualified wagers, then grade any that have completed."""
    cols = ["game_id", "season", "week", "logged_at", "away_name", "home_name",
            "pick_team", "side", "pred", "mkt_at_log", "pick_spread", "edge"]

    log = pd.read_csv(BET_LOG) if BET_LOG.exists() else pd.DataFrame(columns=cols)

    if len(picks):
        new = picks[picks.qualified].copy()
        if len(new):
            new["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            new = new.rename(columns={"mkt": "mkt_at_log"})
            new = new[[c for c in cols if c in new.columns]]
            log = pd.concat([log, new], ignore_index=True)

    if not len(log):
        return log, {}

    # One row per game: the first time it qualified is the number we'd have taken.
    log["game_id"] = pd.to_numeric(log["game_id"], errors="coerce")
    log = (log.dropna(subset=["game_id"])
              .sort_values("logged_at")
              .drop_duplicates("game_id", keep="first")
              .reset_index(drop=True))
    BET_LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(BET_LOG, index=False)

    # ---- grade ----
    res = finished[["game_id", "margin", "mkt"]].rename(columns={"mkt": "mkt_close"})
    g = log.merge(res, on="game_id", how="left")

    graded = g.dropna(subset=["margin"]).copy()
    if not len(graded):
        return log, {"pending": int(len(g)), "settled": 0}

    home = graded.side == "home"
    graded["result"] = np.select(
        [graded.margin == graded.mkt_at_log,
         home & (graded.margin > graded.mkt_at_log),
         (~home) & (graded.margin < graded.mkt_at_log)],
        ["push", "win", "win"], default="loss")

    profit = american_to_profit(PRICE)
    graded["units"] = graded.result.map({"win": profit, "loss": -1.0, "push": 0.0})
    graded["clv"] = np.where(home,
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
    return log, {"summary": summary, "rows": graded}


# ======================================================================
# main
# ======================================================================

def main():
    data = load_everything()

    d = build_games(data["games"], data["fbs"])
    d, sign = attach_lines(d, data["lines"])
    d, use_epa = attach_epa(d, data["adv"])
    d = d.sort_values(["season", "week"]).reset_index(drop=True)
    print(f"History: {len(d):,} games | lines {int(d.spread.notna().sum()):,} | EPA {use_epa}")

    season_r, season_hfa = season_ratings(d, HIST_YEARS, use_epa)
    talent = talent_composite(data["recruit"], HIST_YEARS + [CURRENT], CONFIG["recruit_window"])
    ret, ret_field = returning_production(data["returning"])
    prior_model = PriorModel(season_r, ret, talent)
    print(f"Prior model: {prior_model.n_train} team-seasons, R2 {prior_model.r2:.3f}")

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
        done = cur[cur.home_pts.notna()]
        played = int(len(done))
        r_in = None
        if played >= 150:
            r_m, _ = fit_ratings(done, "margin", ridge=14.0, cap=35.0)
            r_in = blend_ratings(r_m, None)
        gp = games_played_counts(done)
        ratings_now = blend_prior(prior, r_in, gp, BLEND_K)
        if prior:
            ws = [gp.get(t, 0) / (gp.get(t, 0) + BLEND_K) for t in prior if t != "NON_FBS"]
            in_season_weight = float(np.mean(ws)) if ws else 0.0

    # ---- backtest ----
    bt, bt_summary = run_backtest(d, prior_model, season_hfa, use_epa)

    # ---- picks + log ----
    picks = find_picks(cur, ratings_now, hfa) if len(cur) else pd.DataFrame()

    finished = d[["game_id", "margin", "mkt"]].copy()
    if len(cur):
        cf = cur[cur.home_pts.notna()][["game_id", "margin", "mkt"]]
        finished = pd.concat([finished, cf], ignore_index=True).drop_duplicates("game_id", keep="last")

    log, log_out = update_bet_log(picks, finished)

    # ---- ratings table ----
    ratings_rows = []
    if ratings_now:
        srt = sorted(((t, v) for t, v in ratings_now.items() if t != "NON_FBS"),
                     key=lambda kv: -kv[1])
        ratings_rows = [{"rank": i, "team": t, "rating": round(v, 2)}
                        for i, (t, v) in enumerate(srt, start=1)]

    def pick_rows(frame):
        rows = []
        for _, r in frame.iterrows():
            rows.append({
                "game_id": int(r.game_id),
                "week": int(r.week) if pd.notna(r.week) else None,
                "kickoff": r.kickoff.isoformat() if pd.notna(r.kickoff) else None,
                "away": r.away_name, "home": r.home_name,
                "pick": r.pick_team,
                "pick_spread": round(float(r.pick_spread), 1),
                "model_margin": round(float(r.pred), 1),
                "market_margin": round(float(r.mkt), 1),
                "edge": round(float(r.edge), 1),
                "qualified": bool(r.qualified),
            })
        return rows

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

    notes = []
    if in_season_weight < 0.30:
        notes.append(
            "Early season: ratings are mostly last year's, adjusted for returning "
            "production and recruiting. The model does not know about incoming "
            "transfers, so treat these numbers as weak."
        )
    if not use_epa:
        notes.append("EPA data unavailable this run; ratings are margin-only.")
    if prior_model.weights is None:
        notes.append("Preseason prior could not be fit. Check the data pull logs.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": CURRENT,
        "min_edge": MIN_EDGE,
        "assumed_price": PRICE,
        "state": {
            "games_played": played,
            "in_season_weight": round(in_season_weight, 3),
            "home_field": round(hfa, 2),
            "uses_epa": bool(use_epa),
            "returning_field": ret_field,
        },
        "prior_model": {
            "r2": round(prior_model.r2, 3) if prior_model.r2 == prior_model.r2 else None,
            "team_seasons": prior_model.n_train,
            "coefficients": {k: round(v, 3) for k, v in prior_model.coefficients().items()},
        },
        "backtest": bt_summary,
        "picks": pick_rows(picks) if len(picks) else [],
        "history": history_rows,
        "record": log_out.get("summary", {}),
        "ratings": ratings_rows,
        "notes": notes,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT}")
    print(f"  picks: {len(payload['picks'])} "
          f"({sum(1 for p in payload['picks'] if p['qualified'])} qualified)")
    print(f"  logged bets: {len(log)}  |  settled: {payload['record'].get('settled', 0)}")


if __name__ == "__main__":
    main()
