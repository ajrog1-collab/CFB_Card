"""
THE PRO CARD — NFL model.

Same machinery as the college model. The differences are all upstream:

  * One data source. nflverse's schedules file carries teams, scores, spread and
    total lines, rest days, roof, temperature and wind in a single CSV. That
    replaces the eight CFBD endpoints the college pipeline stitches together.
  * No lower divisions, so no pooling. All 32 teams are peers.
  * No recruiting or returning production. The preseason prior is prior-season
    ratings only, which is more predictive here anyway — rosters are stable and
    there are 32 teams instead of 130.
  * Far fewer games. 272 a season against roughly 800 in FBS, so ratings settle
    more slowly and a season yields tens of qualified picks, not hundreds. Read
    every number with that in mind.

Writes docs/data-nfl.json. The site reads it when the sport toggle is on PRO.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from model import (
    PriorModel, apply_debias, assign_tier, blend_prior, blend_ratings,
    fit_calibration, fit_debias, fit_points_ratings, fit_ratings, fit_uncertainty,
    confidence_features, confidence_score, games_played_counts, predict_total,
    predict_uncertainty, rating_diff, regime_cuts, tier_cutpoints, TIER_NAMES,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config-nfl.json").read_text())
CACHE = ROOT / "data" / "cache"
BET_LOG = ROOT / "data" / "bet_log_nfl.csv"
OUT = ROOT / "docs" / "data-nfl.json"

GAMES_URL  = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
LOGOS_URL  = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/logos.csv"
COLORS_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/teamcolors.csv"

CURRENT = int(CONFIG["current_season"])
HIST_START = int(CONFIG["history_start"])
TEST_SEASONS = CONFIG["backtest_seasons"]
HOLDOUT = CONFIG.get("holdout_season")
MIN_EDGE = float(CONFIG.get("min_edge", 2.0))
MIN_EDGE_TOTAL = float(CONFIG.get("min_edge_total", 3.0))
BLEND_K = float(CONFIG.get("blend_k", 4.0))
LOOKAHEAD_DAYS = int(CONFIG.get("lookahead_days", 9))
PRICE = float(CONFIG.get("assumed_price", -110))
TOP_PICK_COUNT = int(CONFIG.get("top_pick_count", 5))
DEBIAS = bool(CONFIG.get("debias", True))
AUTO_THRESHOLD = bool(CONFIG.get("auto_threshold", True))
TARGET_QUALIFIED = float(CONFIG.get("target_qualified_share", 0.25))
# Totals get their own, much stricter share. The pro model has no play-level
# data to rate offence and defence with, and backtested 1.4 points worse than
# the book while losing money. The picks stay visible for context; almost none
# of them should clear the bar.
TARGET_QUALIFIED_TOTAL = float(CONFIG.get("target_qualified_share_total", TARGET_QUALIFIED))
TIER_EDGES = CONFIG.get("tier_edges", [2, 3, 4, 6, 9])

DIVISIONS = {
    "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West", "KC": "AFC West", "LV": "AFC West", "LAC": "AFC West",
    "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
    "ARI": "NFC West", "LA": "NFC West", "SF": "NFC West", "SEA": "NFC West",
}
NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}
# Fallback colours, used if the colour file cannot be fetched. Logos come from
# nflverse rather than a guessed URL pattern: their abbreviations do not all
# match ESPN's slugs (LA vs lar, WAS vs wsh), and hand-mapping invites 404s.
FALLBACK_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB":  "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#101820", "KC":  "#E31837",
    "LA":  "#003594", "LAC": "#0080C6", "LV":  "#000000", "MIA": "#008E97",
    "MIN": "#4F2683", "NE":  "#002244", "NO":  "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#002244",
    "SF":  "#AA0000", "TB":  "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414",
}

# Cards show the nickname, but people search by city or abbreviation. These are
# matched against as well, without ever being displayed.
CITIES = {
    "ARI": "Arizona Phoenix", "ATL": "Atlanta", "BAL": "Baltimore",
    "BUF": "Buffalo", "CAR": "Carolina", "CHI": "Chicago",
    "CIN": "Cincinnati", "CLE": "Cleveland", "DAL": "Dallas",
    "DEN": "Denver", "DET": "Detroit", "GB": "Green Bay",
    "HOU": "Houston", "IND": "Indianapolis", "JAX": "Jacksonville",
    "KC": "Kansas City", "LA": "Los Angeles LA", "LAC": "Los Angeles LA",
    "LV": "Las Vegas Oakland", "MIA": "Miami", "MIN": "Minnesota",
    "NE": "New England Patriots Boston", "NO": "New Orleans",
    "NYG": "New York", "NYJ": "New York", "PHI": "Philadelphia",
    "PIT": "Pittsburgh", "SEA": "Seattle", "SF": "San Francisco Niners",
    "TB": "Tampa Bay", "TEN": "Tennessee", "WAS": "Washington",
}

CONF_GROUPS = [
    {"value": "afc", "label": "AFC",
     "members": ["AFC East", "AFC North", "AFC South", "AFC West"]},
    {"value": "nfc", "label": "NFC",
     "members": ["NFC East", "NFC North", "NFC South", "NFC West"]},
]


def american_to_profit(price: float) -> float:
    return (100.0 / abs(price)) if price < 0 else (price / 100.0)


# ======================================================================
# data
# ======================================================================

def load_games(force: bool = True) -> pd.DataFrame:
    """The whole dataset in one request.

    Cached to disk as a fallback only: if the fetch fails mid-season we would
    rather serve slightly stale numbers than nothing, but a successful run
    always refetches because lines move.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "nflverse_games.parquet"
    try:
        r = requests.get(GAMES_URL, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            print(f"  could not cache games: {e}")
        print(f"  fetched {len(df):,} games from nflverse")
        return df
    except Exception as e:
        print(f"  nflverse fetch failed ({e})")
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  falling back to cache: {len(df):,} games")
            return df
        raise RuntimeError("Could not load NFL schedule data and no cache exists.")


def load_team_meta() -> dict:
    """Logos and colours straight from nflverse, keyed by the same abbreviations
    the schedule uses. Degrades to initials on the team colour if unavailable."""
    meta = {t: {"abbr": t, "conf": DIVISIONS.get(t),
                "color": FALLBACK_COLORS.get(t), "alt": None,
                "alias": f"{CITIES.get(t, '')} {t}".strip(),
                "logo": None, "logo_dark": None} for t in NAMES}
    try:
        lg = pd.read_csv(io.StringIO(requests.get(LOGOS_URL, timeout=45).text))
        tcol = next((c for c in lg.columns if c.lower() in ("team", "team_abbr", "abbr")), None)
        url = next((c for c in lg.columns
                    if "url" in c.lower() or c.lower() in ("logo", "team_logo_espn")), None)
        dark = next((c for c in lg.columns if "dark" in c.lower() or "wordmark" in c.lower()), None)
        if tcol and url:
            for _, r in lg.iterrows():
                k = str(r[tcol])
                if k in meta and isinstance(r[url], str):
                    meta[k]["logo"] = r[url].replace("http://", "https://")
                    d = r[dark] if dark and isinstance(r.get(dark), str) else r[url]
                    meta[k]["logo_dark"] = str(d).replace("http://", "https://")
            print(f"  logos for {sum(1 for v in meta.values() if v['logo'])} teams")
    except Exception as e:
        print(f"  logo file unavailable ({e}); using initials")

    try:
        tc = pd.read_csv(io.StringIO(requests.get(COLORS_URL, timeout=45).text))
        tcol = next((c for c in tc.columns if c.lower() in ("team", "team_abbr", "abbr")), None)
        pcol = next((c for c in tc.columns
                     if c.lower() in ("color", "color1", "primary", "team_color")), None)
        if tcol and pcol:
            for _, r in tc.iterrows():
                k = str(r[tcol])
                if k in meta and isinstance(r[pcol], str):
                    meta[k]["color"] = r[pcol]
    except Exception as e:
        print(f"  colour file unavailable ({e}); using defaults")

    return {NAMES[k]: v for k, v in meta.items()}


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise into the same shape the shared model code expects."""
    d = df.copy()
    d = d[d["game_type"].astype(str).str.upper().isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
    d = d[(d["season"] >= HIST_START) & (d["season"] <= CURRENT)]

    out = pd.DataFrame({
        "game_id": d["game_id"].astype(str),
        "season": pd.to_numeric(d["season"], errors="coerce"),
        "week": pd.to_numeric(d["week"], errors="coerce"),
        "home_team": d["home_team"].astype(str),
        "away_team": d["away_team"].astype(str),
        "home_name": d["home_team"].astype(str),
        "away_name": d["away_team"].astype(str),
        "home_pts": pd.to_numeric(d.get("home_score"), errors="coerce"),
        "away_pts": pd.to_numeric(d.get("away_score"), errors="coerce"),
        "neutral": d.get("location", "Home").astype(str).str.lower().ne("home"),
        "home_rest": pd.to_numeric(d.get("home_rest"), errors="coerce"),
        "away_rest": pd.to_numeric(d.get("away_rest"), errors="coerce"),
        "div_game": pd.to_numeric(d.get("div_game"), errors="coerce").fillna(0),
        "roof": d.get("roof", "outdoors").astype(str).str.lower(),
        "temp": pd.to_numeric(d.get("temp"), errors="coerce"),
        "wind": pd.to_numeric(d.get("wind"), errors="coerce"),
        "home_qb": d.get("home_qb_name", pd.Series(dtype=str)).astype(str),
        "away_qb": d.get("away_qb_name", pd.Series(dtype=str)).astype(str),
    })

    # kickoff, from the date and time columns
    gd = d.get("gameday").astype(str)
    gt = d.get("gametime", pd.Series(["13:00"] * len(d))).astype(str).replace("nan", "13:00")
    out["kickoff"] = pd.to_datetime(gd + " " + gt, errors="coerce", utc=True) + pd.Timedelta(hours=5)

    out["margin"] = out["home_pts"] - out["away_pts"]
    out["actual_total"] = out["home_pts"] + out["away_pts"]

    # nflverse quotes spread_line as points the home team is favoured by, which
    # is already "expected home margin" — no sign detection needed.
    out["mkt"] = pd.to_numeric(d.get("spread_line"), errors="coerce")
    out["mkt_total"] = pd.to_numeric(d.get("total_line"), errors="coerce")

    # indoors means no weather rather than unknown weather
    indoors = out["roof"].isin(["dome", "closed"])
    out["indoor"] = indoors.astype(float)
    known_wind = out.loc[~indoors, "wind"].dropna()
    fill_wind = float(known_wind.mean()) if len(known_wind) > 200 else 8.0
    out["wind_excess"] = np.where(
        indoors, 0.0,
        np.where(out["wind"].notna(), np.clip(out["wind"].fillna(0) - 10, 0, 30),
                 max(fill_wind - 10, 0.0)))
    out["cold"] = np.where(indoors, 0.0,
                           np.clip(40 - out["temp"].fillna(55), 0, 60) / 10.0)
    out["precip_flag"] = 0.0
    out["wx_known"] = (~indoors & out["wind"].notna()).astype(float)

    out["rest_diff"] = np.clip(out["home_rest"].fillna(7) - out["away_rest"].fillna(7), -10, 10)
    out["rivalry"] = out["div_game"]        # division games are the NFL's rivalry proxy
    out["travel_diff_k"] = 0.0
    out["east_body"] = 0.0
    out["adj_margin"] = out["margin"]       # no turnover feed in this file

    return out.dropna(subset=["season", "week"]).sort_values(
        ["season", "week"]).reset_index(drop=True)


def situational(frame, rating_col):
    """Rest, plus a division-game compression term. Travel and body clock are
    absent: this file has no venue coordinates, and NFL travel is far less
    extreme than a Hawaii-to-Florida college trip."""
    cols, names = [], []
    if "rest_diff" in frame.columns:
        cols.append(frame["rest_diff"].fillna(0).to_numpy(dtype=float)); names.append("rest_diff")
    if "rivalry" in frame.columns:
        cols.append(frame["rivalry"].fillna(0).to_numpy(dtype=float) * rating_col)
        names.append("rivalry_compress")
    return (np.column_stack(cols) if cols else np.zeros((len(frame), 0))), names


# ======================================================================
# backtest
# ======================================================================

def season_ratings(d, years):
    R, H = {}, {}
    for yr in years:
        sub = d[d.season == yr]
        r, hfa = fit_ratings(sub, "margin", ridge=8.0, cap=35.0, min_games=120)
        if r is not None:
            R[yr], H[yr] = r, hfa
    return R, H


def tier_stats_generic(s, edge_col="edge", outcome_col="margin", line_col="mkt", thr_list=None):
    profit = american_to_profit(PRICE)
    edges = thr_list or TIER_EDGES
    f = s.dropna(subset=[edge_col, outcome_col, line_col]).copy()
    if not len(f):
        return []
    f["_r"] = np.where(f[edge_col] > 0, f[outcome_col] - f[line_col], f[line_col] - f[outcome_col])
    f["_p"] = f[edge_col].abs()
    out = []
    bounds = [(edges[i], edges[i+1] if i+1 < len(edges) else None) for i in range(len(edges))]
    for lo, hi in bounds:
        sub = f[(f._p >= lo) & ((f._p < hi) if hi else True)]
        dec = sub[sub._r != 0]
        if len(sub) < 10:
            continue
        wins = int((sub._r > 0).sum()); losses = int((sub._r < 0).sum())
        wp = wins / max(wins + losses, 1) * 100
        se = float(np.sqrt((wp/100)*(1-wp/100)/max(wins+losses, 1)) * 100)
        units = wins * profit - losses
        pred = float(sub._p.mean()); real = float(sub._r.mean())
        out.append({"label": f"{lo}-{hi}" if hi else f"{lo}+", "bets": int(len(sub)),
                    "wins": wins, "losses": losses, "pushes": int((sub._r == 0).sum()),
                    "win_pct": round(wp, 1), "se": round(se, 1),
                    "units": round(units, 2), "roi_pct": round(units/len(sub)*100, 1),
                    "pred_edge": round(pred, 1), "realized_edge": round(real, 1),
                    "calibration": round(real/pred, 2) if pred > 0.01 else None})
    return out


def season_breakdown(s, thr, outcome_col="margin", line_col="mkt"):
    profit = american_to_profit(PRICE)
    out = []
    for season in sorted(s["season"].dropna().unique()):
        sub = s[s.season == season]
        q = sub[sub["edge"].abs() >= thr].dropna(subset=[outcome_col, line_col])
        if len(q) < 10:
            continue
        realized = np.where(q["edge"] > 0, q[outcome_col] - q[line_col], q[line_col] - q[outcome_col])
        wins = int((realized > 0).sum()); losses = int((realized < 0).sum())
        units = wins * profit - losses
        row = {"season": int(season), "bets": int(len(q)), "wins": wins, "losses": losses,
               "pushes": int((realized == 0).sum()),
               "win_pct": round(wins/max(wins+losses, 1)*100, 1),
               "units": round(units, 2), "roi_pct": round(units/len(q)*100, 1)}
        if "e_model" in sub.columns:
            row["model_mae"] = round(float(sub.e_model.mean()), 2)
            row["market_mae"] = round(float(sub.e_mkt.mean()), 2)
            row["gap"] = round(float(sub.e_model.mean() - sub.e_mkt.mean()), 2)
            if "e_raw" in sub.columns and sub.e_raw.notna().any():
                row["gap_raw"] = round(float(sub.e_raw.mean() - sub.e_mkt.mean()), 2)
        out.append(row)
    return out


def run_backtest(d, prior_model, season_hfa):
    preds = []
    for season in TEST_SEASONS:
        prior = prior_model.preseason(season)
        if prior is None:
            continue
        earlier = [h for y, h in season_hfa.items() if y < season]
        hfa_ref = float(np.mean(earlier)) if earlier else 1.6

        cal = d[d.season < season]
        cal_x = rating_diff(cal, prior_model.season_r.get(season - 1), hfa_ref)
        cal_S, sit_names = situational(cal, cal_x)
        w_cal, sit_report = fit_calibration(
            cal_x, cal_S, sit_names, cal["margin"].to_numpy(dtype=float))

        for wk in sorted(d.loc[d.season == season, "week"].unique()):
            so_far = d[(d.season == season) & (d.week < wk)]
            test = d[(d.season == season) & (d.week == wk)]
            if test.empty:
                continue
            r_in = None
            if len(so_far) >= 60:
                r_in, _ = fit_ratings(so_far, "margin", ridge=10.0, cap=35.0, min_games=60)
            gp = games_played_counts(so_far)
            R = blend_prior(prior, r_in, gp, BLEND_K)
            rd = rating_diff(test, R, hfa_ref)
            S, _ = situational(test, rd)
            A = np.nan_to_num(np.column_stack([rd, S, np.ones(len(test))]), nan=0.0)
            out = test.copy()
            out["pred_raw"] = A @ w_cal
            out["pred_alt"] = rating_diff(test, prior, hfa_ref) * w_cal[0] + w_cal[-1]
            out["_gp_min"] = [min(gp.get(h, 0), gp.get(a, 0))
                              for h, a in zip(test["home_team"], test["away_team"])]
            preds.append(out)

    if not preds:
        return pd.DataFrame(), {}

    bt = pd.concat(preds, ignore_index=True)

    deb_a, deb_b = 0.0, 1.0
    if DEBIAS:
        bt["pred"] = bt["pred_raw"]
        for season in sorted(bt["season"].dropna().unique()):
            hist = bt[(bt.season < season) & bt.mkt.notna() & bt.pred_raw.notna()]
            a, b = fit_debias(hist["pred_raw"], hist["mkt"], min_n=200) if len(hist) >= 200 else (0.0, 1.0)
            m = bt.season == season
            bt.loc[m, "pred"] = apply_debias(bt.loc[m, "pred_raw"].to_numpy(dtype=float),
                                             bt.loc[m, "mkt"].to_numpy(dtype=float), a, b)
        full = bt.dropna(subset=["mkt", "pred_raw"])
        if len(full) >= 200:
            deb_a, deb_b = fit_debias(full["pred_raw"], full["mkt"], min_n=200)
    else:
        bt["pred"] = bt["pred_raw"]

    s = bt.dropna(subset=["mkt", "pred", "margin"]).copy()
    if not len(s):
        return bt, {}

    s["e_model"] = (s.pred - s.margin).abs()
    s["e_mkt"] = (s.mkt - s.margin).abs()
    s["e_raw"] = (s.pred_raw - s.margin).abs()
    s["edge"] = s.pred - s.mkt
    s["cover"] = np.where(s.margin > s.mkt, 1.0, np.where(s.margin < s.mkt, 0.0, np.nan))

    X, cnames = confidence_features(s, pred_col="pred", alt_pred_col="pred_alt")
    imm = 4.0 / np.clip(s["_gp_min"].to_numpy(dtype=float), 1.0, None)
    X = np.column_stack([X, imm]) if X.size else imm.reshape(-1, 1)
    cnames = cnames + ["immaturity"]
    conf_w, conf_mae = fit_uncertainty(X, (s.pred - s.margin).to_numpy(dtype=float))
    s["sigma"] = predict_uncertainty(X, conf_w, conf_mae)
    s["confidence"] = confidence_score(s["edge"].to_numpy(dtype=float),
                                       s["sigma"].to_numpy(dtype=float))

    thr = MIN_EDGE
    if AUTO_THRESHOLD and len(s) > 200:
        thr = round(max(float(np.quantile(s["edge"].abs().dropna(), 1 - TARGET_QUALIFIED)), 0.5), 1)

    qual = s[(s.edge.abs() >= thr) & s.cover.notna()]
    win = float(np.where(qual.edge > 0, qual.cover, 1 - qual.cover).mean()) if len(qual) > 20 else None
    hold = s[s.season == HOLDOUT] if HOLDOUT else pd.DataFrame()
    hq = hold[(hold.edge.abs() >= thr) & hold.cover.notna()] if len(hold) else pd.DataFrame()
    hwin = float(np.where(hq.edge > 0, hq.cover, 1 - hq.cover).mean()) if len(hq) > 15 else None

    def slice_stats(f):
        if not len(f):
            return None
        return {"games": int(len(f)), "model_mae": round(float(f.e_model.mean()), 2),
                "market_mae": round(float(f.e_mkt.mean()), 2),
                "model_mae_raw": round(float(f.e_raw.mean()), 2)}

    cuts_early = tier_cutpoints(s.loc[s.week <= 4, "confidence"])
    cuts_late = tier_cutpoints(s.loc[s.week >= 5, "confidence"])
    cuts_all = tier_cutpoints(s["confidence"])
    cuts = {"early": cuts_early or cuts_all, "late": cuts_late or cuts_all, "all": cuts_all}

    tier_trend, tier_summary = {}, {}
    if cuts_all:
        s["_tier"] = [assign_tier(c, regime_cuts(cuts, week=w))
                      for c, w in zip(s["confidence"], s["week"])]
        realized = np.where(s.edge > 0, s.margin - s.mkt, s.mkt - s.margin)
        s["_res"] = np.where(realized > 0, "win", np.where(realized < 0, "loss", "push"))
        profit = american_to_profit(PRICE)

        def summarize(f):
            dec = f[f._res != "push"]
            if not len(dec):
                return None
            wins = int((f._res == "win").sum()); losses = int((f._res == "loss").sum())
            units = wins * profit - losses
            return {"bets": int(len(f)), "wins": wins, "losses": losses,
                    "pushes": int((f._res == "push").sum()),
                    "win_pct": round(wins/(wins+losses)*100, 1) if wins+losses else None,
                    "units": round(units, 2), "roi_pct": round(units/len(f)*100, 1),
                    "avg_conf": round(float(f.confidence.mean()), 2),
                    "avg_gap": round(float(f.edge.abs().mean()), 1)}

        for name in TIER_NAMES:
            sub = s[s._tier == name]
            if len(sub) < 25:
                continue
            tier_summary[name] = summarize(sub)
            seasons = []
            for season in sorted(sub.season.dropna().unique()):
                ss = sub[sub.season == season]
                if len(ss) < 8:
                    continue
                row = summarize(ss)
                if row:
                    row["season"] = int(season); seasons.append(row)
            if seasons:
                tier_trend[name] = seasons
        allr = summarize(s)
        if allr:
            tier_summary["ALL"] = allr
            seasons = []
            for season in sorted(s.season.dropna().unique()):
                ss = s[s.season == season]
                if len(ss) < 15:
                    continue
                row = summarize(ss)
                if row:
                    row["season"] = int(season); seasons.append(row)
            if seasons:
                tier_trend["ALL"] = seasons

    x = s["mkt"].to_numpy(dtype=float); y = s["edge"].to_numpy(dtype=float)
    bias = {
        "edge_vs_line_corr": round(float(np.corrcoef(x, y)[0, 1]), 3) if x.std() > 0 and y.std() > 0 else 0.0,
        "pct_on_underdog": round(float((((x > 0) != (y > 0))).mean()) * 100, 1),
        "qualified_share": round(float((s.edge.abs() >= thr).mean()) * 100, 1),
    }

    return bt, {
        "overall": slice_stats(s), "early": slice_stats(s[s.week <= 4]),
        "late": slice_stats(s[s.week >= 5]),
        "qualified_win_pct": round(win * 100, 1) if win is not None else None,
        "qualified_bets": int(len(qual)),
        "holdout": ({"season": HOLDOUT, "win_pct": round(hwin*100, 1) if hwin is not None else None,
                     "bets": int(len(hq))} if HOLDOUT and len(hq) else None),
        "tiers": tier_stats_generic(s), "seasons": TEST_SEASONS,
        "by_season": season_breakdown(s, thr), "min_edge": thr,
        "tier_cuts": cuts, "tier_trend": tier_trend, "tier_summary": tier_summary,
        "situational": sit_report, "bias_check": bias,
        "debias": {"intercept": round(deb_a, 3), "slope": round(deb_b, 3), "active": bool(DEBIAS)},
        "conf_weights": ([float(v) for v in conf_w] if conf_w is not None else None),
        "conf_names": cnames,
        "conf_mae": round(conf_mae, 2) if conf_mae == conf_mae else None,
    }


# ======================================================================
# totals
# ======================================================================

def totals_matrix(frame):
    cols, names = [], []
    for c in ("wind_excess", "cold", "rest_diff"):
        if c in frame.columns:
            cols.append(frame[c].fillna(0).to_numpy(dtype=float)); names.append(c)
    return (np.column_stack(cols) if cols else np.zeros((len(frame), 0))), names


def run_totals(d):
    preds = []
    for season in TEST_SEASONS:
        for wk in sorted(d.loc[(d.season == season) & (d.week >= 4), "week"].unique()):
            so_far = d[(d.season == season) & (d.week < wk)]
            test = d[(d.season == season) & (d.week == wk)]
            if len(so_far) < 80 or test.empty:
                continue
            off, dfn, hfa_off, base = fit_points_ratings(so_far, ridge=10.0, cap=56.0, min_games=80)
            if off is None:
                continue
            raw_prev = predict_total(so_far, off, dfn, hfa_off, base)
            S_prev, _ = totals_matrix(so_far)
            ok = ~np.isnan(raw_prev) & so_far["actual_total"].notna().to_numpy()
            if ok.sum() < 80:
                continue
            A = np.column_stack([raw_prev[ok], S_prev[ok], np.ones(int(ok.sum()))])
            w, *_ = np.linalg.lstsq(A, so_far.loc[ok, "actual_total"].to_numpy(dtype=float), rcond=None)
            raw = predict_total(test, off, dfn, hfa_off, base)
            S, _ = totals_matrix(test)
            out = test.copy()
            out["pred_total_raw"] = np.nan_to_num(
                np.column_stack([raw, S, np.ones(len(test))]), nan=0.0) @ w
            preds.append(out)

    if not preds:
        return pd.DataFrame(), {}
    bt = pd.concat(preds, ignore_index=True)

    t_a, t_b = 0.0, 1.0
    if DEBIAS:
        bt["pred_total"] = bt["pred_total_raw"]
        for season in sorted(bt.season.dropna().unique()):
            hist = bt[(bt.season < season) & bt.mkt_total.notna() & bt.pred_total_raw.notna()]
            a, b = fit_debias(hist["pred_total_raw"], hist["mkt_total"], min_n=200) if len(hist) >= 200 else (0.0, 1.0)
            m = bt.season == season
            bt.loc[m, "pred_total"] = apply_debias(
                bt.loc[m, "pred_total_raw"].to_numpy(dtype=float),
                bt.loc[m, "mkt_total"].to_numpy(dtype=float), a, b)
        full = bt.dropna(subset=["mkt_total", "pred_total_raw"])
        if len(full) >= 200:
            t_a, t_b = fit_debias(full["pred_total_raw"], full["mkt_total"], min_n=200)
    else:
        bt["pred_total"] = bt["pred_total_raw"]

    s = bt.dropna(subset=["mkt_total", "pred_total", "actual_total"]).copy()
    if not len(s):
        return bt, {}
    s["e_model"] = (s.pred_total - s.actual_total).abs()
    s["e_mkt"] = (s.mkt_total - s.actual_total).abs()
    s["edge"] = s.pred_total - s.mkt_total
    s["cover"] = np.where(s.actual_total > s.mkt_total, 1.0,
                          np.where(s.actual_total < s.mkt_total, 0.0, np.nan))

    thr = MIN_EDGE_TOTAL
    if AUTO_THRESHOLD and len(s) > 200:
        thr = round(max(float(np.quantile(s["edge"].abs().dropna(), 1 - TARGET_QUALIFIED_TOTAL)),
                        MIN_EDGE_TOTAL), 1)
    qual = s[(s.edge.abs() >= thr) & s.cover.notna()]
    win = float(np.where(qual.edge > 0, qual.cover, 1 - qual.cover).mean()) if len(qual) > 20 else None

    return bt, {
        "games": int(len(s)),
        "model_mae": round(float(s.e_model.mean()), 2),
        "market_mae": round(float(s.e_mkt.mean()), 2),
        "qualified_win_pct": round(win * 100, 1) if win is not None else None,
        "qualified_bets": int(len(qual)), "min_edge": thr,
        "tiers": tier_stats_generic(s, outcome_col="actual_total", line_col="mkt_total"),
        "by_season": season_breakdown(s, thr, "actual_total", "mkt_total"),
        "seasons": TEST_SEASONS, "uses_weather": True,
        "debias": {"intercept": round(t_a, 3), "slope": round(t_b, 3), "active": bool(DEBIAS)},
        "bias_check": {
            "pct_over": round(float((s.edge > 0).mean()) * 100, 1),
            "edge_vs_line_corr": (round(float(np.corrcoef(s.mkt_total, s.edge)[0, 1]), 3)
                                  if len(s) > 100 else None),
            "qualified_share": round(float((s.edge.abs() >= thr).mean()) * 100, 1)},
    }


# ======================================================================
# picks, log, payload
# ======================================================================

def find_picks(cur, ratings, hfa, sit_weights, debias, conf, gp, min_edge):
    now = datetime.now(timezone.utc)
    up = cur[cur.home_pts.isna() & cur.kickoff.notna()
             & (cur.kickoff > now) & (cur.kickoff < now + timedelta(days=LOOKAHEAD_DAYS))].copy()
    if not len(up):
        return pd.DataFrame()

    rd = rating_diff(up, ratings, hfa)
    scale = (sit_weights or {}).get("rating_scale", 1.0) or 1.0
    S, names = situational(up, rd)
    adj = np.zeros(len(up))
    for j, nm in enumerate(names):
        v = (sit_weights or {}).get(nm, 0.0)
        adj += S[:, j] * (v if isinstance(v, (int, float)) else 0.0)
    up["pred_raw"] = rd * scale + adj

    if DEBIAS:
        a = float((debias or {}).get("intercept", 0.0))
        b = float((debias or {}).get("slope", 1.0))
        fit = up.dropna(subset=["pred_raw", "mkt"])
        if len(fit) >= 12 and fit["mkt"].std() > 1:
            sa, sb = fit_debias(fit["pred_raw"], fit["mkt"], min_n=12)
            if (sa, sb) != (0.0, 1.0):
                a, b = sa, sb
        up["pred"] = apply_debias(up["pred_raw"].to_numpy(dtype=float),
                                  up["mkt"].to_numpy(dtype=float), a, b)
    else:
        up["pred"] = up["pred_raw"]

    up["edge"] = up["pred"] - up["mkt"]
    up = up.dropna(subset=["edge"])
    if not len(up):
        return pd.DataFrame()

    up["pred_alt"] = up["pred"]
    X, cn = confidence_features(up, pred_col="pred", alt_pred_col="pred_alt")
    imm = np.array([4.0 / max(min(gp.get(h, 0), gp.get(a, 0)), 1.0)
                    for h, a in zip(up["home_team"], up["away_team"])])
    X = np.column_stack([X, imm]) if X.size else imm.reshape(-1, 1)
    cn = cn + ["immaturity"]
    cw = conf.get("weights")
    up["sigma"] = (predict_uncertainty(X, np.asarray(cw), conf.get("mae"))
                   if cw is not None and list(cn) == list(conf.get("names") or [])
                   else (conf.get("mae") or 12.0))
    up["confidence"] = confidence_score(up["edge"].to_numpy(dtype=float),
                                        up["sigma"].to_numpy(dtype=float))

    up["side"] = np.where(up.edge > 0, "home", "away")
    up["pick_team"] = np.where(up.edge > 0, up.home_name, up.away_name)
    up["pick_spread"] = np.where(up.edge > 0, -up["mkt"], up["mkt"])
    up["qualified"] = up.edge.abs() >= min_edge
    up = up.sort_values("confidence", ascending=False).reset_index(drop=True)
    up["top_pick"] = False
    up.loc[up.index[up.qualified][:TOP_PICK_COUNT], "top_pick"] = True
    return up


def find_total_picks(cur, off, dfn, hfa_off, base, w, debias, min_edge):
    now = datetime.now(timezone.utc)
    up = cur[cur.home_pts.isna() & cur.kickoff.notna()
             & (cur.kickoff > now) & (cur.kickoff < now + timedelta(days=LOOKAHEAD_DAYS))].copy()
    if not len(up) or off is None or w is None:
        return pd.DataFrame()
    raw = predict_total(up, off, dfn, hfa_off, base)
    S, _ = totals_matrix(up)
    A = np.nan_to_num(np.column_stack([raw, S, np.ones(len(up))]), nan=0.0)
    if A.shape[1] != len(w):
        return pd.DataFrame()
    up["pred_total_raw"] = A @ w
    if DEBIAS:
        a = float((debias or {}).get("intercept", 0.0))
        b = float((debias or {}).get("slope", 1.0))
        fit = up.dropna(subset=["pred_total_raw", "mkt_total"])
        if len(fit) >= 12 and fit["mkt_total"].std() > 1:
            sa, sb = fit_debias(fit["pred_total_raw"], fit["mkt_total"], min_n=12)
            if (sa, sb) != (0.0, 1.0):
                a, b = sa, sb
        up["pred_total"] = apply_debias(up["pred_total_raw"].to_numpy(dtype=float),
                                        up["mkt_total"].to_numpy(dtype=float), a, b)
    else:
        up["pred_total"] = up["pred_total_raw"]
    up["edge"] = up["pred_total"] - up["mkt_total"]
    up = up.dropna(subset=["edge"])
    if not len(up):
        return pd.DataFrame()
    up["side"] = np.where(up.edge > 0, "over", "under")
    up["pick_team"] = np.where(up.edge > 0, "Over", "Under")
    up["qualified"] = up.edge.abs() >= min_edge
    up["confidence"] = up["edge"].abs() / 12.0
    up = up.sort_values("confidence", ascending=False).reset_index(drop=True)
    up["top_pick"] = False
    up.loc[up.index[up.qualified][:TOP_PICK_COUNT], "top_pick"] = True
    return up


def update_log(picks, total_picks, finished, cuts):
    cols = ["game_id", "market", "season", "week", "logged_at", "away_name", "home_name",
            "pick_team", "side", "pred", "mkt_at_log", "pick_spread", "edge",
            "confidence", "top_pick"]
    log = pd.read_csv(BET_LOG) if BET_LOG.exists() else pd.DataFrame(columns=cols)
    if "market" not in log.columns:
        log["market"] = "spread"

    def add(frame, market, pred_col, line_col):
        nonlocal log
        if frame is None or not len(frame):
            return
        new = frame[frame.qualified].copy()
        if not len(new):
            return
        new["market"] = market
        new["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        new = new.rename(columns={line_col: "mkt_at_log", pred_col: "pred"})
        if "pick_spread" not in new.columns:
            new["pick_spread"] = new["mkt_at_log"]
        log = pd.concat([log, new[[c for c in cols if c in new.columns]]], ignore_index=True)

    add(picks, "spread", "pred", "mkt")
    add(total_picks, "total", "pred_total", "mkt_total")
    if not len(log):
        return log, {}

    log = (log.dropna(subset=["game_id"]).sort_values("logged_at")
              .drop_duplicates(["game_id", "market"], keep="first").reset_index(drop=True))
    BET_LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(BET_LOG, index=False)

    res = finished[["game_id", "margin", "mkt", "actual_total", "mkt_total"]].rename(
        columns={"mkt": "close_sp", "mkt_total": "close_to"})
    g = log.merge(res, on="game_id", how="left")
    is_tot = g["market"] == "total"
    g["outcome"] = np.where(is_tot, g["actual_total"], g["margin"])
    g["mkt_close"] = np.where(is_tot, g["close_to"], g["close_sp"])
    graded = g.dropna(subset=["outcome"]).copy()
    if not len(graded):
        return log, {"pending": int(len(g)), "settled": 0}

    over_home = graded["side"].isin(["home", "over"])
    graded["result"] = np.select(
        [graded.outcome == graded.mkt_at_log,
         over_home & (graded.outcome > graded.mkt_at_log),
         (~over_home) & (graded.outcome < graded.mkt_at_log)],
        ["push", "win", "win"], default="loss")
    profit = american_to_profit(PRICE)
    graded["units"] = graded.result.map({"win": profit, "loss": -1.0, "push": 0.0})
    graded["clv"] = np.where(over_home, graded.mkt_close - graded.mkt_at_log,
                             graded.mkt_at_log - graded.mkt_close)

    dec = graded[graded.result != "push"]
    clv = graded.dropna(subset=["clv"])
    summary = {
        "settled": int(len(graded)), "pending": int(len(g) - len(graded)),
        "wins": int((graded.result == "win").sum()),
        "losses": int((graded.result == "loss").sum()),
        "pushes": int((graded.result == "push").sum()),
        "win_pct": round(float((dec.result == "win").mean()) * 100, 1) if len(dec) else None,
        "units": round(float(graded.units.sum()), 2),
        "roi_pct": round(float(graded.units.sum() / len(graded)) * 100, 1),
        "avg_clv": round(float(clv.clv.mean()), 2) if len(clv) >= 5 else None,
        "beat_close_pct": round(float((clv.clv > 0).mean()) * 100, 1) if len(clv) >= 5 else None,
        "breakeven_pct": round(100.0 / (1.0 + profit), 1),
    }

    def market_summary(f):
        if not len(f):
            return None
        d2 = f[f.result != "push"]
        u = float(f.units.sum())
        return {"settled": int(len(f)), "wins": int((f.result == "win").sum()),
                "losses": int((f.result == "loss").sum()),
                "pushes": int((f.result == "push").sum()),
                "win_pct": round(float((d2.result == "win").mean()) * 100, 1) if len(d2) else None,
                "units": round(u, 2), "roi_pct": round(u / len(f) * 100, 1)}

    def clv_stats(f):
        c = f.dropna(subset=["clv"])
        if len(c) < 3:
            return None
        return {"n": int(len(c)), "avg": round(float(c.clv.mean()), 2),
                "beat_pct": round(float((c.clv > 0).mean()) * 100, 1),
                "median": round(float(c.clv.median()), 2)}

    sp = graded[graded.market != "total"]; to = graded[graded.market == "total"]
    top = graded[graded.get("top_pick", pd.Series(False, index=graded.index)) == True]
    clv_tier = {}
    if cuts and "confidence" in graded.columns:
        graded["_tier"] = [assign_tier(c, regime_cuts(cuts, week=w))
                           for c, w in zip(graded["confidence"],
                                           graded.get("week", pd.Series([9]*len(graded))))]
        for n in TIER_NAMES:
            st = clv_stats(graded[graded._tier == n])
            if st:
                clv_tier[n] = st

    return log, {
        "summary": summary, "rows": graded.sort_values("logged_at", ascending=False),
        "by_market": {"spread": market_summary(sp), "total": market_summary(to),
                      "top": market_summary(top)},
        "clv_by_market": {"spread": clv_stats(sp), "total": clv_stats(to),
                          "all": clv_stats(graded)},
        "clv_by_tier": clv_tier,
    }


def main():
    print("THE PRO CARD — NFL")
    raw = load_games()
    d = build(raw)
    hist = d[d.season < CURRENT]
    cur = d[d.season == CURRENT]
    print(f"History: {len(hist):,} games | current season rows: {len(cur):,}")
    print(f"Lines present: {int(d.mkt.notna().sum()):,} spreads, {int(d.mkt_total.notna().sum()):,} totals")

    years = list(range(HIST_START, CURRENT))
    season_r, season_hfa = season_ratings(hist, years)
    prior_model = PriorModel(season_r, {}, {}, sp={})
    hfa = float(np.mean(list(season_hfa.values()))) if season_hfa else 1.6
    print(f"Prior: {prior_model.n_train} team-seasons, R2 {prior_model.r2:.3f}, "
          f"features {prior_model.feats} | home field {hfa:.2f}")

    bt, bts = run_backtest(hist, prior_model, season_hfa)
    tbt, tts = run_totals(hist)
    if bts:
        print(f"Spreads: model {bts['overall']['model_mae']} vs book {bts['overall']['market_mae']} "
              f"| qualified {bts['qualified_win_pct']}% on {bts['qualified_bets']}")
    if tts:
        print(f"Totals : model {tts['model_mae']} vs book {tts['market_mae']} "
              f"| qualified {tts['qualified_win_pct']}% on {tts['qualified_bets']}")

    prior = prior_model.preseason(CURRENT)
    played = cur[cur.home_pts.notna()]
    r_in, _ = fit_ratings(played, "margin", ridge=10.0, cap=35.0, min_games=60) if len(played) >= 60 else (None, 0)
    gp = games_played_counts(played)
    ratings_now = blend_prior(prior, r_in, gp, BLEND_K)
    ws = [gp.get(t, 0)/(gp.get(t, 0)+BLEND_K) for t in (prior or {})]
    in_season_weight = float(np.mean(ws)) if ws else 0.0

    conf = {"weights": bts.get("conf_weights"), "names": bts.get("conf_names"),
            "mae": bts.get("conf_mae")}
    picks = find_picks(cur, ratings_now, hfa, bts.get("situational"), bts.get("debias"),
                       conf, gp, bts.get("min_edge", MIN_EDGE)) if len(cur) else pd.DataFrame()

    t_off = t_dfn = t_w = None; t_hfa = t_base = 0.0
    if len(hist) > 200:
        t_off, t_dfn, t_hfa, t_base = fit_points_ratings(d[d.home_pts.notna()], ridge=10.0, cap=56.0)
        if t_off is not None:
            fin = d[d.home_pts.notna()]
            rawt = predict_total(fin, t_off, t_dfn, t_hfa, t_base)
            S, _ = totals_matrix(fin)
            ok = ~np.isnan(rawt) & fin["actual_total"].notna().to_numpy()
            if ok.sum() > 100:
                A = np.column_stack([rawt[ok], S[ok], np.ones(int(ok.sum()))])
                t_w, *_ = np.linalg.lstsq(A, fin.loc[ok, "actual_total"].to_numpy(dtype=float), rcond=None)
    total_picks = (find_total_picks(cur, t_off, t_dfn, t_hfa, t_base, t_w,
                                    tts.get("debias"), tts.get("min_edge", MIN_EDGE_TOTAL))
                   if len(cur) else pd.DataFrame())

    finished = d[d.home_pts.notna()][["game_id", "margin", "mkt", "actual_total", "mkt_total"]]
    log, log_out = update_log(picks, total_picks, finished, bts.get("tier_cuts"))

    ranks = {}
    if ratings_now:
        for i, (t, _v) in enumerate(sorted(ratings_now.items(), key=lambda kv: -kv[1]), start=1):
            ranks[t] = i

    def drivers(r, is_total=False):
        out = []
        gpm = min(gp.get(r.get("home_team"), 0), gp.get(r.get("away_team"), 0))
        if gpm == 0:
            out.append({"kind": "caution", "text": "Preseason ratings only, no games played"})
        elif gpm < 3:
            out.append({"kind": "caution", "text": f"Only {gpm} game{'s' if gpm != 1 else ''} of data"})
        rest = float(r.get("rest_diff") or 0)
        if abs(rest) >= 3:
            side = r.get("home_name") if rest > 0 else r.get("away_name")
            out.append({"kind": "rest",
                        "text": f"{NAMES.get(side, side)} on {abs(rest):.0f} more days of rest"})
        if is_total:
            w = r.get("wind")
            if r.get("indoor"):
                out.append({"kind": "weather", "text": "Indoors, weather removed"})
            elif w is not None and w == w and float(w) >= 15:
                out.append({"kind": "weather", "text": f"{float(w):.0f} mph wind"})
            t = r.get("temp")
            if t is not None and t == t and float(t) <= 25:
                out.append({"kind": "weather", "text": f"{float(t):.0f}\u00b0F at kickoff"})
        else:
            if r.get("div_game"):
                out.append({"kind": "rivalry", "text": "Division game, historically closer"})
            rp, rf = ranks.get(r.get("home_team")), ranks.get(r.get("away_team"))
            if rp and rf and abs(rp - rf) >= 12:
                hi, lo = (r.get("home_name"), r.get("away_name")) if rp < rf else (r.get("away_name"), r.get("home_name"))
                out.append({"kind": "rating",
                            "text": f"Model ranks {NAMES.get(hi, hi)} #{min(rp, rf)}, "
                                    f"{NAMES.get(lo, lo)} #{max(rp, rf)}"})
        return out[:2]

    def rows(frame, is_total=False):
        res = []
        for _, r in frame.iterrows():
            base = {
                "game_id": r.game_id, "week": int(r.week) if pd.notna(r.week) else None,
                "kickoff": r.kickoff.isoformat() if pd.notna(r.kickoff) else None,
                "away": NAMES.get(r.away_name, r.away_name),
                "home": NAMES.get(r.home_name, r.home_name),
                "pick": r.pick_team if not is_total else r.pick_team,
                "edge": round(float(r.edge), 1),
                "confidence": round(float(r.confidence), 2) if pd.notna(r.confidence) else None,
                "tier": assign_tier(r.get("confidence"),
                                    regime_cuts(bts.get("tier_cuts"),
                                                in_season_weight=in_season_weight)),
                "top_pick": bool(r.top_pick), "qualified": bool(r.qualified),
                "drivers": drivers(r, is_total),
            }
            if is_total:
                base.update({"line": round(float(r.mkt_total), 1),
                             "model_total": round(float(r.pred_total), 1),
                             "wind": (round(float(r.wind), 1) if pd.notna(r.get("wind")) else None),
                             "indoor": bool(r.get("indoor", 0))})
            else:
                base.update({"pick_spread": round(float(r.pick_spread), 1),
                             "model_margin": round(float(r.pred), 1),
                             "market_margin": round(float(r.mkt), 1),
                             "sigma": round(float(r.sigma), 1) if pd.notna(r.get("sigma")) else None})
            res.append(base)
        return res

    teams_meta = load_team_meta()

    equity, hist_rows = [], []
    if "rows" in log_out and len(log_out["rows"]):
        eq = log_out["rows"].sort_values("logged_at"); run = 0.0
        for i, (_, r) in enumerate(eq.iterrows(), start=1):
            run += float(r.units)
            equity.append({"n": i, "units": round(run, 2),
                           "market": "total" if r.market == "total" else "spread"})
        for _, r in log_out["rows"].iterrows():
            hist_rows.append({
                "week": int(r.week) if pd.notna(r.week) else None,
                "away": NAMES.get(r.away_name, r.away_name),
                "home": NAMES.get(r.home_name, r.home_name),
                "pick": r.pick_team,
                "pick_spread": round(float(r.pick_spread), 1) if pd.notna(r.pick_spread) else None,
                "edge": round(float(r.edge), 1) if pd.notna(r.edge) else None,
                "margin": int(r.outcome) if pd.notna(r.outcome) else None,
                "result": r.result, "units": round(float(r.units), 2),
                "clv": round(float(r.clv), 2) if pd.notna(r.clv) else None})

    notes = []
    if in_season_weight < 0.35:
        notes.append("Early season: ratings lean on last year. The model does not know "
                     "about injuries, holdouts or scheme changes.")
    notes.append("272 regular-season games a year means far fewer picks than college and "
                 "wider error bars on every number here.")
    notes.append("Totals are shown for context but rarely qualify. Without play-level data "
                 "the pro totals model ran 1.4 points behind the book and lost money in "
                 "backtest, so the bar for a totals pick is set very high.")

    def balance(frame, kind):
        if not len(frame):
            return None
        e = frame["edge"].dropna()
        if not len(e):
            return None
        side = (((frame["mkt"] > 0) != (frame["edge"] > 0)).mean() if kind == "spread"
                else (frame["edge"] > 0).mean())
        out = {"n": int(len(e)), "pct": round(float(side)*100, 1),
               "side": "underdog" if kind == "spread" else "over"}
        line = frame["mkt"] if kind == "spread" else frame["mkt_total"]
        if len(e) > 8 and frame["edge"].std() > 0 and line.std() > 0:
            out["corr"] = round(float(np.corrcoef(line, frame["edge"])[0, 1]), 3)
        return out

    payload = {
        "sport": "nfl", "title": "THE PRO CARD",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": CURRENT, "min_edge": bts.get("min_edge", MIN_EDGE),
        "assumed_price": PRICE, "top_pick_count": TOP_PICK_COUNT,
        "state": {"games_played": int(len(played)),
                  "in_season_weight": round(in_season_weight, 3),
                  "home_field": round(hfa, 2), "uses_epa": False,
                  "uses_turnovers": False, "uses_weather": True,
                  "weather_games": int(d["wind"].notna().sum()),
                  "domes": int(d.loc[d.indoor > 0, "home_team"].nunique()),
                  "turnover_points": None, "turnover_shrink": None,
                  "returning_field": None, "uses_sp": False},
        "prior_model": {"r2": round(prior_model.r2, 3) if prior_model.r2 == prior_model.r2 else None,
                        "team_seasons": prior_model.n_train,
                        "coefficients": {k: round(v, 3) for k, v in prior_model.coefficients().items()}},
        "backtest": bts, "totals": tts,
        "conf_label": "Div", "conf_groups": CONF_GROUPS,
        "gap_trend": [], "teams": teams_meta,
        "picks": rows(picks) if len(picks) else [],
        "total_picks": rows(total_picks, True) if len(total_picks) else [],
        "history": hist_rows, "equity": equity,
        "record": log_out.get("summary", {}),
        "record_by_market": log_out.get("by_market", {}),
        "clv_by_market": log_out.get("clv_by_market", {}),
        "clv_by_tier": log_out.get("clv_by_tier", {}),
        "record_tiers": [], "record_tiers_total": [],
        "tier_trend": bts.get("tier_trend", {}), "tier_summary": bts.get("tier_summary", {}),
        "tier_cuts": bts.get("tier_cuts"),
        "tier_regime": "early" if in_season_weight < 0.35 else "late",
        "situational": bts.get("situational", {}), "debias": bts.get("debias", {}),
        "bias_check": bts.get("bias_check", {}),
        "totals_bias_check": tts.get("bias_check", {}),
        "board_balance": {"spread": balance(picks, "spread"),
                          "total": balance(total_picks, "total")},
        "ratings": ([{"rank": i, "team": NAMES.get(t, t), "rating": round(v, 2)}
                     for i, (t, v) in enumerate(
                         sorted(ratings_now.items(), key=lambda kv: -kv[1]), start=1)]
                    if ratings_now else []),
        "qb_value": {}, "qb_season": None, "notes": notes,
    }

    # season-over-season accuracy trend, spreads and totals combined
    by = {}
    for r in bts.get("by_season", []):
        if r.get("gap") is not None:
            by.setdefault(r["season"], {})["spread"] = r["gap"]
            if r.get("gap_raw") is not None:
                by[r["season"]]["spread_raw"] = r["gap_raw"]
    for r in tts.get("by_season", []):
        if r.get("gap") is not None:
            by.setdefault(r["season"], {})["total"] = r["gap"]
    payload["gap_trend"] = [
        {"season": int(k), "spread": v.get("spread"), "spread_raw": v.get("spread_raw"),
         "total": v.get("total"),
         "all": round(float(np.mean([x for x in (v.get("spread"), v.get("total")) if x is not None])), 2)
                if any(x is not None for x in (v.get("spread"), v.get("total"))) else None}
        for k, v in sorted(by.items())]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT}")
    print(f"  picks {len(payload['picks'])} ({sum(1 for p in payload['picks'] if p['qualified'])} qualified)"
          f" | totals {len(payload['total_picks'])}"
          f" | logged {len(log)} | settled {payload['record'].get('settled', 0)}")


if __name__ == "__main__":
    main()
