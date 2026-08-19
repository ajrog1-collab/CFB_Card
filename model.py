"""
College football power-rating model.

Ported from the v4 Colab notebook. Differences:
  - API key comes from the CFBD_API_KEY environment variable
  - cache lives in the repo (data/cache) instead of Google Drive
  - no interactive prompts, no display calls

Method, in short:
  1. Fit per-season ridge power ratings from scoring margin and net EPA/play.
  2. Regress end-of-season rating on prior-year ratings, returning production,
     and recruiting talent to build a preseason prior.
  3. Blend prior with in-season ratings, weighted by games played per team.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://api.collegefootballdata.com"
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"


# ----------------------------------------------------------------------
# fetching
# ----------------------------------------------------------------------

def _headers() -> dict:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY is not set. Add it as a repository secret named "
            "CFBD_API_KEY (Settings > Secrets and variables > Actions)."
        )
    return {"Authorization": f"Bearer {key}"}


def cfbd_get(endpoint: str, params: dict, force: bool = False) -> pd.DataFrame:
    """Fetch an endpoint, caching to the repo. Returns empty frame on failure."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = endpoint.strip("/").replace("/", "_")
    tag = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    path = CACHE_DIR / f"{slug}__{tag}.parquet"

    if path.exists() and not force:
        try:
            return pd.read_parquet(path)
        except Exception:
            pass  # corrupt cache file, refetch

    try:
        r = requests.get(
            f"{BASE}/{endpoint.strip('/')}", headers=_headers(), params=params, timeout=90
        )
    except Exception as e:
        print(f"    network error on {endpoint} {params}: {e}")
        return pd.DataFrame()

    if r.status_code == 401:
        raise RuntimeError("CFBD returned 401 Unauthorized. Check the CFBD_API_KEY secret.")
    if r.status_code == 429:
        raise RuntimeError("CFBD returned 429. Monthly API call limit reached.")
    if r.status_code >= 400:
        print(f"    HTTP {r.status_code} on {endpoint} {params}")
        return pd.DataFrame()

    try:
        data = r.json()
    except Exception:
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()

    df = pd.json_normalize(data)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        print(f"    could not cache {path.name}: {e}")
    time.sleep(0.35)
    return df


def col(df: pd.DataFrame, *names):
    """First matching column name, else None. Absorbs CFBD field renames."""
    for n in names:
        if n in df.columns:
            return n
    return None


# ----------------------------------------------------------------------
# ratings
# ----------------------------------------------------------------------

def fit_ratings(train, target, ridge=8.0, half_life=None, cap=None, min_games=150):
    """Ridge power ratings: target ~ rating[home] - rating[away] + HFA."""
    sub = train.dropna(subset=[target])
    if len(sub) < min_games:
        return None, 0.0

    teams = sorted(set(sub["home_team"]) | set(sub["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, m = len(sub), len(teams)

    X = np.zeros((n, m + 1))
    rows = np.arange(n)
    X[rows, sub["home_team"].map(idx).to_numpy()] = 1.0
    X[rows, sub["away_team"].map(idx).to_numpy()] = -1.0
    X[:, m] = np.where(sub["neutral"].to_numpy(), 0.0, 1.0)

    y = sub[target].to_numpy(dtype=float)
    if cap is not None:
        y = np.clip(y, -cap, cap)
    if half_life:
        w = np.sqrt(0.5 ** (np.arange(n)[::-1] / half_life))
        X, y = X * w[:, None], y * w

    R = np.eye(m + 1) * ridge
    R[m, m] = 0.0  # never penalize home field
    beta = np.linalg.solve(X.T @ X + R, X.T @ y)

    ratings = {t: float(beta[idx[t]]) for t in teams}
    mu = np.mean([v for t, v in ratings.items() if t != "NON_FBS"])
    return {t: v - mu for t, v in ratings.items()}, float(beta[m])


def blend_ratings(r_margin, r_ppa):
    """Average margin and EPA ratings on a common scale.

    Teams with no EPA rating keep their full margin rating. (Blending them
    against an implicit zero was a bug in an earlier version and it halved
    the NON_FBS baseline.)
    """
    if r_margin is None:
        return None
    if r_ppa is None:
        return dict(r_margin)
    sd_m = np.std([v for k, v in r_margin.items() if k != "NON_FBS"])
    sd_p = np.std([v for k, v in r_ppa.items() if k != "NON_FBS"])
    k = sd_m / max(sd_p, 1e-9)
    out = {}
    for t in set(r_margin) | set(r_ppa):
        if t in r_margin and t in r_ppa:
            out[t] = 0.5 * r_margin[t] + 0.5 * k * r_ppa[t]
        elif t in r_margin:
            out[t] = r_margin[t]
        else:
            out[t] = k * r_ppa[t]
    return out


def rating_diff(frame, ratings, hfa, default=None):
    if ratings is None or not len(frame):
        return np.full(len(frame), np.nan)
    if default is None:
        default = ratings.get("NON_FBS", min(ratings.values()))
    h = frame["home_team"].map(ratings).fillna(default).to_numpy()
    a = frame["away_team"].map(ratings).fillna(default).to_numpy()
    return h - a + np.where(frame["neutral"].to_numpy(), 0.0, hfa)


def blend_prior(prior, in_season, games_played, k=6.0):
    """Per-team weighted blend. w = n/(n+k): 0 at week 1, rising with games."""
    if in_season is None:
        return dict(prior) if prior else None
    if prior is None:
        return dict(in_season)
    out = {}
    for t in set(prior) | set(in_season):
        n = games_played.get(t, 0)
        w = n / (n + k)
        p = prior.get(t, in_season.get(t, 0.0))
        i = in_season.get(t, p)
        out[t] = w * i + (1 - w) * p
    return out


def games_played_counts(frame):
    if not len(frame):
        return {}
    return pd.concat([frame["home_team"], frame["away_team"]]).value_counts().to_dict()


# ----------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------

def build_games(games_raw, fbs, keep_unplayed=False):
    g = games_raw
    sd = col(g, "startDate", "start_date")
    out = pd.DataFrame({
        "game_id":   g[col(g, "id", "gameId", "game_id")],
        "season":    g[col(g, "season", "year")],
        "week":      g[col(g, "week")],
        "home_team": g[col(g, "homeTeam", "home_team")],
        "away_team": g[col(g, "awayTeam", "away_team")],
        "home_pts":  pd.to_numeric(g[col(g, "homePoints", "home_points")], errors="coerce"),
        "away_pts":  pd.to_numeric(g[col(g, "awayPoints", "away_points")], errors="coerce"),
        "neutral":   g[col(g, "neutralSite", "neutral_site")].fillna(False).astype(bool),
        "kickoff":   pd.to_datetime(g[sd], errors="coerce", utc=True) if sd else pd.NaT,
    })
    if not keep_unplayed:
        out = out.dropna(subset=["home_pts", "away_pts"])
    out["margin"] = out["home_pts"] - out["away_pts"]

    # Keep real school names alongside the pooled labels, for display.
    out["home_name"] = out["home_team"]
    out["away_name"] = out["away_team"]

    if fbs:
        fallback = fbs.get(max(fbs), set())
        for side in ("home_team", "away_team"):
            out[side] = [
                tm if tm in fbs.get(sn, fallback) else "NON_FBS"
                for tm, sn in zip(out[side], out["season"])
            ]
        out = out[~((out.home_team == "NON_FBS") & (out.away_team == "NON_FBS"))]
    return out.reset_index(drop=True)


def attach_lines(frame, lines_raw, sign=None):
    if not len(lines_raw):
        frame["spread"] = np.nan
        frame["mkt"] = np.nan
        return frame, (sign if sign is not None else -1)

    lr = lines_raw.copy()
    lc = col(lr, "lines")
    if lc:
        lr = lr.explode(lc).dropna(subset=[lc])
        bk = pd.json_normalize(lr[lc])
        bk.index = lr.index
        lr = pd.concat([lr.drop(columns=[lc]), bk], axis=1)

    gid = col(lr, "id", "gameId", "game_id")
    if "provider" in lr.columns:
        lr["_r"] = (lr["provider"].astype(str).str.lower() != "consensus").astype(int)
        lr = lr.sort_values("_r")
    lr = lr.groupby(gid, as_index=False).first()

    ln = pd.DataFrame({
        "game_id": lr[gid],
        "spread": pd.to_numeric(lr.get("spread"), errors="coerce"),
        "total": pd.to_numeric(lr.get("overUnder", lr.get("over_under")), errors="coerce"),
    })
    frame = frame.merge(ln, on="game_id", how="left")

    if sign is None:
        ck = frame.dropna(subset=["spread", "margin"])
        sign = -1 if len(ck) > 50 and ck["spread"].corr(ck["margin"]) < 0 else 1
    frame["mkt"] = sign * frame["spread"]
    return frame, sign


def attach_epa(d, adv_raw):
    """Add ppa_margin (home net EPA/play minus away). Returns (frame, using_epa)."""
    d = d.copy()
    d["ppa_margin"] = np.nan
    if not len(adv_raw):
        return d, False

    ac = adv_raw.copy()
    gid = col(ac, "gameId", "game_id", "id")
    tm = col(ac, "team", "school")
    off = col(ac, "offense.ppa", "offense_ppa", "offense.overall.ppa")
    dfn = col(ac, "defense.ppa", "defense_ppa", "defense.overall.ppa")
    if not all([gid, tm, off, dfn]):
        return d, False

    ac["net_ppa"] = pd.to_numeric(ac[off], errors="coerce") - pd.to_numeric(ac[dfn], errors="coerce")
    ac = ac[[gid, tm, "net_ppa"]].rename(columns={gid: "game_id", tm: "team"}).dropna()

    gm = d[["game_id", "home_name", "away_name"]]
    h = gm.merge(ac, left_on=["game_id", "home_name"], right_on=["game_id", "team"])[["game_id", "net_ppa"]]
    a = gm.merge(ac, left_on=["game_id", "away_name"], right_on=["game_id", "team"])[["game_id", "net_ppa"]]
    pm = h.merge(a, on="game_id", suffixes=("_h", "_a"))
    pm["ppa_margin"] = pm["net_ppa_h"] - pm["net_ppa_a"]

    d = d.drop(columns=["ppa_margin"]).merge(pm[["game_id", "ppa_margin"]], on="game_id", how="left")
    return d, bool(d["ppa_margin"].notna().sum() > 1000)


# ----------------------------------------------------------------------
# preseason prior
# ----------------------------------------------------------------------

def season_ratings(d, years, use_epa):
    """Per-season ratings, each season fit in isolation."""
    R, H = {}, {}
    for yr in years:
        sub = d[d.season == yr]
        r_m, hfa = fit_ratings(sub, "margin", ridge=12.0, cap=35.0)
        if r_m is None:
            continue
        r_p = None
        if use_epa:
            r_p, _ = fit_ratings(sub, "ppa_margin", ridge=0.9, cap=1.5)
        R[yr] = blend_ratings(r_m, r_p)
        H[yr] = hfa
    return R, H


def talent_composite(recruit, years, window=4):
    """4-year rolling recruiting average, z-scored within season."""
    out = {}
    if not len(recruit):
        return out
    tcol = col(recruit, "team", "school")
    pcol = col(recruit, "points", "point")
    if not (tcol and pcol):
        return out
    rc = recruit.copy()
    rc["pts"] = pd.to_numeric(rc[pcol], errors="coerce")
    for yr in years:
        win = rc[(rc._year >= yr - window) & (rc._year < yr)]
        if not len(win):
            continue
        avg = win.groupby(tcol)["pts"].mean()
        z = (avg - avg.mean()) / max(avg.std(), 1e-9)
        out[yr] = z.to_dict()
    return out


def returning_production(returning):
    """Share of last year's production coming back, z-scored within season."""
    out, field = {}, None
    if not len(returning):
        return out, field
    tcol = col(returning, "team", "school")
    field = col(returning, "percentPPA", "percent_ppa", "totalPPA", "total_ppa", "usage")
    if not (tcol and field):
        return out, field
    rp = returning.copy()
    rp["val"] = pd.to_numeric(rp[field], errors="coerce")
    for yr, grp in rp.groupby("_year"):
        g = grp.dropna(subset=["val"])
        if not len(g):
            continue
        z = (g.set_index(tcol)["val"] - g["val"].mean()) / max(g["val"].std(), 1e-9)
        out[int(yr)] = z.to_dict()
    return out, field


class PriorModel:
    """Regression from prior-year ratings + returning production + talent
    onto end-of-season rating."""

    FEATURES = ["r_prev", "r_prev2", "ret", "talent"]

    def __init__(self, season_r, ret, talent):
        self.season_r = season_r
        self.ret = ret
        self.talent = talent
        self.feats = []
        self.weights = None
        self.r2 = float("nan")
        self.n_train = 0
        self.medians = {}
        self._fit()

    def _rows(self):
        rows = []
        for yr in sorted(self.season_r):
            if (yr - 1) not in self.season_r or (yr - 2) not in self.season_r:
                continue
            for team, target in self.season_r[yr].items():
                if team == "NON_FBS":
                    continue
                rows.append({
                    "season": yr, "team": team, "target": target,
                    "r_prev": self.season_r[yr - 1].get(team, np.nan),
                    "r_prev2": self.season_r[yr - 2].get(team, np.nan),
                    "ret": self.ret.get(yr, {}).get(team, np.nan),
                    "talent": self.talent.get(yr, {}).get(team, np.nan),
                })
        return pd.DataFrame(rows)

    def _fit(self):
        P = self._rows()
        if not len(P):
            return
        self.feats = [f for f in self.FEATURES if f in P and P[f].notna().sum() > 200]
        if not self.feats:
            return
        for f in self.feats:
            self.medians[f] = float(P[f].median())
            P[f] = P[f].fillna(self.medians[f])
        P = P.dropna(subset=["target"])

        A = np.column_stack([P[f].to_numpy(dtype=float) for f in self.feats] + [np.ones(len(P))])
        y = P["target"].to_numpy(dtype=float)
        self.weights, *_ = np.linalg.lstsq(A, y, rcond=None)
        fitted = A @ self.weights
        self.r2 = float(1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
        self.n_train = int(len(P))

    def coefficients(self):
        if self.weights is None:
            return {}
        out = {f: float(w) for f, w in zip(self.feats, self.weights)}
        out["intercept"] = float(self.weights[-1])
        return out

    def preseason(self, season):
        """Predicted rating for every team before any games are played."""
        if self.weights is None:
            return None
        if (season - 1) not in self.season_r or (season - 2) not in self.season_r:
            return None
        teams = set(self.season_r[season - 1]) | set(self.talent.get(season, {}))
        teams.discard("NON_FBS")
        if not teams:
            return None

        out = {}
        for t in teams:
            vals = {
                "r_prev": self.season_r[season - 1].get(t, self.medians.get("r_prev", 0.0)),
                "r_prev2": self.season_r[season - 2].get(t, self.medians.get("r_prev2", 0.0)),
                "ret": self.ret.get(season, {}).get(t, self.medians.get("ret", 0.0)),
                "talent": self.talent.get(season, {}).get(t, self.medians.get("talent", 0.0)),
            }
            x = np.array([vals[f] for f in self.feats] + [1.0])
            out[t] = float(x @ self.weights)

        mu = np.mean(list(out.values()))
        out = {t: v - mu for t, v in out.items()}
        out["NON_FBS"] = self.season_r[season - 1].get("NON_FBS", -22.0)
        return out
