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


def cfbd_get(endpoint: str, params: dict, force: bool = False,
             required: bool = True) -> pd.DataFrame:
    """Fetch an endpoint, caching to the repo. Returns empty frame on failure.

    `required=False` marks an endpoint the model can live without (weather,
    advanced stats, box scores, SP+). Several of those sit behind paid CFBD
    tiers, so a 401 there means "not included in your plan" rather than "bad
    key" and must not abort the whole run.
    """
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

    if r.status_code in (401, 403):
        if required:
            raise RuntimeError(
                f"CFBD returned {r.status_code} on {endpoint}, which this model requires. "
                "Check the CFBD_API_KEY secret."
            )
        print(f"    {endpoint} not available on this API tier ({r.status_code}) — skipping")
        return pd.DataFrame()
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
    vid = col(g, "venueId", "venue_id")
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
        "venue_id":  pd.to_numeric(g[vid], errors="coerce") if vid else np.nan,
    })
    if not keep_unplayed:
        out = out.dropna(subset=["home_pts", "away_pts"])
    out["margin"] = out["home_pts"] - out["away_pts"]
    out["actual_total"] = out["home_pts"] + out["away_pts"]

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
    if "total" in frame.columns:
        frame["mkt_total"] = frame["total"]
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
# turnover adjustment
# ----------------------------------------------------------------------

def parse_team_stats(raw):
    """Flatten /games/teams into one row per game-team with turnover counts.

    Response shape is nested: game -> teams[] -> stats[] of {category, stat}.
    Returns DataFrame [game_id, team, home_away, turnovers] or empty.
    """
    if not len(raw):
        return pd.DataFrame()

    gid = col(raw, "id", "gameId", "game_id")
    tcol = col(raw, "teams")
    if not (gid and tcol):
        return pd.DataFrame()

    df = raw[[gid, tcol]].rename(columns={gid: "game_id"}).explode(tcol).dropna(subset=[tcol])
    if not len(df):
        return pd.DataFrame()

    teams = pd.json_normalize(df[tcol])
    teams.index = df.index
    df = pd.concat([df.drop(columns=[tcol]), teams], axis=1)

    school = col(df, "school", "team")
    ha = col(df, "homeAway", "home_away")
    scol = col(df, "stats")
    if not (school and scol):
        return pd.DataFrame()

    st = df[["game_id", school, scol]].rename(columns={school: "team"})
    st = st.explode(scol).dropna(subset=[scol])
    if not len(st):
        return pd.DataFrame()

    sn = pd.json_normalize(st[scol])
    sn.index = st.index
    st = pd.concat([st.drop(columns=[scol]), sn], axis=1)

    ccol = col(st, "category")
    vcol = col(st, "stat", "value")
    if not (ccol and vcol):
        return pd.DataFrame()

    to = st[st[ccol].astype(str).str.lower() == "turnovers"].copy()
    if not len(to):
        return pd.DataFrame()
    to["turnovers"] = pd.to_numeric(to[vcol], errors="coerce")
    to = to.dropna(subset=["turnovers"])[["game_id", "team", "turnovers"]]

    out = to.groupby(["game_id", "team"], as_index=False)["turnovers"].first()
    if ha:
        side = df[["game_id", school, ha]].rename(columns={school: "team", ha: "home_away"})
        out = out.merge(side, on=["game_id", "team"], how="left")
    return out


def attach_turnovers(d, to_df):
    """Add to_margin: home turnovers lost minus away turnovers lost.

    Negative to_margin means the home team lost the turnover battle.
    """
    d = d.copy()
    d["to_margin"] = np.nan
    if not len(to_df):
        return d, False

    gm = d[["game_id", "home_name", "away_name"]]
    h = gm.merge(to_df, left_on=["game_id", "home_name"], right_on=["game_id", "team"])
    a = gm.merge(to_df, left_on=["game_id", "away_name"], right_on=["game_id", "team"])
    both = h[["game_id", "turnovers"]].merge(
        a[["game_id", "turnovers"]], on="game_id", suffixes=("_h", "_a"))
    # away giveaways minus home giveaways: positive helps the home team
    both["to_margin"] = both["turnovers_a"] - both["turnovers_h"]

    d = d.drop(columns=["to_margin"]).merge(
        both[["game_id", "to_margin"]], on="game_id", how="left")
    return d, bool(d["to_margin"].notna().sum() > 1000)


def fit_turnover_points(d):
    """Points of scoring margin associated with one net turnover.

    Regresses margin on turnover margin. Historically lands near 4-5.
    """
    sub = d.dropna(subset=["margin", "to_margin"])
    if len(sub) < 500:
        return None
    x = sub["to_margin"].to_numpy(dtype=float)
    y = sub["margin"].to_numpy(dtype=float)
    A = np.column_stack([x, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[0])


def add_adjusted_margin(d, shrink=0.7):
    """Remove the luck portion of turnover margin from scoring margin.

    Fumble recovery is close to a coin flip and interception rate is far less
    stable than it looks, so most of turnover margin does not repeat. Some of
    it is skill (pressure, ball security), which is why we remove only
    `shrink` of the fitted effect rather than all of it.
    """
    d = d.copy()
    k = fit_turnover_points(d)
    if k is None:
        d["adj_margin"] = d["margin"]
        return d, None
    d["adj_margin"] = np.where(
        d["to_margin"].notna(),
        d["margin"] - shrink * k * d["to_margin"].fillna(0.0),
        d["margin"],
    )
    return d, k


# ----------------------------------------------------------------------
# published ratings
# ----------------------------------------------------------------------

def parse_sp(raw):
    """Team -> overall SP+ rating for one season."""
    if not len(raw):
        return {}
    tcol = col(raw, "team", "school")
    rcol = col(raw, "rating", "overall.rating", "overallRating")
    if not (tcol and rcol):
        return {}
    sub = raw[[tcol, rcol]].copy()
    sub[rcol] = pd.to_numeric(sub[rcol], errors="coerce")
    sub = sub.dropna()
    sub = sub[sub[tcol].astype(str).str.lower() != "nationalaverages"]
    return dict(zip(sub[tcol], sub[rcol]))


# ----------------------------------------------------------------------
# situational adjustments
# ----------------------------------------------------------------------

# Curated list of annual rivalries. Names follow CFBD school naming. A name
# mismatch simply means the flag never fires for that pair, which is safe.
RIVALRIES = [
    ("Ohio State", "Michigan"), ("Alabama", "Auburn"), ("Army", "Navy"),
    ("Oklahoma", "Oklahoma State"), ("Texas", "Texas A&M"), ("Michigan", "Michigan State"),
    ("Florida", "Florida State"), ("Georgia", "Georgia Tech"), ("Clemson", "South Carolina"),
    ("USC", "UCLA"), ("Oregon", "Oregon State"), ("Washington", "Washington State"),
    ("California", "Stanford"), ("Utah", "BYU"), ("Kansas", "Kansas State"),
    ("Iowa", "Iowa State"), ("Minnesota", "Wisconsin"), ("Indiana", "Purdue"),
    ("Illinois", "Northwestern"), ("Pittsburgh", "West Virginia"),
    ("Virginia", "Virginia Tech"), ("North Carolina", "Duke"),
    ("North Carolina", "NC State"), ("Wake Forest", "Duke"),
    ("Kentucky", "Louisville"), ("Tennessee", "Vanderbilt"),
    ("Mississippi", "Mississippi State"), ("LSU", "Arkansas"),
    ("Missouri", "Kansas"), ("Nebraska", "Iowa"), ("Colorado", "Colorado State"),
    ("Arizona", "Arizona State"), ("Texas Tech", "Baylor"), ("TCU", "SMU"),
    ("Houston", "Rice"), ("Cincinnati", "Miami (OH)"), ("Miami (OH)", "Ohio"),
    ("Toledo", "Bowling Green"), ("Air Force", "Army"), ("Air Force", "Navy"),
    ("Boise State", "Fresno State"), ("Nevada", "UNLV"),
    ("Notre Dame", "USC"), ("Notre Dame", "Navy"), ("Marshall", "Ohio"),
    ("Louisiana", "Louisiana Monroe"), ("Southern Mississippi", "Memphis"),
    ("Utah State", "Wyoming"), ("San Diego State", "San Jose State"),
    ("East Carolina", "Marshall"), ("Auburn", "Georgia"), ("Florida", "Georgia"),
    ("Tennessee", "Alabama"), ("Penn State", "Pittsburgh"), ("Maryland", "Penn State"),
    ("Boston College", "Syracuse"), ("Connecticut", "UMass"),
]
RIVALRY_SET = frozenset(frozenset(p) for p in RIVALRIES)


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles. Vectorized."""
    R = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def parse_venues(raw):
    """venue_id -> {lat, lon, tz}."""
    if not len(raw):
        return {}
    idc = col(raw, "id", "venueId")
    latc = col(raw, "latitude", "location.x", "location.latitude")
    lonc = col(raw, "longitude", "location.y", "location.longitude")
    tzc = col(raw, "timezone", "timeZone")
    if not (idc and latc and lonc):
        return {}
    out = {}
    for _, r in raw.iterrows():
        try:
            vid = int(r[idc])
            lat, lon = float(r[latc]), float(r[lonc])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(lat) and np.isfinite(lon)) or (lat == 0 and lon == 0):
            continue
        dome_c = None
        for cand in ("dome", "isDome", "grass"):
            if cand in raw.columns and cand != "grass":
                dome_c = cand
                break
        out[vid] = {"lat": lat, "lon": lon,
                    "tz": (str(r[tzc]) if tzc and pd.notna(r.get(tzc)) else None),
                    "dome": bool(r[dome_c]) if dome_c and pd.notna(r.get(dome_c)) else False}
    return out


def tz_offset_hours(tzname, when):
    """UTC offset in hours for a timezone at a given moment. Falls back to None."""
    if not tzname:
        return None
    try:
        from zoneinfo import ZoneInfo
        if pd.isna(when):
            return None
        off = when.astimezone(ZoneInfo(tzname)).utcoffset()
        return off.total_seconds() / 3600.0 if off else None
    except Exception:
        return None


def infer_home_venues(d):
    """Each team's usual home venue, from the games where they hosted."""
    if "venue_id" not in d.columns:
        return {}
    home = d[d["neutral"] == False].dropna(subset=["venue_id"])
    if not len(home):
        return {}
    mode = (home.groupby("home_name")["venue_id"]
                .agg(lambda s: s.value_counts().idxmax()))
    return mode.to_dict()


def add_situational(d, venues, home_venue):
    """Travel, time-zone/body-clock, rest, and rivalry columns.

    All of these are game-level context rather than team strength, so they
    enter the final linear calibration alongside the rating differential
    instead of changing the ratings themselves.
    """
    d = d.copy()
    n = len(d)
    for c in ("travel_diff_k", "east_body", "rest_diff", "rivalry"):
        d[c] = 0.0
    if not n:
        return d

    # ---- rivalry ----
    d["rivalry"] = [
        1.0 if frozenset((h, a)) in RIVALRY_SET else 0.0
        for h, a in zip(d["home_name"], d["away_name"])
    ]

    # ---- rest differential ----
    if "kickoff" in d.columns:
        ko = pd.to_datetime(d["kickoff"], errors="coerce", utc=True)
        last = {}
        rest_h, rest_a = np.zeros(n), np.zeros(n)
        order = np.argsort(ko.values.astype("datetime64[ns]"))
        for pos in order:
            t = ko.iloc[pos]
            if pd.isna(t):
                continue
            for side, arr in (("home_name", rest_h), ("away_name", rest_a)):
                team = d[side].iloc[pos]
                prev = last.get(team)
                arr[pos] = 7.0 if prev is None else min((t - prev).total_seconds() / 86400.0, 21.0)
            for side in ("home_name", "away_name"):
                last[d[side].iloc[pos]] = t
        d["rest_diff"] = np.clip(rest_h - rest_a, -10, 10)

    # ---- travel and body clock ----
    if venues and home_venue and "venue_id" in d.columns:
        def coords(team):
            v = home_venue.get(team)
            return venues.get(v) if v is not None else None

        miles_h, miles_a, east_body = np.zeros(n), np.zeros(n), np.zeros(n)
        ko = pd.to_datetime(d["kickoff"], errors="coerce", utc=True) \
            if "kickoff" in d.columns else pd.Series([pd.NaT] * n)

        for i in range(n):
            vid = d["venue_id"].iloc[i]
            site = venues.get(int(vid)) if pd.notna(vid) else None
            if site is None:
                continue
            for team, arr in ((d["home_name"].iloc[i], miles_h),
                              (d["away_name"].iloc[i], miles_a)):
                hv = coords(team)
                if hv:
                    arr[i] = float(haversine(hv["lat"], hv["lon"], site["lat"], site["lon"]))

            # body-clock penalty: eastward travel into an early kickoff
            av = coords(d["away_name"].iloc[i])
            t = ko.iloc[i]
            if av and site and pd.notna(t):
                off_site = tz_offset_hours(site["tz"], t)
                off_away = tz_offset_hours(av["tz"], t)
                if off_site is not None and off_away is not None:
                    shift = off_site - off_away          # >0 means travelled east
                    # wrap into 0-23; a bare sum goes negative for late kicks
                    local_hour = (t.hour + t.minute / 60.0 + off_site) % 24
                    body_hour = (local_hour - shift) % 24
                    if shift > 0 and body_hour <= 11:
                        east_body[i] = float(shift)

        d["travel_diff_k"] = (miles_a - miles_h) / 1000.0
        d["east_body"] = east_body

    return d


SITUATIONAL = ["travel_diff_k", "east_body", "rest_diff"]


def situational_matrix(frame, rating_col):
    """Feature columns for the calibration step, including rivalry compression.

    Rivalry enters as an interaction with the rating differential: rivalry games
    tend to play closer than ratings suggest, so we expect a negative weight
    that shrinks the prediction toward a pick'em rather than shifting it.
    """
    cols, names = [], []
    for c in SITUATIONAL:
        if c in frame.columns:
            cols.append(frame[c].fillna(0.0).to_numpy(dtype=float))
            names.append(c)
    if "rivalry" in frame.columns:
        cols.append(frame["rivalry"].fillna(0.0).to_numpy(dtype=float) * rating_col)
        names.append("rivalry_compress")
    return (np.column_stack(cols) if cols else np.zeros((len(frame), 0))), names


# Direction and plausible magnitude for each situational effect, taken from
# published work rather than fitted. A least-squares fit of weak correlated
# features next to a dominant rating differential will happily return the wrong
# sign or an absurd magnitude, so anything outside these bounds is zeroed rather
# than trusted.
SIT_BOUNDS = {
    "travel_diff_k":    (-3.0, 0.0),    # travelling further should not help you
    "east_body":        (-2.5, 0.0),    # eastward body-clock shift should not help
    "rest_diff":        (0.0, 1.0),     # extra rest should not hurt
    "rivalry_compress": (-0.35, 0.0),   # compression, at most a third of the spread
}
SIT_MIN_ROWS = 300   # non-zero rows a feature needs before it is fit at all


def fit_calibration(rating_col, S, names, y, ridge=25.0):
    """Fit outcome ~ rating + situational, penalizing the situational block.

    Returns (weights, report) with weights aligned to
    [rating, *S columns, intercept]. Features that are too sparse to estimate,
    or that come back with an implausible value, are forced to zero and listed
    in report["_excluded"] with the reason.
    """
    rating_col = np.asarray(rating_col, dtype=float)
    y = np.asarray(y, dtype=float)
    k = S.shape[1] if S.size else 0

    ok = ~np.isnan(rating_col) & ~np.isnan(y)
    if S.size:
        ok = ok & ~np.isnan(S).any(axis=1)
    n = int(ok.sum())
    if n < 300:
        return np.array([1.0] + [0.0] * k + [0.0]), {}

    keep, dropped = [], {}
    for j, nm in enumerate(names):
        nz = int((np.abs(S[ok, j]) > 1e-9).sum())
        if nz < SIT_MIN_ROWS:
            dropped[nm] = f"only {nz} games with the effect present, need {SIT_MIN_ROWS}"
        else:
            keep.append(j)

    Sk = S[:, keep] if keep else np.zeros((len(rating_col), 0))
    A = np.column_stack([rating_col[ok], Sk[ok], np.ones(n)])

    # ridge on the situational block only; rating scale and intercept stay free
    R = np.zeros((A.shape[1], A.shape[1]))
    for i in range(1, 1 + len(keep)):
        R[i, i] = ridge
    try:
        beta = np.linalg.solve(A.T @ A + R, A.T @ y[ok])
    except np.linalg.LinAlgError:
        return np.array([1.0] + [0.0] * k + [0.0]), dropped

    w = np.zeros(k + 2)
    w[0] = float(beta[0])
    w[-1] = float(beta[-1])
    report = {}
    for slot, j in enumerate(keep):
        nm = names[j]
        val = float(beta[1 + slot])
        lo, hi = SIT_BOUNDS.get(nm, (-np.inf, np.inf))
        if val < lo or val > hi:
            dropped[nm] = f"fitted {val:+.3f}, outside plausible range [{lo}, {hi}]"
            continue
        w[1 + j] = val
        report[nm] = round(val, 3)

    report["rating_scale"] = round(float(beta[0]), 3)
    if dropped:
        report["_excluded"] = dropped
    return w, report


# ----------------------------------------------------------------------
# totals: offense / defense points ratings
# ----------------------------------------------------------------------

def fit_points_ratings(train, ridge=14.0, half_life=None, cap=56.0, min_games=150):
    """Rate offense and defense separately on points scored.

    Each game contributes two rows:
        points = base + offense[team] - defense[opponent] + hfa_off * is_home

    A higher defense rating means fewer points allowed. Returns
    (offense, defense, hfa_off, base) or (None, None, 0, 0).
    """
    sub = train.dropna(subset=["home_pts", "away_pts"])
    if len(sub) < min_games:
        return None, None, 0.0, 0.0

    teams = sorted(set(sub["home_team"]) | set(sub["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    m = len(teams)
    n = 2 * len(sub)

    # columns: [offense 0..m-1][defense m..2m-1][is_home][intercept]
    X = np.zeros((n, 2 * m + 2))
    y = np.zeros(n)

    h_off = sub["home_team"].map(idx).to_numpy()
    a_off = sub["away_team"].map(idx).to_numpy()
    rows_h = np.arange(len(sub))
    rows_a = rows_h + len(sub)

    X[rows_h, h_off] = 1.0
    X[rows_h, m + a_off] = -1.0
    X[rows_h, 2 * m] = np.where(sub["neutral"].to_numpy(), 0.0, 1.0)
    X[rows_h, 2 * m + 1] = 1.0
    y[rows_h] = np.clip(sub["home_pts"].to_numpy(dtype=float), 0, cap)

    X[rows_a, a_off] = 1.0
    X[rows_a, m + h_off] = -1.0
    X[rows_a, 2 * m + 1] = 1.0
    y[rows_a] = np.clip(sub["away_pts"].to_numpy(dtype=float), 0, cap)

    if half_life:
        age = np.concatenate([np.arange(len(sub))[::-1]] * 2)
        w = np.sqrt(0.5 ** (age / half_life))
        X, y = X * w[:, None], y * w

    R = np.eye(2 * m + 2) * ridge
    R[2 * m, 2 * m] = 0.0        # home-field scoring bump unpenalized
    R[2 * m + 1, 2 * m + 1] = 0.0  # intercept unpenalized

    try:
        beta = np.linalg.solve(X.T @ X + R, X.T @ y)
    except np.linalg.LinAlgError:
        return None, None, 0.0, 0.0

    off = {t: float(beta[idx[t]]) for t in teams}
    dfn = {t: float(beta[m + idx[t]]) for t in teams}
    hfa_off = float(beta[2 * m])
    base = float(beta[2 * m + 1])

    # center both so ratings read as points above/below average
    mo = np.mean([v for t, v in off.items() if t != "NON_FBS"])
    md = np.mean([v for t, v in dfn.items() if t != "NON_FBS"])
    off = {t: v - mo for t, v in off.items()}
    dfn = {t: v - md for t, v in dfn.items()}
    base = base + mo - md
    return off, dfn, hfa_off, base


def predict_total(frame, off, dfn, hfa_off, base):
    """Expected combined points."""
    if off is None or not len(frame):
        return np.full(len(frame), np.nan)
    do = min(off.values()) if off else 0.0
    dd = min(dfn.values()) if dfn else 0.0
    oh = frame["home_team"].map(off).fillna(do).to_numpy()
    oa = frame["away_team"].map(off).fillna(do).to_numpy()
    dh = frame["home_team"].map(dfn).fillna(dd).to_numpy()
    da = frame["away_team"].map(dfn).fillna(dd).to_numpy()
    home_pts = base + oh - da + np.where(frame["neutral"].to_numpy(), 0.0, hfa_off)
    away_pts = base + oa - dh
    return home_pts + away_pts


# ----------------------------------------------------------------------
# weather via Open-Meteo (free, no key required)
# ----------------------------------------------------------------------

OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
OM_DAILY = "wind_speed_10m_max,temperature_2m_mean,precipitation_sum"


def _om_get(url, params, label, timeout=15):
    """One Open-Meteo call. Returns parsed daily dict or None.

    Short timeout on purpose: Open-Meteo throttles shared CI addresses, and a
    slow failure is far worse than a fast one when there are 130 venues to get
    through inside a job time limit.
    """
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code >= 400:
            print(f"    open-meteo {label}: HTTP {r.status_code}")
            return None
        return r.json().get("daily")
    except Exception as e:
        print(f"    open-meteo {label} failed: {e}")
        return None


def fetch_venue_weather(venues, start_date, end_date, budget=40, force=False,
                        seconds_budget=300, abort_after_failures=6,
                        venue_order=None):
    """Daily weather per venue for the whole date range.

    One archive call per venue covers every season at once, so the whole history
    costs ~130 calls rather than one per game. Domes are skipped entirely.

    `budget` caps how many NEW venues are fetched per run. Cached venues are
    always loaded and cost nothing. This keeps every run comfortably inside the
    workflow timeout: the backfill completes over several runs, and each run
    commits what it got, so progress is never lost to a timeout.

    Returns a DataFrame [venue_id, date, wind, temp, precip].
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    fetched = 0
    pending = 0
    failures = 0
    consecutive_failures = 0
    started = time.time()
    gave_up = False

    # CFBD lists every venue it has ever known — 800+, most of them FCS or
    # defunct. Only fetch the ones that actually host games here, busiest first,
    # so coverage grows as fast as possible per call spent.
    if venue_order:
        order = [(vid, venues[vid]) for vid in venue_order if vid in venues]
    else:
        order = sorted(venues.items())

    for vid, v in order:
        if v.get("dome"):
            continue
        path = CACHE_DIR / f"om_{vid}.parquet"
        if path.exists() and not force:
            try:
                frames.append(pd.read_parquet(path))
                continue
            except Exception:
                pass

        # stop fetching on any of: call budget, wall-clock budget, or a run of
        # failures that means the host is refusing us right now
        if gave_up or fetched >= budget or (time.time() - started) > seconds_budget:
            pending += 1
            continue
        fetched += 1

        daily = _om_get(OM_ARCHIVE, {
            "latitude": round(v["lat"], 4), "longitude": round(v["lon"], 4),
            "start_date": start_date, "end_date": end_date,
            "daily": OM_DAILY, "wind_speed_unit": "mph",
            "temperature_unit": "fahrenheit", "timezone": "UTC",
        }, f"archive venue {vid}")
        if not daily or not daily.get("time"):
            failures += 1
            consecutive_failures += 1
            if consecutive_failures >= abort_after_failures:
                gave_up = True
                print(f"    weather: {consecutive_failures} failures in a row — "
                      f"Open-Meteo is refusing this runner, skipping the rest this run")
            continue
        consecutive_failures = 0

        df = pd.DataFrame({
            "venue_id": vid,
            "date": pd.to_datetime(daily["time"], errors="coerce").date,
            "wind": pd.to_numeric(daily.get("wind_speed_10m_max"), errors="coerce"),
            "temp": pd.to_numeric(daily.get("temperature_2m_mean"), errors="coerce"),
            "precip": pd.to_numeric(daily.get("precipitation_sum"), errors="coerce"),
        }).dropna(subset=["date"])
        try:
            df.to_parquet(path, index=False)
        except Exception:
            pass
        frames.append(df)
        time.sleep(0.15)

    got = fetched - failures
    if fetched:
        print(f"    weather: {got} of {fetched} attempts succeeded "
              f"({time.time() - started:.0f}s)")
    if pending:
        print(f"    weather: {pending} venues still to backfill — they will fill in "
              f"on the next scheduled run")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_venue_forecast(venues, venue_ids, days=16, seconds_budget=90,
                         abort_after_failures=4, max_venues=60):
    """Forecast for venues with games coming up. Never cached, so it needs the
    same guards as the archive fetch: a wall-clock budget and an early exit when
    the host is refusing us."""
    frames = []
    failures = 0
    started = time.time()
    attempted = 0

    for vid in sorted(set(int(v) for v in venue_ids if pd.notna(v))):
        v = venues.get(vid)
        if not v or v.get("dome"):
            continue
        if (attempted >= max_venues
                or failures >= abort_after_failures
                or (time.time() - started) > seconds_budget):
            break
        attempted += 1
        daily = _om_get(OM_FORECAST, {
            "latitude": round(v["lat"], 4), "longitude": round(v["lon"], 4),
            "daily": OM_DAILY, "forecast_days": days, "wind_speed_unit": "mph",
            "temperature_unit": "fahrenheit", "timezone": "UTC",
        }, f"forecast venue {vid}")
        if not daily or not daily.get("time"):
            failures += 1
            continue
        failures = 0
        frames.append(pd.DataFrame({
            "venue_id": vid,
            "date": pd.to_datetime(daily["time"], errors="coerce").date,
            "wind": pd.to_numeric(daily.get("wind_speed_10m_max"), errors="coerce"),
            "temp": pd.to_numeric(daily.get("temperature_2m_mean"), errors="coerce"),
            "precip": pd.to_numeric(daily.get("precipitation_sum"), errors="coerce"),
        }).dropna(subset=["date"]))
        time.sleep(0.15)

    if attempted:
        print(f"    forecast: {len(frames)} of {attempted} venues "
              f"({time.time() - started:.0f}s)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_venue_weather(d, wx_daily, venues):
    """Join daily venue weather onto games by venue and kickoff date.

    Domes get calm-weather values rather than missing ones: a closed roof really
    does mean no wind, so treating it as unknown would throw away information.
    """
    d = d.copy()
    for c in ("wind", "temp", "precip"):
        if c not in d.columns:
            d[c] = np.nan

    if len(wx_daily) and "venue_id" in d.columns and "kickoff" in d.columns:
        j = d[["game_id", "venue_id", "kickoff"]].copy()
        j["date"] = pd.to_datetime(j["kickoff"], errors="coerce", utc=True).dt.date
        j["venue_id"] = pd.to_numeric(j["venue_id"], errors="coerce")
        w = wx_daily.copy()
        w["venue_id"] = pd.to_numeric(w["venue_id"], errors="coerce")
        merged = j.merge(w, on=["venue_id", "date"], how="left")[
            ["game_id", "wind", "temp", "precip"]]
        d = d.drop(columns=["wind", "temp", "precip"], errors="ignore").merge(
            merged, on="game_id", how="left")

    # indoor games: calm and mild by definition
    if venues and "venue_id" in d.columns:
        domes = {vid for vid, v in venues.items() if v.get("dome")}
        if domes:
            is_dome = d["venue_id"].isin(domes)
            d.loc[is_dome, "wind"] = 0.0
            d.loc[is_dome, "temp"] = d.loc[is_dome, "temp"].fillna(70.0)
            d.loc[is_dome, "precip"] = 0.0
            d["indoor"] = is_dome.astype(float)
    if "indoor" not in d.columns:
        d["indoor"] = 0.0

    d["wind_excess"] = np.clip(d["wind"].fillna(8.0) - 10.0, 0, 30)
    d["cold"] = np.clip(40.0 - d["temp"].fillna(60.0), 0, 60) / 10.0
    d["precip_flag"] = (d["precip"].fillna(0.0) > 0.05).astype(float)
    d["wx_known"] = d["wind"].notna().astype(float)
    covered = int(d["wind"].notna().sum())
    return d, bool(covered > 500)


# ----------------------------------------------------------------------
# weather (totals only)
# ----------------------------------------------------------------------

def parse_weather(raw):
    """game_id -> {wind, temp, precip}. Empty if the endpoint is unavailable."""
    if not len(raw):
        return pd.DataFrame()
    gid = col(raw, "id", "gameId", "game_id")
    wind = col(raw, "windSpeed", "wind_speed")
    temp = col(raw, "temperature", "temp")
    prec = col(raw, "precipitation", "precip")
    if not gid:
        return pd.DataFrame()
    out = pd.DataFrame({"game_id": pd.to_numeric(raw[gid], errors="coerce")})
    out["wind"] = pd.to_numeric(raw[wind], errors="coerce") if wind else np.nan
    out["temp"] = pd.to_numeric(raw[temp], errors="coerce") if temp else np.nan
    out["precip"] = pd.to_numeric(raw[prec], errors="coerce") if prec else np.nan
    return out.dropna(subset=["game_id"]).drop_duplicates("game_id")


def attach_weather(d, wx):
    """Add wind_excess, cold, and precip columns for the totals model.

    Wind enters as excess over 10 mph: light wind does nothing measurable,
    and the effect on scoring shows up in the tail.
    """
    d = d.copy()
    for c in ("wind_excess", "cold", "precip_flag"):
        d[c] = 0.0
    if not len(wx):
        return d, False

    d = d.drop(columns=["wind_excess", "cold", "precip_flag"]).merge(
        wx, on="game_id", how="left")
    d["wind_excess"] = np.clip(d["wind"].fillna(8.0) - 10.0, 0, 30)
    d["cold"] = np.clip(40.0 - d["temp"].fillna(60.0), 0, 60) / 10.0
    d["precip_flag"] = (d["precip"].fillna(0.0) > 0.05).astype(float)
    return d, bool(d["wind"].notna().sum() > 500)


TOTALS_SITUATIONAL = ["wind_excess", "cold", "precip_flag", "rest_diff"]


def totals_matrix(frame):
    """Context columns for the totals calibration."""
    cols, names = [], []
    for c in TOTALS_SITUATIONAL:
        if c in frame.columns:
            cols.append(frame[c].fillna(0.0).to_numpy(dtype=float))
            names.append(c)
    return (np.column_stack(cols) if cols else np.zeros((len(frame), 0))), names


# ----------------------------------------------------------------------
# confidence: how much to trust a given prediction
# ----------------------------------------------------------------------
#
# Edge size alone tells you nothing — the tier tables show win rate flat from 3
# to 15+ points of disagreement. What might carry information is edge relative
# to how accurate the model is *on that particular game*. A 5-point edge on a
# mature, fully-measured matchup is a stronger claim than a 12-point edge in
# Week 5 between teams the ratings barely know.
#
# So: fit expected absolute error per game, then score confidence as
# |edge| / expected_error. Whether that actually sorts winners from losers is an
# empirical question, and the backtest reports it either way.

CONF_FEATURES = [
    "abs_pred",        # big predicted margins have bigger errors
    "model_disagree",  # margin-based vs EPA-based ratings disagreeing = uncertainty
    "immaturity",      # 1/(games played), high early in the season
    "missing_epa",
    "rivalry",
    "wx_unknown",
]


def confidence_features(frame, pred_col="pred", alt_pred_col=None,
                        games_played=None, min_games_ref=6.0):
    """Build the uncertainty feature matrix. Returns (X, names)."""
    n = len(frame)
    cols, names = [], []

    if pred_col in frame.columns:
        cols.append(frame[pred_col].abs().fillna(0.0).to_numpy(dtype=float))
        names.append("abs_pred")

    if alt_pred_col and alt_pred_col in frame.columns and pred_col in frame.columns:
        cols.append((frame[pred_col] - frame[alt_pred_col]).abs().fillna(0.0)
                    .to_numpy(dtype=float))
        names.append("model_disagree")

    if games_played is not None:
        gp = np.array([
            min(games_played.get(h, 0), games_played.get(a, 0))
            for h, a in zip(frame["home_team"], frame["away_team"])
        ], dtype=float)
        cols.append(min_games_ref / np.clip(gp, 1.0, None))
        names.append("immaturity")

    if "ppa_margin" in frame.columns:
        cols.append(frame["ppa_margin"].isna().astype(float).to_numpy())
        names.append("missing_epa")

    if "rivalry" in frame.columns:
        cols.append(frame["rivalry"].fillna(0.0).to_numpy(dtype=float))
        names.append("rivalry")

    if "wx_known" in frame.columns:
        cols.append((1.0 - frame["wx_known"].fillna(0.0)).to_numpy(dtype=float))
        names.append("wx_unknown")

    return (np.column_stack(cols) if cols else np.zeros((n, 0))), names


def fit_uncertainty(X, residuals, floor=6.0, ceiling=30.0):
    """Regress absolute error on the confidence features.

    Returns (weights, mean_abs_error). Predictions are clipped to a sane band so
    a weird feature combination cannot produce a near-zero denominator and an
    absurd confidence score.
    """
    y = np.abs(np.asarray(residuals, dtype=float))
    ok = np.isfinite(y)
    if X.size:
        ok = ok & np.isfinite(X).all(axis=1)
    if ok.sum() < 300:
        return None, float(np.nanmean(y)) if ok.sum() else float("nan")
    A = np.column_stack([X[ok], np.ones(int(ok.sum()))]) if X.size else np.ones((int(ok.sum()), 1))
    w, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return w, float(np.mean(y[ok]))


def predict_uncertainty(X, w, fallback, floor=6.0, ceiling=30.0):
    """Expected absolute error per game."""
    n = X.shape[0] if X.size else 0
    if w is None or not n:
        return np.full(max(n, 0), fallback if fallback == fallback else 13.0)
    A = np.column_stack([X, np.ones(n)]) if X.size else np.ones((n, 1))
    out = np.nan_to_num(A @ w, nan=fallback)
    return np.clip(out, floor, ceiling)


def confidence_score(edge, sigma):
    """|edge| divided by expected error. Higher means a stronger claim."""
    edge = np.asarray(edge, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.abs(edge) / np.clip(sigma, 1e-6, None)


# ----------------------------------------------------------------------
# preseason prior
# ----------------------------------------------------------------------

def season_ratings(d, years, use_epa, target="margin"):
    """Per-season ratings, each season fit in isolation."""
    R, H = {}, {}
    for yr in years:
        sub = d[d.season == yr]
        r_m, hfa = fit_ratings(sub, target, ridge=12.0, cap=35.0)
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
    + prior-year SP+ onto end-of-season rating.

    Note on SP+: only the PRIOR season's final SP+ is used. Using season Y's
    final SP+ to predict games inside season Y would leak end-of-season
    knowledge backwards and inflate the backtest.
    """

    FEATURES = ["r_prev", "r_prev2", "ret", "talent", "sp_prev"]

    def __init__(self, season_r, ret, talent, sp=None):
        self.season_r = season_r
        self.ret = ret
        self.talent = talent
        self.sp = sp or {}
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
            sp_prev = self.sp.get(yr - 1, {})
            for team, target in self.season_r[yr].items():
                if team == "NON_FBS":
                    continue
                rows.append({
                    "season": yr, "team": team, "target": target,
                    "r_prev": self.season_r[yr - 1].get(team, np.nan),
                    "r_prev2": self.season_r[yr - 2].get(team, np.nan),
                    "ret": self.ret.get(yr, {}).get(team, np.nan),
                    "talent": self.talent.get(yr, {}).get(team, np.nan),
                    "sp_prev": sp_prev.get(team, np.nan),
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
                "sp_prev": self.sp.get(season - 1, {}).get(t, self.medians.get("sp_prev", 0.0)),
            }
            x = np.array([vals[f] for f in self.feats] + [1.0])
            out[t] = float(x @ self.weights)

        mu = np.mean(list(out.values()))
        out = {t: v - mu for t, v in out.items()}
        out["NON_FBS"] = self.season_r[season - 1].get("NON_FBS", -22.0)
        return out
