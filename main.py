"""gta6-twitch-ccu-forecast-01 entry V2 — L1k6: last-1-hour-anchored kNN.

Forecasts GTA VI's launch-day Twitch viewers, hours 2..24, from the case
file's comparables (past big launches' hourly SullyGnome curves).

Method
------
Level anchor: the viewers in the single hour right before the release day
starts (23:00 UTC the day before day_start). For every comparable launch,
p1 is its hour-1 viewers and d = log(p1 / anchor). Neighbours are the
k = 6 comparables closest to the case in standardised [log p1, d] space
(population mean / sd over the usable comparables). For each hour h the
forecast is the median of the neighbours' log(viewers_h / anchor); the
spread is sigma = 0.7 * IQR / 1.349 of those log-ratios, floored at 0.08. The 0.7 factor
calibrates the neighbour spread: the IQR across six selected neighbours overstates how far a
new game's curve sits from their median (on the four practice cases the implied sigma runs
2-4x the realised median error in hours 7-24), so the raw IQR is shrunk toward the
sampling scale of the median.
Quantiles: anchor * exp(median + z_tau * sigma).

Fallbacks (never crash): if the case anchor is unreadable or fewer than
4 comparables are usable, use the house baseline (flat hour-1 with spread
0.15 * sqrt(h - 1)); if even hour 1 is missing, emit a flat level of 1.

Python 3 standard library only, deterministic, no network, no randomness.
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
(else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
K_NEIGHBOURS = 6
MIN_USABLE = 4
SIGMA_FLOOR = 0.08
# Neighbour-IQR calibration: the raw IQR overstates the predictive spread
# (see module docstring); 0.7 keeps the 90% coverage near-honest on practice
# while capturing most of the pinball gain. The practice-optimal factor is
# ~0.6; 0.7 is the conservative choice inside the flat optimum, so the band
# stays honest if the final launch surprises like p03 did.
SIGMA_SCALE = 0.7
BASE_SPREAD = 0.15


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(value):
    """Round to 2 decimals, half up (the house's rounding)."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 100
    return int(out) if out.is_integer() else out


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def percentile_linear(values, q):
    """Linear-interpolation percentile (numpy 'linear' method)."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def spread_of(values):
    """Calibrated IQR / 1.349, floored; degenerate for a single value."""
    if len(values) < 2:
        return SIGMA_FLOOR
    iqr = percentile_linear(values, 0.75) - percentile_linear(values, 0.25)
    return max(SIGMA_FLOOR, SIGMA_SCALE * iqr / 1.349)


def hourly_lookup(rows):
    """Hourly rows [{from, viewers, ...}] -> {datetime: viewers}."""
    table = {}
    for row in rows:
        try:
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers > 0:
            table[parse_time(row["from"])] = viewers
    return table


def last_hour_anchor(table, day_start):
    """Viewers in the single hour before day_start, or None."""
    return table.get(day_start - timedelta(hours=1))


def baseline_row(p1, hour):
    """House baseline quantiles for one hour."""
    grow = BASE_SPREAD * math.sqrt(hour - 1)
    return [round2(p1 * math.exp(z * grow)) for z in Z]


def baseline_forecast(p1):
    return {h: baseline_row(p1, h) for h in range(2, 25)}


def usable_comparables(case):
    """Comparables with a usable anchor, hour-1 print and release-day curve."""
    pool = []
    for comp in case.get("comparables") or []:
        start = parse_time(comp["day_start"])
        table = hourly_lookup(comp["hourly"])
        anchor = last_hour_anchor(table, start)
        first = table.get(start)
        if anchor is None or first is None:
            continue
        curve = {}
        for h in range(2, 25):
            value = table.get(start + timedelta(hours=h - 1))
            if value is not None:
                curve[h] = value
        pool.append({"name": comp["name"], "p1": first, "anchor": anchor,
                     "curve": curve})
    return pool


def standardise(pool):
    """Population mean/sd of [log p1, log(p1/anchor)] over the pool."""
    xs = [math.log(c["p1"]) for c in pool]
    ys = [math.log(c["p1"] / c["anchor"]) for c in pool]
    n = len(pool)
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in xs) / n) or 1.0
    sy = math.sqrt(sum((v - my) ** 2 for v in ys) / n) or 1.0
    return mx, my, sx, sy


def knn_forecast(case, anchor, p1, pool):
    """Median log-ratio curves from the k nearest comparables."""
    mx, my, sx, sy = standardise(pool)
    tx = (math.log(p1) - mx) / sx
    ty = (math.log(p1 / anchor) - my) / sy

    def distance(comp):
        dx = (math.log(comp["p1"]) - mx) / sx - tx
        dy = (math.log(comp["p1"] / comp["anchor"]) - my) / sy - ty
        return dx * dx + dy * dy

    neighbours = sorted(pool, key=lambda c: (distance(c), c["name"]))
    neighbours = neighbours[:K_NEIGHBOURS]
    rows = {}
    for h in range(2, 25):
        ratios = [math.log(n["curve"][h] / n["anchor"])
                  for n in neighbours if h in n["curve"]]
        if not ratios:
            rows[h] = baseline_row(p1, h)
            continue
        centre = median_of(ratios)
        sigma = spread_of(ratios)
        rows[h] = [round2(anchor * math.exp(centre + z * sigma)) for z in Z]
    return rows


def forecast_case(case):
    """Forecast one case, with the fallback chain."""
    day_start = parse_time(case["day_start"])
    known = next((r for r in case.get("known") or [] if r.get("hour") == 1),
                 None)
    p1 = max(1.0, float(known["viewers"])) if known else 1.0
    history = hourly_lookup(case.get("history_hourly") or [])
    anchor = last_hour_anchor(history, day_start)
    pool = usable_comparables(case)
    if anchor is not None and len(pool) >= MIN_USABLE:
        rows = knn_forecast(case, anchor, p1, pool)
    elif known is not None:
        rows = baseline_forecast(p1)
    else:
        rows = {h: [1, 1, 1, 1, 1] for h in range(2, 25)}
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = [min(max(q, 1e-9), 1e9) for q in rows[h]]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(result, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
