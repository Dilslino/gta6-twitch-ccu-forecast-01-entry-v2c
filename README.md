[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Dilslino/gta6-twitch-ccu-forecast-01-entry-v2c/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Dilslino/gta6-twitch-ccu-forecast-01-entry-v2c/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers, hour by hour — V2 (L1k6)

Entry for the [make.fun bounty gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast the release day's hourly Twitch viewers (hours 2-24) as five
quantiles (5/25/50/75/95%) on the log scale, scored by mean pinball loss.
The program reads every `<id>.in` in the input directory (env
`MAKEFUN_INPUT_DIR`, else `/input`) and writes `<id>.out` to the output
directory (env `MAKEFUN_OUTPUT_DIR`, else `/output`).

## Design: last-1-hour-anchored kNN (L1k6)

The hour-1 print (00:00 UTC) is a noisy, depressed read of a launch's
size, so the level is anchored to a single hour: the viewers at 23:00 UTC
of the day before the release day — the last pre-day hour. The 5-hour
mean dilutes that signal; the single 23:00 hour tracks it best.

Each comparable launch is summarised by two features: log(p1), the log of
its hour-1 print, and d = log(p1 / anchor), how depressed its hour 1 was
against its own pre-day level. The target's six nearest neighbours in
that (standardised, population mean/sd) 2-D space lend their release-day
curves: for each hour h the median of the neighbours' log(viewers_h /
anchor) is the median log-ratio, and the IQR of those log-ratios (as a
normal sigma, IQR / 1.349, floored at 0.08) is the spread. The forecast
is anchor * exp(median + z_tau * sigma).

Launch size drives the evening peak almost entirely (corr(log p1,
log-ratio at hour 21) about -0.94 across the comparables): small launches
peak ~9x their hour 1, huge ones ~1x, and the neighbours carry that shape.

If the anchor cannot be read, or fewer than 4 comparables are usable,
the entry falls back to the house baseline (flat hour-1, spread
0.15 * sqrt(h - 1)); it never crashes.

## Practice results (4 launch cases, kit pinball convention)

| model | mean | p00 | p01 | p02 | p03 |
|---|---|---|---|---|---|
| V1 (pre-5h anchor, k=8) | 0.1813 | 0.2354 | 0.1694 | 0.1242 | 0.1962 |
| **L1k6 (this entry)** | **0.1519** | 0.1574 | 0.0999 | 0.1391 | 0.2111 |

House baseline: 0.3871. L1k6 also beats V1 on leave-one-out over the 37
comparables (all 0.4534 vs 0.5087; p1>=10k 0.3055 vs 0.3442;
p1>=100k 0.2849 vs 0.3423).

Python 3 standard library only, deterministic, runtime ~0.06 s per case.
