"""Generate a synthetic weekly sales series for a Kenyan dairy (yoghurt) portfolio.

Components (each one is something a forecasting model has to learn):
  * Trend          : ~7% annual volume growth, with a one-off -6% level drop after a
                     price increase in July 2024 (a structural break)
  * Yearly season  : school-holiday peaks (Apr, Aug, Dec), a January slump, and a
                     softer long-rains period (May)
  * Holidays       : Christmas / New Year and Easter weeks get extra uplift
  * Promotions     : planned trade promotions (~12% of weeks) lift volume ~18%, with a
                     small "post-promo dip" the following week (pantry loading)
  * Noise          : AR(1) autocorrelated noise (shocks persist for a few weeks)

Promotions and holidays are known in advance by the commercial team, so they are
exported as "future-known" regressors, exactly as they would be in a real S&OP process.
All data is synthetic. Usage: python data/generate_sales.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
OUT = Path(__file__).parent

weeks = pd.date_range("2021-01-04", "2026-09-28", freq="W-MON")   # week-start Mondays
n = len(weeks)
t = np.arange(n)
woy = weeks.isocalendar().week.values.astype(int)

trend = 10_000 * (1.07 ** (t / 52))
trend *= np.where(weeks >= "2024-07-01", 0.94, 1.0)                # price-increase level shift

def bump(center, width, height):                                    # smooth seasonal bump by week-of-year
    d = np.minimum(abs(woy - center), 52 - abs(woy - center))
    return height * np.exp(-0.5 * (d / width) ** 2)

season = (1 + bump(15, 1.5, .10) + bump(33, 2, .12) + bump(51, 1.5, .20)
          - bump(2, 1.5, .15) - bump(20, 2.5, .07))

easter = pd.to_datetime(["2021-04-04", "2022-04-17", "2023-04-09", "2024-03-31", "2025-04-20", "2026-04-05"])
holiday = np.zeros(n)
for i, w in enumerate(weeks):
    if (w.month == 12 and w.day >= 18) or (w.month == 1 and w.day <= 3):
        holiday[i] = 1
    if any((w <= e) & (e < w + pd.Timedelta(days=7)) for e in easter):
        holiday[i] = 1

promo = np.zeros(n)
i = 3
while i < n:
    if RNG.random() < .12:
        promo[i] = 1
        i += 4                                                       # promos are spaced out
    i += 1
post_promo = np.r_[0, promo[:-1]]

noise = np.zeros(n)
for k in range(1, n):
    noise[k] = .55 * noise[k - 1] + RNG.normal(0, .035)

units = trend * season * (1 + .08 * holiday) * (1 + .18 * promo) * (1 - .06 * post_promo) * np.exp(noise)
df = pd.DataFrame({"week": weeks.date, "units": units.round(0).astype(int),
                   "promo": promo.astype(int), "holiday": holiday.astype(int)})
df.to_csv(OUT / "weekly_sales.csv", index=False)
print(f"{len(df)} weeks: {df.week.min()} to {df.week.max()} | mean {df.units.mean():,.0f} units/week | "
      f"promo weeks {int(promo.sum())} | holiday weeks {int(holiday.sum())}")
