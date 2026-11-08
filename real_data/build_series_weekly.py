"""
Same as build_series.py but aggregated WEEKLY instead of daily.

Why: the paper's model assumes capacity c(t) is a SLOW variable that only
moves during a disruption. Daily real retail data has a strong day-of-week
pattern (weekday vs weekend volume) that violates that assumption and
dominates the signal. Weekly aggregation removes that nuisance variation
and leaves the slower trend the model is actually designed to explain.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("dataco_raw.csv", encoding="latin1")
df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
df["ship_date"] = pd.to_datetime(df["shipping date (DateOrders)"], format="%m/%d/%Y %H:%M")

t0 = df["order_date"].min().normalize()
df["order_week"] = ((df["order_date"].dt.normalize() - t0).dt.days // 7)
df["ship_week"] = ((df["ship_date"].dt.normalize() - t0).dt.days // 7)

T = int(max(df["order_week"].max(), df["ship_week"].max())) + 1
print(f"T = {T} weeks")

lam = np.zeros(T)
r = np.zeros(T)
late_rate = np.zeros(T)

orders_per_wk = df.groupby("order_week").size()
ships_per_wk = df.groupby("ship_week").size()
late_per_wk = df.groupby("order_week")["Late_delivery_risk"].mean()

for wk, cnt in orders_per_wk.items():
    if 0 <= wk < T:
        lam[wk] = cnt
for wk, cnt in ships_per_wk.items():
    if 0 <= wk < T:
        r[wk] = cnt
for wk, rate in late_per_wk.items():
    if 0 <= wk < T:
        late_rate[wk] = rate

scale = lam.mean()
lam_n, r_n = lam / scale, r / scale

b = np.zeros(T)
for i in range(1, T):
    b[i] = max(0.0, b[i - 1] + (lam_n[i - 1] - r_n[i - 1]))

np.savez("real_series_weekly.npz", t=np.arange(T), lambda_obs=lam_n, r_obs=r_n, b=b,
         late_rate=late_rate, scale=scale)
print(f"backlog: min={b.min():.2f} max={b.max():.2f} mean={b.mean():.2f}")
print(f"lam_n std={lam_n.std():.3f}  r_n std={r_n.std():.3f}")

roll = pd.Series(late_rate).rolling(8, center=True).mean()
peak_wk = int(roll.idxmax())
print(f"Highest late-delivery 8-week window centered at week {peak_wk} "
      f"({(t0 + pd.Timedelta(weeks=peak_wk)).date()}), rate={roll.max():.3f}")
