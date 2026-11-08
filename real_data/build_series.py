"""
Turn the real DataCo Smart Supply Chain transactional dataset (180k+ orders,
Jan 2015 - Sep 2017) into the same kind of daily time series our paper
reproduction expects: an order-inflow series lambda(t), a fulfillment-rate
series r(t), and a backlog series b(t) = cumulative(lambda - r), clipped
at 0 -- exactly the same conservation law as Eq. 1 in the paper.

lambda(t): count of orders PLACED that day (order date)
r(t):      count of orders SHIPPED that day (shipping date)
b(t):      running backlog = cumulative(orders placed - orders shipped)

This is a simple, real-world proxy: it is not labelled with a known
disruption window, so we can't compute MAE against ground truth. Instead
we (a) fit the grey-box model directly -- it needs no labels, just the
observed series -- and (b) look at where the real data's own Late_delivery_risk
rate spikes, as an independent sanity check on whatever window grey-box finds.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("dataco_raw.csv", encoding="latin1")

df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
df["ship_date"] = pd.to_datetime(df["shipping date (DateOrders)"], format="%m/%d/%Y %H:%M")

t0 = df["order_date"].min().normalize()
df["order_day"] = (df["order_date"].dt.normalize() - t0).dt.days
df["ship_day"] = (df["ship_date"].dt.normalize() - t0).dt.days

T = int(max(df["order_day"].max(), df["ship_day"].max())) + 1
print(f"T = {T} days, t0 = {t0.date()}")

lam = np.zeros(T)   # orders placed per day
r = np.zeros(T)     # orders shipped per day
late_rate = np.zeros(T)
late_n = np.zeros(T)

orders_per_day = df.groupby("order_day").size()
ships_per_day = df.groupby("ship_day").size()
late_per_day = df.groupby("order_day")["Late_delivery_risk"].mean()

for day, cnt in orders_per_day.items():
    if 0 <= day < T:
        lam[day] = cnt
for day, cnt in ships_per_day.items():
    if 0 <= day < T:
        r[day] = cnt
for day, rate in late_per_day.items():
    if 0 <= day < T:
        late_rate[day] = rate

# Normalize to roughly the same scale as our synthetic simulator (order of ~1)
scale = lam.mean()
lam_n = lam / scale
r_n = r / scale

b = np.zeros(T)
for i in range(1, T):
    b[i] = max(0.0, b[i - 1] + (lam_n[i - 1] - r_n[i - 1]))

np.savez("real_series.npz", t=np.arange(T), lambda_obs=lam_n, r_obs=r_n, b=b,
         late_rate=late_rate, scale=scale, raw_lambda=lam, raw_r=r)

print("Saved real_series.npz")
print(f"backlog: min={b.min():.2f} max={b.max():.2f} mean={b.mean():.2f}")
print(f"late_rate: min={late_rate.min():.2f} max={late_rate.max():.2f} mean={late_rate.mean():.2f}")

# A simple, independent "where's the real stress period" signal: the 60-day
# rolling window with the highest average late-delivery rate.
roll = pd.Series(late_rate).rolling(60, center=True).mean()
peak_day = int(roll.idxmax())
print(f"Highest late-delivery-rate 60-day window centered at day {peak_day} "
      f"({(t0 + pd.Timedelta(days=peak_day)).date()}), rate={roll.max():.3f}")
