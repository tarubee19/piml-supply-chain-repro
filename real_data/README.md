Applying the models to a real dataset
======================================

Everything in the main repo uses our own synthetic simulator. This folder
applies the same models to a real, public dataset instead, as a second,
independent check.

Dataset
-------

DataCo Smart Supply Chain (Constante, Silva, Pereira, 2019), 180,508 real
e-commerce transactions from Jan 2015 to Sep 2017: order date, shipping
date, quantities, region, and a Late_delivery_risk flag per order. Hosted
on Mendeley Data (DOI 10.17632/8gx2fvg2k6) and Kaggle; publicly available,
no login needed to obtain a copy.

What we built from it
----------------------

The paper's model needs an order-inflow series lambda(t), a fulfillment-rate
series r(t), and a backlog b(t) = cumulative(lambda - r), floored at 0 --
the same conservation law as the paper's Eq. 1. We built that directly from
the real data: lambda(t) is the count of orders placed per day (or week),
r(t) is the count of orders shipped per day (or week). We tried both daily
and weekly aggregation.

What we ran
-----------

1. Grey-box per-series fit (scipy least-squares) directly on the real
   backlog series -- this needs no labels, so it's a fair, direct
   application to real data.
2. Our PIML network, already trained on the synthetic simulator, run
   out-of-distribution on the real series' features (no retraining).

Result: it doesn't work, and that's the finding
-------------------------------------------------

Grey-box collapses to a zero-duration window (ts = te) on both the daily
and weekly real series, and its reconstructed backlog drifts far from the
real one (mean error ~5, vs. the real backlog's own range of 0-0.6 on the
weekly series). PIML, run out-of-distribution, outputs a window on its own
900-step synthetic training scale that has no meaningful mapping onto the
real 162-week series.

The reason is structural, not a bug: the paper's model assumes the
fulfillment rate r(t) is capacity-limited and changes only slowly, during
a disruption. In the real DataCo data, orders placed and orders shipped
move together week to week (std ~0.21 around their mean, closely
correlated) -- a real retailer flexes its shipping rate to match demand
day to day, rather than running at a fixed rate that only drops during a
supplier disruption. That correlation is exactly what the model doesn't
allow for, so it has no slow, isolated "window" to find. The real data's
own Late_delivery_risk rate confirms this: it hovers around 55% almost
the entire 2.7-year period (peak 57%), rather than spiking during one
identifiable incident -- the lateness here looks chronic and systemic
(e.g., to do with shipping-mode SLAs), not a one-off supplier disruption
of the kind the paper models.

What this means
----------------

The paper's physics (and our grey-box/PIML reproduction of it) is built
for a clean, single-supplier disruption against a baseline where supply
would otherwise track demand almost perfectly. That's a reasonable model
for the kind of event the paper targets (a single supplier going down for
weeks), but it doesn't transfer as-is to messy, chronic-lateness retail
data like this one, where the "backlog" signal the model expects isn't
really there. A fair next step, if this direction is worth pursuing
further, would be a real dataset built around one specific, dated
disruption (e.g. a semiconductor shortage or a named shipping-lane
closure) rather than general retail lateness, so there's an actual
isolated window for the model to recover.

Files
-----

build_series.py and build_series_weekly.py build the daily/weekly series
from the raw CSV. dataco_raw.csv is the real dataset itself (not committed
here due to size -- see the Dataset section above to get your own copy).
