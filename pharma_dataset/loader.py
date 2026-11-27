"""
Convenience loaders for PharmaNet-Disrupt.

    from loader import load_split, legacy_episodes
    d = load_split("data", "train")                 # dict of DataFrames
    eps = legacy_episodes(d)                        # single-link episodes in the old (lambda, r, b, ts, te) format

Observed columns end in `_obs` (with NaNs for data gaps); hidden truth starts with `true_`;
supervision targets start with `label_`. Models must only consume `_obs` columns, static
network metadata and alerts at inference time.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

STATIC = ["scenarios", "nodes", "products", "links", "mfr_molecule", "events", "shortages", "alerts"]
DAILY = ["link_daily", "product_daily", "node_daily", "api_daily"]


def load_split(root, split, daily=("link_daily", "product_daily"), columns=None):
    """columns: optional dict {table_name: [cols]} to load a subset of daily columns."""
    d = os.path.join(root, split)
    out = {k: pd.read_parquet(os.path.join(d, f"{k}.parquet")) for k in STATIC}
    for k in daily:
        files = sorted(glob.glob(os.path.join(d, k, "*.parquet")))
        cols = (columns or {}).get(k)
        out[k] = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return out


def observed_view(d):
    """Strip all hidden-truth and label columns (what an agent is allowed to see)."""
    keep = lambda df: df[[c for c in df.columns if not c.startswith(("true_", "label_"))]]
    v = {k: keep(d[k]) for k in DAILY if k in d and k in ("link_daily", "product_daily")}
    v["alerts"] = d["alerts"][["scenario_id", "alert_id", "day", "source", "entity_type", "entity", "reported_cause", "text"]]
    v["products"], v["links"], v["scenarios"] = d["products"], d["links"], d["scenarios"]
    return v


def legacy_episodes(d, min_severity=0.3):
    """Map the network data to the single-supplier episode format used in Experiments 1-3:
    one episode per (link, dominant supply event) with
        lambda_obs = orders placed on that supplier, r_obs = receipts, b = outstanding orders,
        ts/te = event start / end, plus severity and type for analysis.
    Links with no supply event become negative (no-disruption) episodes."""
    ld = d["link_daily"]
    ev = d["events"]
    ev = ev[(ev.category == "supply") & ev.in_horizon]
    eps = []
    for (sid, lid), g in ld.groupby(["scenario_id", "link_id"], sort=False):
        g = g.sort_values("day")
        lam = g.orders_obs.interpolate(limit_direction="both").to_numpy()
        r = g.receipts_obs.interpolate(limit_direction="both").to_numpy()
        b = g.outstanding_obs.interpolate(limit_direction="both").to_numpy()
        cand = ev[(ev.scenario_id == sid)]
        hit = np.array([lid in json.loads(s) for s in cand.affected_links], dtype=bool)
        cand = cand[hit & (cand.severity.to_numpy() >= min_severity)] if len(cand) else cand
        base = dict(scenario_id=sid, link_id=int(lid), t=g.day.to_numpy(), lambda_obs=lam, r_obs=r, b=b,
                    c_true=g.true_site_capacity.to_numpy())
        if len(cand) == 0:
            eps.append(dict(base, has_disruption=False, ts=-1.0, te=-1.0, event_type="none", severity=0.0))
            continue
        e = cand.sort_values("severity", ascending=False).iloc[0]
        te = float(e.end_day) if e.end_day == e.end_day and e.end_day is not None else float(len(g))
        eps.append(dict(base, has_disruption=True, ts=float(e.start_day), te=min(te, float(len(g))),
                        event_type=e.event_type, severity=float(e.severity), event_id=int(e.event_id)))
    return eps
