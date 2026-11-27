"""
Build the PharmaNet-Disrupt dataset: sample scenarios, simulate, derive labels, add observation
noise and external alerts, and write sharded Parquet files per split.

    python generate.py --out data --scale full      # full dataset (default sizes below)
    python generate.py --out sample --scale sample  # small sample for the repo

Fully deterministic: scenario seed = hash(split, index).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import pharmasim as ps

SPLITS = {  # split -> (regime, n_scenarios)
    "full": {"train": ("iid", 2000), "val": ("iid", 400), "test": ("iid", 400),
             "test_ood_severe": ("ood_severe", 200), "test_ood_network": ("ood_network", 200),
             "test_ood_noise": ("ood_noise", 200)},
    "sample": {"train": ("iid", 20), "val": ("iid", 5), "test": ("iid", 5),
               "test_ood_severe": ("ood_severe", 3), "test_ood_network": ("ood_network", 3),
               "test_ood_noise": ("ood_noise", 3)},
}
SHARD = 100
HORIZONS = (14, 30, 60)
CAP_IMPAIRED = 0.90


def scenario_seed(split, i):
    return int(hashlib.sha256(f"pharmanet-v1/{split}/{i}".encode()).hexdigest()[:8], 16)


def _runs(mask):
    """Start/end (inclusive) of contiguous True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


# --------------------------------------------------------------------------------------
def build(sc, out, split, scen_idx):
    sid = f"{split}-{scen_idx:05d}"
    T = sc["T"]
    rng = np.random.default_rng(sc["seed"] + 104729)
    P, M, A, LK, MM, EV = (sc["products"], sc["manufacturers"], sc["api_suppliers"], sc["links"],
                          sc["mfr_molecule"], sc["events"])
    nL, nP = len(LK), len(P)
    mfr_id = [m["node_id"] for m in M]
    api_id = [a["node_id"] for a in A]
    mm_index = {(x["mfr"], x["molecule"]): x for x in MM}
    link_api = [api_id[mm_index[(L["mfr"], P[L["product"]]["molecule_id"])]["api_node"]] for L in LK]
    lm = np.array([L["mfr"] for L in LK]); lp = np.array([L["product"] for L in LK])

    # ------------------------------------------------------------------ event truth
    node_c = {mfr_id[i]: out["c_mfr"][i] for i in range(len(M))}
    node_c.update({api_id[i]: out["c_api"][i] for i in range(len(A))})
    node_region = {m["node_id"]: m["region"] for m in M + A}
    ev_rows, ev_links = [], {}
    for e in EV:
        s = e["start"]
        end = T if e.get("end") is None else e["end"]
        typ = e["type"]
        if e["tier"] == "manufacturer" or e["tier"] == "lane":
            aff = [L["link_id"] for L in LK if mfr_id[L["mfr"]] == e["node"]]
            nodes = [e["node"]]
        elif e["tier"] == "api_supplier":
            aff = [L["link_id"] for L in LK if link_api[L["link_id"]] == e["node"]]
            nodes = [e["node"]]
        elif e["tier"] == "region":
            reg = e["node"].replace("REGION-", "")
            nodes = [n for n, r in node_region.items() if r == reg]
            aff = [L["link_id"] for L in LK if mfr_id[L["mfr"]] in nodes or link_api[L["link_id"]] in nodes]
        elif e["tier"] == "product_line":
            aff = [int(e["node"].split("-")[1])]
            nodes = [mfr_id[LK[aff[0]]["mfr"]]]
        else:  # demand surge on a therapeutic class
            cls = e["node"].replace("CLASS-", "")
            aff = [L["link_id"] for L in LK if P[L["product"]]["therapeutic_class"] == cls]
            nodes = []
        # recovery day: first day >= end with capacity back >= 95% on all hit nodes
        rec_day, censored = None, False
        if "floor" in e and nodes:
            if end >= T:
                censored = True
            else:
                ok = np.ones(T - end, bool)
                for n in nodes:
                    ok &= node_c[n][end:] >= 0.95
                idx = np.where(ok)[0]
                if len(idx):
                    rec_day = int(end + idx[0])
                else:
                    censored = True
        elif typ == "discontinuation":
            censored = True
        elif typ in ("logistics_disruption", "demand_surge"):
            rec_day = min(end, T)
            censored = end >= T
        ev_links[e["event_id"]] = aff
        severity = (1.0 - e["floor"]) if "floor" in e else (
            1.0 if typ == "discontinuation" else e.get("loss", e.get("mult", 0.0)))
        ev_rows.append(dict(
            scenario_id=sid, event_id=e["event_id"], event_type=typ, category=e["category"], tier=e["tier"],
            root_node=e["node"], hit_nodes=json.dumps(nodes), start_day=int(s),
            end_day=None if e.get("end") is None else int(e["end"]), recovery_day=rec_day,
            censored=bool(censored or s >= T), in_horizon=bool(s < T),
            severity=float(severity), capacity_floor=e.get("floor"), theta_down=e.get("theta_down"),
            theta_up=e.get("theta_up"), reject_rate=e.get("reject"), leadtime_mult=e.get("lt_mult"),
            transit_loss=e.get("loss"), demand_mult=e.get("mult"), notice_days=e.get("notice_days"),
            affected_links=json.dumps(aff),
            affected_products=json.dumps(sorted({int(lp[l]) for l in aff})),
        ))

    # ------------------------------------------------------------------ per-link daily labels
    lab_dis = np.zeros((nL, T), bool)          # a supply disruption cause is active upstream of the link
    lab_eid = np.full((nL, T), -1, np.int32)   # dominant (most severe) active event
    lab_sev = np.zeros((nL, T), np.float32)
    lab_planned = np.zeros((nL, T), bool)
    for r in ev_rows:
        if not r["in_horizon"] or r["category"] == "demand":
            continue
        s = r["start_day"]
        e = T if r["end_day"] is None else min(T, r["end_day"])
        for l in json.loads(r["affected_links"]):
            if r["category"] == "planned":
                lab_planned[l, s:e] = True
                continue
            lab_dis[l, s:e] = True
            upd = np.zeros(T, bool); upd[s:e] = r["severity"] > lab_sev[l, s:e]
            lab_eid[l, upd] = r["event_id"]; lab_sev[l, upd] = r["severity"]
    lnk = out["link"]
    # supply capability of the link is physically reduced: site capacity x line status x API
    # availability below 90%, or the lane is degraded (lead time / cold-chain loss), or batches fail QC
    cap_frac = out["c_mfr"][lm] * (out["line"] > 0) * lnk["api_scale"]
    impaired = cap_frac < CAP_IMPAIRED
    impaired |= (out["lt_mult"][lm] > 1.2) | (out["loss"][lm] > 0.05)

    # per-event impact window on links (first/last impaired day while event in force or recovering)
    for r in ev_rows:
        if r["category"] != "supply" or not r["in_horizon"]:
            r["impact_start"] = r["impact_end"] = None
            continue
        aff = json.loads(r["affected_links"])
        hi = T if (r["recovery_day"] is None) else min(T, r["recovery_day"] + 120)
        m = impaired[aff, r["start_day"]:hi].any(0) if aff else np.zeros(0, bool)
        idx = np.where(m)[0]
        r["impact_start"] = int(r["start_day"] + idx[0]) if len(idx) else None
        r["impact_end"] = int(r["start_day"] + idx[-1]) if len(idx) else None

    # ------------------------------------------------------------------ shortage episodes + attribution
    pr = out["product"]
    unmet = pr["unmet"]; dem = pr["demand"]
    short = unmet > 0.02 * np.maximum(dem, 1e-9)
    sh_rows = []
    sh_id = np.full((nP, T), -1, np.int32); sh_cause = np.full((nP, T), "", object)
    for p in range(nP):
        runs = _runs(short[p])
        merged = []
        for a, bnd in runs:                                  # merge gaps < 7 days
            if merged and a - merged[-1][1] <= 7:
                merged[-1][1] = bnd
            else:
                merged.append([a, bnd])
        plinks = np.where(lp == p)[0]
        for a, bnd in merged:
            if bnd - a + 1 < 3:
                continue
            best, best_score = None, 0.0
            for r in ev_rows:
                if not r["in_horizon"] or r["category"] == "planned":
                    continue
                aff = set(json.loads(r["affected_links"])) & set(plinks.tolist())
                if not aff:
                    continue
                lo = r["start_day"]
                hi = T if r["recovery_day"] is None else r["recovery_day"] + 150
                if not (lo <= bnd and hi >= a - 30):
                    continue
                share = sum(LK[l]["share0"] for l in aff)
                score = share * r["severity"] if r["category"] == "supply" else 0.5 * (r["severity"] - 1.0)
                if r["event_type"] == "discontinuation":
                    score = share
                if score > best_score:
                    best, best_score = r, score
            k = len(sh_rows)
            cause_type = best["event_type"] if best else "unattributed"
            sh_rows.append(dict(scenario_id=sid, shortage_id=k, product_id=p, start_day=int(a), end_day=int(bnd),
                                duration=int(bnd - a + 1), censored=bool(bnd == T - 1),
                                unmet_units=float(unmet[p, a:bnd + 1].sum()),
                                fill_rate=float(1 - unmet[p, a:bnd + 1].sum() / max(dem[p, a:bnd + 1].sum(), 1e-9)),
                                cause_event_id=None if best is None else best["event_id"], cause_type=cause_type,
                                critical=P[p]["critical"]))
            sh_id[p, a:bnd + 1] = k
            sh_cause[p, a:bnd + 1] = cause_type

    # forward-looking early-warning labels
    fut = {}
    for h in HORIZONS:
        c = np.concatenate([np.zeros((nP, 1)), np.cumsum(short, 1)], 1)
        idx = np.arange(T)
        hi = np.minimum(idx + h + 1, T)
        fut[h] = (c[:, hi] - c[:, idx + 1]) > 0                # any shortage day in (t, t+h]

    # onset-based early warning: a NEW shortage episode starts 15-60 days ahead (lead time to act)
    onset = np.zeros((nP, T), bool)
    for r in sh_rows:
        onset[r["product_id"], r["start_day"]] = True
    co = np.concatenate([np.zeros((nP, 1)), np.cumsum(onset, 1)], 1)
    idx = np.arange(T)
    onset_15_60 = (co[:, np.minimum(idx + 61, T)] - co[:, np.minimum(idx + 15, T)]) > 0
    days_to_onset = np.full((nP, T), -1, np.int16)            # -1 = no further onset in horizon
    for p in range(nP):
        nxt = -1
        for t in range(T - 1, -1, -1):
            if onset[p, t]:
                nxt = t
            days_to_onset[p, t] = (nxt - t) if nxt >= 0 else -1

    # ------------------------------------------------------------------ observation layer
    nmul = sc["noise_mult"]
    gap = np.zeros((nL, T), bool)                     # EDI / data-feed outages per link
    for l in range(nL):
        for _ in range(rng.poisson(1.5 * nmul * T / 365)):
            s = int(rng.integers(0, T)); gap[l, s:s + int(rng.integers(1, 11))] = True
    pgap = np.zeros((nP, T), bool)
    for p in range(nP):
        for _ in range(rng.poisson(0.8 * nmul * T / 365)):
            s = int(rng.integers(0, T)); pgap[p, s:s + int(rng.integers(1, 6))] = True

    def obs(x, mask, rel_sd=0.0):
        y = x * (1 + rng.normal(0, rel_sd, x.shape)) if rel_sd > 0 else x.copy()
        y = np.maximum(y, 0).astype(np.float32)
        y[mask] = np.nan
        return y

    day = np.arange(T, dtype=np.int16)
    link_df = pd.DataFrame({
        "scenario_id": np.repeat(sid, nL * T), "link_id": np.repeat(np.arange(nL, dtype=np.int16), T),
        "day": np.tile(day, nL),
        # ---- observed by the buyer ----
        "orders_obs": obs(lnk["orders"], gap).ravel(),
        "receipts_obs": obs(lnk["receipts"], gap).ravel(),
        "damaged_obs": obs(lnk["damaged"], gap).ravel(),
        "cancelled_obs": obs(lnk["cancelled"], gap).ravel(),
        "outstanding_obs": obs(lnk["outstanding"], gap).ravel(),
        "share": lnk["share"].astype(np.float32).ravel(),
        "data_gap": gap.ravel(),
        # ---- hidden truth ----
        "true_orders": lnk["orders"].astype(np.float32).ravel(),
        "true_receipts": lnk["receipts"].astype(np.float32).ravel(),
        "true_backlog": lnk["backlog"].astype(np.float32).ravel(),
        "true_in_transit": lnk["in_transit"].astype(np.float32).ravel(),
        "true_fg_stock": lnk["fg"].astype(np.float32).ravel(),
        "true_qc_wip": lnk["wip"].astype(np.float32).ravel(),
        "true_production": lnk["production"].astype(np.float32).ravel(),
        "true_rejected": lnk["rejected"].astype(np.float32).ravel(),
        "true_shipments": lnk["shipments"].astype(np.float32).ravel(),
        "true_site_capacity": out["c_mfr"][lm].astype(np.float32).ravel(),
        "true_eff_capacity": lnk["cap_eff"].astype(np.float32).ravel(),
        "true_api_availability": lnk["api_scale"].astype(np.float32).ravel(),
        "true_leadtime_mult": out["lt_mult"][lm].astype(np.float32).ravel(),
        "true_transit_loss": out["loss"][lm].astype(np.float32).ravel(),
        "true_reject_rate": out["reject"][lm].astype(np.float32).ravel(),
        "true_line_active": (out["line"] > 0).ravel(),
        # ---- labels ----
        "label_disrupted": lab_dis.ravel(),
        "label_planned": lab_planned.ravel(),
        "label_impaired": impaired.ravel(),
        "label_event_id": lab_eid.ravel(),
        "label_severity": lab_sev.ravel(),
    })
    prod_df = pd.DataFrame({
        "scenario_id": np.repeat(sid, nP * T), "product_id": np.repeat(np.arange(nP, dtype=np.int16), T),
        "day": np.tile(day, nP),
        "demand_obs": obs(dem, pgap, 0.02 * nmul).ravel(),
        "dispensed_obs": obs(pr["dispensed"], pgap).ravel(),
        "unmet_obs": obs(unmet, pgap).ravel(),
        "inventory_obs": obs(pr["inventory"], pgap, 0.01 * nmul).ravel(),
        "data_gap": pgap.ravel(),
        "true_demand": dem.astype(np.float32).ravel(),
        "true_inventory": pr["inventory"].astype(np.float32).ravel(),
        "true_demand_mult": out["demand_mult"].astype(np.float32).ravel(),
        "label_shortage": short.ravel(),
        "label_shortage_id": sh_id.ravel(),
        "label_shortage_cause": sh_cause.astype(str).ravel(),
        **{f"label_shortage_next{h}d": fut[h].ravel() for h in HORIZONS},
        "label_onset_15_60d": onset_15_60.ravel(),
        "label_days_to_onset": days_to_onset.ravel(),
        "label_right_censored60": np.tile(day >= T - 60, nP),
    })
    node_rows = []
    for i, m in enumerate(M):
        node_rows.append((m["node_id"], out["c_mfr"][i], out["active_mfr"][i]))
    for i, a in enumerate(A):
        node_rows.append((a["node_id"], out["c_api"][i], out["active_api"][i]))
    node_df = pd.DataFrame({
        "scenario_id": np.repeat(sid, len(node_rows) * T),
        "node_id": np.repeat([n for n, _, _ in node_rows], T),
        "day": np.tile(day, len(node_rows)),
        "true_capacity": np.concatenate([c for _, c, _ in node_rows]).astype(np.float32),
        "label_event_active": np.concatenate([a for _, _, a in node_rows]),
    })
    mmd = out["mfr_molecule"]
    api_df = pd.DataFrame({
        "scenario_id": np.repeat(sid, len(MM) * T),
        "mm_id": np.repeat(np.arange(len(MM), dtype=np.int16), T), "day": np.tile(day, len(MM)),
        "true_api_on_hand": mmd["api_on_hand"].astype(np.float32).ravel(),
        "true_api_arrivals": mmd["api_arrivals"].astype(np.float32).ravel(),
        "true_api_consumed": mmd["api_consumed"].astype(np.float32).ravel(),
    })

    # ------------------------------------------------------------------ external alerts (for text / news agents)
    alerts = make_alerts(sc, sid, ev_rows, lnk, pr, short, rng, mfr_id, link_api)

    # ------------------------------------------------------------------ static tables
    scen = dict(scenario_id=sid, split=split, regime=sc["regime"], seed=sc["seed"], start_date=sc["start_date"],
                horizon_days=T, risk_multiplier=sc["risk"], noise_multiplier=sc["noise_mult"],
                n_products=nP, n_manufacturers=len(M), n_api_suppliers=len(A), n_links=nL,
                n_events=sum(r["in_horizon"] for r in ev_rows),
                n_supply_events=sum(r["in_horizon"] and r["category"] == "supply" for r in ev_rows),
                n_shortages=len(sh_rows),
                fill_rate=float(1 - unmet.sum() / max(dem.sum(), 1e-9)))
    nodes = [dict(scenario_id=sid, node_id=m["node_id"], tier="manufacturer", region=m["region"],
                  headroom=m["headroom"], surge_max=m["surge_max"], ramp_days=m["ramp_days"],
                  fg_stock_days=m["fg_days"], api_stock_days=m["api_stock_days"], base_reject=m["base_reject"])
             for m in M]
    nodes += [dict(scenario_id=sid, node_id=a["node_id"], tier="api_supplier", region=a["region"],
                   headroom=a["headroom"], surge_max=a["surge_max"], ramp_days=a["ramp_days"]) for a in A]
    prods = [dict(scenario_id=sid, **{k: v for k, v in p.items()}) for p in P]
    links = [dict(scenario_id=sid, link_id=L["link_id"], product_id=L["product"], manufacturer=mfr_id[L["mfr"]],
                  api_supplier=link_api[L["link_id"]], share0=L["share0"], qc_release_days=L["qc_days"],
                  batch_interval=L["batch_interval"], transit_days=L["transit_days"],
                  nominal_daily=float(out["nominal"][L["link_id"]]),
                  capacity_daily=float(out["Kcap"][L["link_id"]])) for L in LK]
    mms = [dict(scenario_id=sid, mm_id=x["mm_id"], manufacturer=mfr_id[x["mfr"]], molecule_id=x["molecule"],
                api_supplier=api_id[x["api_node"]], api_lead_days=x["lead_days"]) for x in MM]
    return dict(scenarios=[scen], nodes=nodes, products=prods, links=links, mfr_molecule=mms, events=ev_rows,
                shortages=sh_rows, alerts=alerts, link_daily=link_df, product_daily=prod_df,
                node_daily=node_df, api_daily=api_df)


# --------------------------------------------------------------------------------------
ALERT_P = {  # probability an event becomes publicly visible, (min, max) reporting delay in days
    "quality_failure": (0.45, (5, 60)), "regulatory_action": (0.90, (3, 30)),
    "api_shortage": (0.25, (20, 90)), "natural_disaster": (0.95, (0, 3)),
    "logistics_disruption": (0.30, (0, 10)), "discontinuation": (0.80, None),
    "demand_surge": (0.65, (7, 30)), "planned_maintenance": (0.15, (-30, 0)),
}
SOURCES = {
    "quality_failure": ["FDA recall notice", "FDA Form 483 report", "supplier notification"],
    "regulatory_action": ["FDA warning letter", "FDA import alert", "regulatory news"],
    "api_shortage": ["trade press", "supplier notification"],
    "natural_disaster": ["news", "government emergency notice"],
    "logistics_disruption": ["carrier notice", "news"],
    "discontinuation": ["FDA discontinuation notice", "supplier notification"],
    "demand_surge": ["CDC surveillance report", "news"],
    "planned_maintenance": ["supplier notification"],
}
CAUSE_TEXT = {
    "quality_failure": "a manufacturing quality issue (sterility / particulate findings)",
    "regulatory_action": "regulatory action following a GMP inspection",
    "api_shortage": "constrained supply of the active pharmaceutical ingredient",
    "natural_disaster": "a natural disaster affecting regional manufacturing",
    "logistics_disruption": "transport and cold-chain delays",
    "discontinuation": "a business decision to discontinue the product",
    "demand_surge": "a sharp increase in clinical demand",
    "planned_maintenance": "scheduled plant maintenance",
}


def make_alerts(sc, sid, ev_rows, lnk, pr, short, rng, mfr_id, link_api):
    T = sc["T"]
    P, LK = sc["products"], sc["links"]
    out = []

    def add(day, source, entity_type, entity, cause, text, false_alarm, eid):
        if 0 <= day < T:
            out.append(dict(scenario_id=sid, alert_id=len(out), day=int(day), source=source,
                            entity_type=entity_type, entity=entity, reported_cause=cause, text=text,
                            is_false_alarm=false_alarm, true_event_id=eid))

    for r in ev_rows:
        typ = r["event_type"]
        p_pub, delay = ALERT_P[typ]
        if rng.random() > p_pub:
            continue
        if typ == "discontinuation":
            day = r["start_day"] - (r["notice_days"] or 0)
        else:
            day = r["start_day"] + int(rng.integers(delay[0], delay[1] + 1))
        prods = json.loads(r["affected_products"])
        pnames = ", ".join(P[p]["name"] for p in prods[:4]) or "multiple products"
        # entity resolution is imperfect: sometimes only region or product is named
        if r["tier"] == "region":
            ent_type, ent = "region", r["root_node"].replace("REGION-", "")
        elif r["tier"] == "therapeutic_class":
            ent_type, ent = "therapeutic_class", r["root_node"].replace("CLASS-", "")
        else:
            node = json.loads(r["hit_nodes"])[0] if json.loads(r["hit_nodes"]) else r["root_node"]
            u = rng.random()
            if u < 0.7:
                ent_type, ent = "supplier", node
            elif u < 0.85:
                ent_type, ent = "region", node.split("-")[1] if "-" in node else "unknown"
            else:
                ent_type, ent = "product", pnames
        cause = typ if rng.random() > 0.25 else "unknown"      # ~1/4 of public notices give no cause
        cause_txt = CAUSE_TEXT[typ] if cause != "unknown" else "reasons not disclosed"
        src = str(rng.choice(SOURCES[typ]))
        text = (f"[{src}] Supply of {pnames} may be affected due to {cause_txt}. "
                f"Entity referenced: {ent}.")
        if typ == "demand_surge":
            text = f"[{src}] Elevated {ent.replace('_', ' ')} drug utilisation reported; demand for {pnames} increasing."
        add(day, src, ent_type, ent, cause, text, False, r["event_id"])

    # supplier allocation notices derived from the true state (supplier rations when stock is thin)
    for l, L in enumerate(LK):
        ratio = lnk["backlog"][l] / np.maximum(lnk["orders"][l].mean(), 1e-9)
        on = np.convolve(ratio > 21, np.ones(14) / 14, mode="same") > 0.5     # sustained, de-flickered
        for a, _b in _runs(on):
            if _b - a >= 7 and rng.random() < 0.6:
                d = a + int(rng.integers(0, 10))
                add(d, "supplier allocation notice", "supplier", mfr_id[L["mfr"]], "unknown",
                    f"[supplier allocation notice] {mfr_id[L['mfr']]} has placed {P[L['product']]['name']} on allocation.",
                    False, None)
    # public shortage listing once the product is visibly short for a while
    for p in range(len(P)):
        sm = np.convolve(short[p], np.ones(14) / 14, mode="same") > 0.4
        for a, b in _runs(sm):
            if b - a >= 10 and rng.random() < 0.7:
                add(a + int(rng.integers(7, 21)), "national shortage list", "product", P[p]["name"], "unknown",
                    f"[national shortage list] {P[p]['name']} added to the drug shortage list.", False, None)
    # false alarms / rumours
    for _ in range(rng.poisson(1.5 * T / 365)):
        node = str(rng.choice(mfr_id + list(set(link_api))))
        typ = str(rng.choice(["quality_failure", "regulatory_action", "api_shortage", "logistics_disruption"]))
        add(int(rng.integers(0, T)), "unverified report", "supplier", node, typ,
            f"[unverified report] Possible disruption at {node} linked to {CAUSE_TEXT[typ]}.", True, None)
    out.sort(key=lambda x: x["day"])
    for i, x in enumerate(out):
        x["alert_id"] = i
    return out


# --------------------------------------------------------------------------------------
def _job(args):
    split, regime, i = args
    sc = ps.sample_scenario(scenario_seed(split, i), regime)
    out = ps.simulate(sc)
    return build(sc, out, split, i)


def write_split(out_dir, split, regime, n, workers):
    d = os.path.join(out_dir, split)
    os.makedirs(d, exist_ok=True)
    jobs = [(split, regime, i) for i in range(n)]
    static = {k: [] for k in ["scenarios", "nodes", "products", "links", "mfr_molecule", "events", "shortages", "alerts"]}
    with Pool(workers) as pool:
        for s0 in range(0, n, SHARD):
            res = pool.map(_job, jobs[s0:s0 + SHARD])
            shard = s0 // SHARD
            for tab in ["link_daily", "product_daily", "node_daily", "api_daily"]:
                df = pd.concat([r[tab] for r in res], ignore_index=True)
                os.makedirs(os.path.join(d, tab), exist_ok=True)
                df.to_parquet(os.path.join(d, tab, f"part-{shard:04d}.parquet"), compression="zstd", index=False)
            for k in static:
                for r in res:
                    static[k].extend(r[k])
            print(f"  {split}: {min(s0 + SHARD, n)}/{n}", flush=True)
    for k, rows in static.items():
        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(d, f"{k}.parquet"), compression="zstd", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--scale", default="full", choices=list(SPLITS))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", default=None, help="comma-separated subset of splits")
    a = ap.parse_args()
    t0 = time.time()
    for split, (regime, n) in SPLITS[a.scale].items():
        if a.only and split not in a.only.split(","):
            continue
        write_split(a.out, split, regime, n, a.workers)
    meta = dict(name="PharmaNet-Disrupt", version="1.0", horizon_days=ps.T_DAYS, burn_in_days=ps.BURN_IN,
                splits={k: dict(regime=v[0], n_scenarios=v[1]) for k, v in SPLITS[a.scale].items()},
                generated_seconds=round(time.time() - t0, 1),
                software=dict(python=__import__("sys").version.split()[0], numpy=np.__version__,
                              pandas=pd.__version__, pyarrow=__import__("pyarrow").__version__))
    with open(os.path.join(a.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
