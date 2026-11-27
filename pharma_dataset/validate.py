"""
Validate PharmaNet-Disrupt: physical invariants, label statistics, example plots and two naive
difficulty baselines. Writes reports/stats.json, reports/STATS.md and reports/*.png

    python validate.py --root data
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loader import load_split

COLORS = {"quality_failure": "#d62728", "regulatory_action": "#9467bd", "api_shortage": "#ff7f0e",
          "natural_disaster": "#8c564b", "logistics_disruption": "#17becf", "discontinuation": "#7f7f7f",
          "demand_surge": "#2ca02c", "planned_maintenance": "#bcbd22"}


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def invariants(d):
    ld, pdl = d["link_daily"], d["product_daily"]
    res = {}
    res["outstanding_eq_backlog_plus_transit_maxerr"] = float(
        (ld.eval("outstanding_obs - true_backlog - true_in_transit").abs()[~ld.data_gap]).max())
    for c in ["true_backlog", "true_in_transit", "true_fg_stock", "true_qc_wip", "true_production", "true_shipments"]:
        res[f"min_{c}"] = float(ld[c].min())
    res["min_true_inventory"] = float(pdl.true_inventory.min())
    res["capacity_in_[0,1]"] = bool(ld.true_site_capacity.between(0, 1 + 1e-6).all())
    ev = d["events"]
    has_end = ev.end_day.notna()
    res["end_after_start"] = bool((ev.end_day[has_end] >= ev.start_day[has_end]).all())
    ok = ev.in_horizon & ~ev.censored & ev.recovery_day.notna() & ev.end_day.notna()
    res["recovery_after_end"] = bool((ev.recovery_day[ok] >= ev.end_day[ok]).all())
    # stock conservation at the buyer: I(t) - I(t-1) = receipts - dispensed
    g = pdl.sort_values(["scenario_id", "product_id", "day"])
    dI = g.groupby(["scenario_id", "product_id"]).true_inventory.diff()
    rec = d["link_daily"].merge(d["links"][["scenario_id", "link_id", "product_id"]], on=["scenario_id", "link_id"]) \
        .groupby(["scenario_id", "product_id", "day"]).true_receipts.sum()
    g = g.set_index(["scenario_id", "product_id", "day"])
    flow = rec.reindex(g.index).fillna(0) - g.dispensed_obs.fillna(np.nan)
    m = g.data_gap.to_numpy() == False
    err = (dI.to_numpy() - flow.to_numpy())[m]
    res["buyer_inventory_balance_maxerr"] = float(np.nanmax(np.abs(err)))
    return res


def stats(d, name):
    sc, ev, sh, al = d["scenarios"], d["events"], d["shortages"], d["alerts"]
    ld, pdl = d["link_daily"], d["product_daily"]
    evh = ev[ev.in_horizon]
    return {
        "split": name,
        "scenarios": int(len(sc)),
        "products": int(len(d["products"])), "manufacturers": int((d["nodes"].tier == "manufacturer").sum()),
        "api_suppliers": int((d["nodes"].tier == "api_supplier").sum()), "links": int(len(d["links"])),
        "link_days": int(len(ld)), "product_days": int(len(pdl)),
        "events": int(len(evh)), "events_by_type": evh.event_type.value_counts().to_dict(),
        "single_source_product_frac": float((d["links"].groupby(["scenario_id", "product_id"]).size() == 1).mean()),
        "sterile_or_biologic_product_frac": float(d["products"].dosage_form.isin(["sterile_injectable", "biologic"]).mean()),
        "shortages": int(len(sh)),
        "shortage_cause_share": sh.cause_type.value_counts(normalize=True).round(4).to_dict(),
        "shortage_duration_median": float(sh.duration.median()) if len(sh) else None,
        "shortage_duration_p90": float(sh.duration.quantile(0.9)) if len(sh) else None,
        "critical_shortage_frac": float(sh.critical.mean()) if len(sh) else None,
        "scenario_fill_rate_median": float(sc.fill_rate.median()),
        "scenarios_without_supply_event": float((sc.n_supply_events == 0).mean()),
        "scenarios_without_shortage": float((sc.n_shortages == 0).mean()),
        "link_day_disrupted_rate": float(ld.label_disrupted.mean()),
        "link_day_impaired_rate": float(ld.label_impaired.mean()),
        "product_day_shortage_rate": float(pdl.label_shortage.mean()),
        "product_day_shortage_next30_rate": float(pdl.label_shortage_next30d.mean()),
        "data_gap_rate_link": float(ld.data_gap.mean()),
        "alerts": int(len(al)), "alerts_false_alarm_frac": float(al.is_false_alarm.mean()),
        "alerts_by_source": al.source.value_counts().to_dict(),
    }


def baselines(d):
    """Naive references so the tasks are demonstrably non-trivial."""
    ld = d["link_daily"].merge(d["links"][["scenario_id", "link_id", "transit_days"]], on=["scenario_id", "link_id"])
    ld = ld.sort_values(["scenario_id", "link_id", "day"])
    g = ld.groupby(["scenario_id", "link_id"])
    ord28 = g.orders_obs.transform(lambda x: x.fillna(0).rolling(28, min_periods=7).mean())
    cover = ld.outstanding_obs / ord28.replace(0, np.nan)          # days of orders still unfilled
    rec28 = g.receipts_obs.transform(lambda x: x.fillna(0).rolling(28, min_periods=7).mean())
    fill = rec28 / ord28.replace(0, np.nan)
    out = {"disruption_detection_auroc_outstanding_cover": auroc(ld.label_disrupted, cover / ld.transit_days),
           "disruption_detection_auroc_receipt_shortfall": auroc(ld.label_disrupted, -fill)}
    pdl = d["product_daily"].sort_values(["scenario_id", "product_id", "day"])
    gp = pdl.groupby(["scenario_id", "product_id"])
    dem28 = gp.demand_obs.transform(lambda x: x.fillna(0).rolling(28, min_periods=7).mean())
    doc = pdl.inventory_obs / dem28.replace(0, np.nan)              # days of cover
    m = ~pdl.label_shortage & ~pdl.label_right_censored60
    out["early_warning30_auroc_days_of_cover"] = auroc(pdl.label_shortage_next30d[m], -doc[m])
    out["early_warning30_positive_rate"] = float(pdl.label_shortage_next30d[m].mean())
    # harder, decision-relevant protocol: predict a NEW shortage 15-60 days ahead while stock still looks healthy
    m2 = m & (doc >= 14)
    out["onset15_60_auroc_days_of_cover_healthy_stock"] = auroc(pdl.label_onset_15_60d[m2], -doc[m2])
    out["onset15_60_positive_rate_healthy_stock"] = float(pdl.label_onset_15_60d[m2].mean())
    return out


def plot_example(d, sid, path):
    ld = d["link_daily"][d["link_daily"].scenario_id == sid]
    pdl = d["product_daily"][d["product_daily"].scenario_id == sid]
    ev = d["events"][(d["events"].scenario_id == sid) & d["events"].in_horizon]
    links = d["links"][d["links"].scenario_id == sid]
    prods = d["products"][d["products"].scenario_id == sid]
    nodes = d["node_daily"][d["node_daily"].scenario_id == sid] if "node_daily" in d else None
    al = d["alerts"][d["alerts"].scenario_id == sid]
    fig, ax = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    for _, e in ev.iterrows():
        end = e.end_day if e.end_day == e.end_day and e.end_day is not None else 730
        end = min(end, 730)
        for a in ax:
            a.axvspan(e.start_day, end, color=COLORS[e.event_type], alpha=0.10)
        ax[0].text(e.start_day, 1.02, e.event_type.replace("_", " "), fontsize=7, rotation=30, color=COLORS[e.event_type])
    if nodes is not None:
        for nid, g in nodes.groupby("node_id"):
            ax[0].plot(g.day, g.true_capacity, lw=1.2, label=nid)
    ax[0].set_ylabel("true node capacity c(t)"); ax[0].set_ylim(-0.05, 1.15); ax[0].legend(fontsize=7, ncol=4, loc="lower left")
    # pick the most disrupted link
    lid = ld.groupby("link_id").label_disrupted.mean().idxmax()
    g = ld[ld.link_id == lid]; L = links[links.link_id == lid].iloc[0]
    pn = prods[prods.product_id == L.product_id].iloc[0]["name"]
    ax[1].plot(g.day, g.orders_obs.rolling(7, min_periods=1).mean(), label="orders placed (7d avg)")
    ax[1].plot(g.day, g.receipts_obs.rolling(7, min_periods=1).mean(), label="receipts (7d avg)")
    ax[1].set_ylabel("units/day"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"buyer view of link {lid}: {pn} from {L.manufacturer} (API from {L.api_supplier})", fontsize=9)
    ax[2].plot(g.day, g.outstanding_obs, label="outstanding orders (observed)")
    ax[2].plot(g.day, g.true_backlog, "--", label="supplier backlog (hidden)")
    ax[2].fill_between(g.day, 0, g.outstanding_obs.max() * g.label_disrupted, color="k", alpha=0.07, label="label_disrupted")
    ax[2].legend(fontsize=8); ax[2].set_ylabel("units")
    pg = pdl[pdl.product_id == L.product_id]
    ax[3].plot(pg.day, pg.inventory_obs, label=f"{pn} inventory (observed)")
    ax3 = ax[3].twinx(); ax3.fill_between(pg.day, 0, pg.unmet_obs, color="r", alpha=0.4, label="unmet demand")
    for _, a in al.iterrows():
        ax[3].axvline(a.day, color="k" if not a.is_false_alarm else "orange", lw=0.5, alpha=0.5)
    ax[3].set_ylabel("units"); ax3.set_ylabel("unmet / day", color="r"); ax[3].legend(fontsize=8, loc="upper left")
    ax[3].set_xlabel("day (vertical ticks = external alerts; orange = false alarms)")
    ax[3].set_xlim(0, 730)
    fig.suptitle(f"PharmaNet-Disrupt scenario {sid}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_distributions(d, path):
    ev, sh = d["events"][d["events"].in_horizon], d["shortages"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ev.event_type.value_counts().plot.barh(ax=ax[0, 0], color=[COLORS[k] for k in ev.event_type.value_counts().index])
    ax[0, 0].set_title("events by type")
    sh.cause_type.value_counts(normalize=True).plot.barh(ax=ax[0, 1], color="#555")
    ax[0, 1].set_title("shortage root cause share (cf. FDA 2019: 63% quality/mfg, 12% demand, 5% disaster, 3% disc.)", fontsize=8)
    sev = ev[ev.category == "supply"].dropna(subset=["capacity_floor"])
    for t, g in sev.groupby("event_type"):
        ax[1, 0].hist(g.end_day.fillna(730) - g.start_day, bins=np.logspace(0.6, 2.9, 30), alpha=0.5, label=t, color=COLORS[t])
    ax[1, 0].set_xscale("log"); ax[1, 0].set_title("disruption cause duration (days, log)"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].hist(sh.duration, bins=np.logspace(0.4, 2.9, 30), color="#444"); ax[1, 1].set_xscale("log")
    ax[1, 1].set_title("shortage episode duration at the buyer (days)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="data")
    ap.add_argument("--report", default="reports")
    a = ap.parse_args()
    os.makedirs(a.report, exist_ok=True)
    meta = json.load(open(os.path.join(a.root, "metadata.json")))
    allstats, inv, base = [], {}, {}
    for split in meta["splits"]:
        cols = {"link_daily": ["scenario_id", "link_id", "day", "orders_obs", "receipts_obs", "outstanding_obs",
                               "data_gap", "true_backlog", "true_in_transit", "true_fg_stock", "true_qc_wip",
                               "true_production", "true_shipments", "true_receipts", "true_site_capacity",
                               "label_disrupted", "label_impaired", "label_planned"]}
        d = load_split(a.root, split, daily=("link_daily", "product_daily", "node_daily"), columns=cols)
        allstats.append(stats(d, split))
        inv[split] = invariants(d)
        if split in ("test", "test_ood_severe", "test_ood_network", "test_ood_noise"):
            base[split] = baselines(d)
        if split == "train":
            plot_distributions(d, os.path.join(a.report, "distributions.png"))
            sc = d["scenarios"].sort_values("n_supply_events")
            picks = [sc[sc.n_supply_events.between(3, 5)].iloc[0].scenario_id,
                     sc[sc.n_shortages > 0].sort_values("fill_rate").iloc[len(sc[sc.n_shortages > 0]) // 3].scenario_id]
            for i, sid in enumerate(picks):
                plot_example(d, sid, os.path.join(a.report, f"example_{i + 1}.png"))
        print(split, "ok", flush=True)
        del d
    rep = dict(stats=allstats, invariants=inv, naive_baselines=base)
    json.dump(rep, open(os.path.join(a.report, "stats.json"), "w"), indent=2, default=str)
    print(json.dumps(inv, indent=1)); print(json.dumps(base, indent=1))


if __name__ == "__main__":
    main()
