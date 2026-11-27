"""
PharmaNet-Disrupt: a physics-grounded, multi-tier pharmaceutical supply-network simulator.

Network (per scenario):
    API suppliers (tier 2)  ->  finished-dose manufacturers (tier 1)  ->  buyer (hospital network / distributor)

Every scenario is a small but complete network: several API supplier nodes, several manufacturing
sites, several drug products, each product multi- or single-sourced. The buyer only sees its own
data (orders, receipts, outstanding orders, inventory, demand, stockouts) plus noisy external
alerts. The hidden truth (capacities, backlogs, events, root causes) is exported as labels.

Physics (all flows are conserved, all states are non-negative):
  * Node capacity follows a generalisation of the PIML capacity ODE:
        dc/dt = -theta_down * alpha(t) * (c - c_floor) + theta_up * (1 - alpha(t)) * (1 - c)
    With c_floor = 0 this is exactly the original paper's equation. c_floor > 0 models partial
    capacity loss (e.g. one line down). alpha(t) = 1 while any disruption on the node is active.
  * Supplier backlog:            db/dt = orders(t) - shipments(t)        (the paper's backlog ODE)
  * API inventory at each site:  dA/dt = API arrivals - API consumption
  * Finished goods / QC WIP / in-transit pipelines are explicit delay lines.
  * Buyer inventory:             dI/dt = receipts - dispensed

Parameter ranges are grounded in public data (see DATASET.md, "Calibration sources").
"""
from __future__ import annotations

import math
import numpy as np

# --------------------------------------------------------------------------------------
# Static domain knowledge
# --------------------------------------------------------------------------------------
REGIONS = ["IN", "CN", "EU", "US", "PR"]
# API site location shares ~ USP Medicine Supply Map (India ~50%, China ~32% of active API DMFs)
API_REGION_P = np.array([0.50, 0.32, 0.10, 0.04, 0.04])
# Finished-dose (FDF) site shares for a US hospital buyer (more US/EU/PR sterile plants)
FDF_REGION_P = np.array([0.35, 0.08, 0.20, 0.27, 0.10])
TRANSIT_DAYS = {"US": (2, 5), "PR": (4, 8), "EU": (7, 14), "IN": (14, 30), "CN": (14, 30)}
API_LEAD_DAYS = (25, 60)          # sea/air freight of API to FDF site
QC_RELEASE_DAYS = {               # batch release testing (sterility test ~14 d, biologics 21-35 d)
    "oral_solid": (5, 12),
    "sterile_injectable": (12, 21),
    "biologic": (21, 35),
}

DRUG_FORM = {
    # anti-infectives
    "cefazolin": "sterile_injectable", "vancomycin": "sterile_injectable",
    "piperacillin-tazobactam": "sterile_injectable", "meropenem": "sterile_injectable",
    "ceftriaxone": "sterile_injectable", "gentamicin": "sterile_injectable",
    "amoxicillin": "oral_solid", "azithromycin": "oral_solid",
    # oncology
    "cisplatin": "sterile_injectable", "carboplatin": "sterile_injectable",
    "methotrexate": "sterile_injectable", "fluorouracil": "sterile_injectable",
    "cytarabine": "sterile_injectable", "doxorubicin": "sterile_injectable",
    # critical care / anaesthesia
    "norepinephrine": "sterile_injectable", "propofol": "sterile_injectable",
    "heparin": "sterile_injectable", "sodium bicarbonate": "sterile_injectable",
    "lidocaine": "sterile_injectable", "dexmedetomidine": "sterile_injectable",
    "potassium chloride": "sterile_injectable",
    # cardiovascular
    "metoprolol": "oral_solid", "amiodarone": "sterile_injectable",
    "diltiazem": "sterile_injectable", "furosemide": "sterile_injectable",
    "atorvastatin": "oral_solid",
    # endocrine
    "insulin glargine": "biologic", "insulin regular": "biologic",
    "levothyroxine": "oral_solid", "hydrocortisone": "sterile_injectable",
    # CNS
    "methylphenidate": "oral_solid", "lorazepam": "sterile_injectable",
    "midazolam": "sterile_injectable", "fentanyl": "sterile_injectable",
    "ketamine": "sterile_injectable",
}

THERAPEUTIC = {
    "anti_infective": dict(drugs=["cefazolin", "vancomycin", "piperacillin-tazobactam", "meropenem",
                                  "ceftriaxone", "gentamicin", "amoxicillin", "azithromycin"],
                           season=(0.15, 0.50), weekend=(0.90, 1.00), crit_p=0.6, surge_rate=0.20),
    "oncology": dict(drugs=["cisplatin", "carboplatin", "methotrexate", "fluorouracil", "cytarabine",
                            "doxorubicin"],
                     season=(0.00, 0.05), weekend=(0.45, 0.75), crit_p=0.85, surge_rate=0.03),
    "critical_care": dict(drugs=["norepinephrine", "propofol", "heparin", "sodium bicarbonate",
                                 "lidocaine", "dexmedetomidine", "potassium chloride"],
                          season=(0.02, 0.15), weekend=(0.90, 1.00), crit_p=0.9, surge_rate=0.10),
    "cardiovascular": dict(drugs=["metoprolol", "amiodarone", "diltiazem", "furosemide", "atorvastatin"],
                           season=(0.00, 0.08), weekend=(0.80, 1.00), crit_p=0.4, surge_rate=0.03),
    "endocrine": dict(drugs=["insulin glargine", "insulin regular", "levothyroxine", "hydrocortisone"],
                      season=(0.00, 0.04), weekend=(0.90, 1.00), crit_p=0.75, surge_rate=0.03),
    "cns": dict(drugs=["methylphenidate", "lorazepam", "midazolam", "fentanyl", "ketamine"],
                season=(0.00, 0.06), weekend=(0.75, 1.00), crit_p=0.5, surge_rate=0.06),
}
DRUG_CLASS = {d: c for c, v in THERAPEUTIC.items() for d in v["drugs"]}

# Event catalogue. Rates are per node (or lane / link / class) per year, before the scenario
# risk multiplier. Mix calibrated so manufacturing/quality problems dominate shortage causes
# (FDA Drug Shortages Task Force 2019: 63% quality/manufacturing, 12% demand, 5% disaster,
# 3% discontinuation).
EVENT_TYPES = {
    "quality_failure":      dict(category="supply", tier="manufacturer", rate_sterile=0.20, rate_oral=0.07,
                                 floor=(0.0, 0.6), theta_down=(0.2, 1.0), theta_up=(0.008, 0.05),
                                 dur=(21, 365), reject=(0.10, 0.60)),
    "regulatory_action":    dict(category="supply", tier="manufacturer", rate=0.04,
                                 floor=(0.0, 0.3), theta_down=(0.5, 1.0), theta_up=(0.03, 0.15),
                                 dur=(60, 720)),
    "api_shortage":         dict(category="supply", tier="api_supplier", rate=0.12,
                                 floor=(0.0, 0.5), theta_down=(0.1, 0.6), theta_up=(0.02, 0.08),
                                 dur=(21, 365)),
    "natural_disaster":     dict(category="supply", tier="region", rate=0.03,
                                 floor=(0.0, 0.4), theta_down=(0.5, 1.0), theta_up=(0.01, 0.06),
                                 dur=(5, 180)),
    "logistics_disruption": dict(category="supply", tier="lane", rate=0.08,
                                 lt_mult=(1.5, 3.0), loss=(0.0, 0.30), dur=(5, 90)),
    "discontinuation":      dict(category="supply", tier="product_line", rate=0.012,
                                 notice=(0, 180)),
    "demand_surge":         dict(category="demand", tier="therapeutic_class",
                                 mult=(1.4, 3.5), dur=(21, 150)),
    "planned_maintenance":  dict(category="planned", tier="manufacturer", rate=0.5,
                                 floor=(0.0, 0.2), dur=(7, 21), prebuild=(20, 45)),
}
SUPPLY_TYPES = [k for k, v in EVENT_TYPES.items() if v["category"] == "supply"]

# Regimes = data splits with different generative settings (IID train/val/test + OOD tests)
REGIMES = {
    "iid":        dict(n_products=(4, 10), n_mfr=(2, 5), n_api=(2, 4), rate_mult=1.0, sev_shift=0.0,
                       noise_mult=1.0, qc_mult=1.0),
    "ood_severe": dict(n_products=(4, 10), n_mfr=(2, 5), n_api=(2, 4), rate_mult=2.0, sev_shift=0.25,
                       noise_mult=1.0, qc_mult=1.4),
    "ood_network": dict(n_products=(11, 16), n_mfr=(6, 8), n_api=(4, 6), rate_mult=1.0, sev_shift=0.0,
                        noise_mult=1.0, qc_mult=1.0),
    "ood_noise":  dict(n_products=(4, 10), n_mfr=(2, 5), n_api=(2, 4), rate_mult=1.0, sev_shift=0.0,
                       noise_mult=3.0, qc_mult=1.0),
}

T_DAYS = 730      # exported horizon (2 years, daily)
BURN_IN = 180     # discarded warm-up so pipelines are in steady state
MAX_LAG = 200


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def _logu(rng, lo_hi):
    """Log-uniform duration: many short events with a heavy tail (as observed for shortages)."""
    return float(np.exp(rng.uniform(math.log(lo_hi[0]), math.log(lo_hi[1]))))


def _sev(rng, lo_hi, shift):
    """Sample a floor (residual capacity) and push it down by `shift` for harsher regimes."""
    return float(np.clip(rng.uniform(*lo_hi) - shift, 0.0, 1.0))


# --------------------------------------------------------------------------------------
# Scenario sampling (network + events). Returns a JSON-serialisable dict.
# --------------------------------------------------------------------------------------
def sample_scenario(seed: int, regime: str = "iid", T: int = T_DAYS) -> dict:
    R = REGIMES[regime]
    rng = np.random.default_rng(seed)

    # ---------- products ----------
    nP = int(rng.integers(R["n_products"][0], R["n_products"][1] + 1))
    all_drugs = list(DRUG_FORM.keys())
    drugs = list(rng.choice(all_drugs, size=min(nP, len(all_drugs)), replace=False))
    nP = len(drugs)
    # molecules (API): mostly one per product, a few shared (e.g. same API, two presentations)
    n_mol = max(1, int(math.ceil(nP * rng.uniform(0.6, 1.0))))
    prod_mol = [i if i < n_mol else int(rng.integers(0, n_mol)) for i in range(nP)]
    rng.shuffle(prod_mol)

    products = []
    for i, d in enumerate(drugs):
        cls = DRUG_CLASS[d]
        C = THERAPEUTIC[cls]
        form = DRUG_FORM[d]
        critical = bool(rng.random() < C["crit_p"])
        products.append(dict(
            product_id=i, name=d, therapeutic_class=cls, dosage_form=form,
            critical=critical, cold_chain=(form == "biologic"),
            molecule_id=int(prod_mol[i]),
            base_demand=float(np.exp(rng.uniform(math.log(40), math.log(4000)))),  # units/day
            season_amp=_u(rng, C["season"]),
            season_peak_doy=float(rng.normal(20, 20)) % 365,
            weekend_factor=_u(rng, C["weekend"]),
            trend_per_year=float(rng.uniform(-0.05, 0.10)),
            demand_cv=float(rng.uniform(0.05, 0.25)),
            safety_days=float(rng.uniform(30, 60) if critical else rng.uniform(10, 30)),
            review_period=int(1 if critical else rng.choice([1, 7])),
            hoard=float(rng.uniform(0.0, 0.5)),
        ))

    # ---------- manufacturers (FDF sites) ----------
    nM = int(rng.integers(R["n_mfr"][0], R["n_mfr"][1] + 1))
    mfrs = []
    for m in range(nM):
        reg = REGIONS[int(rng.choice(len(REGIONS), p=FDF_REGION_P))]
        mfrs.append(dict(node_id=f"MFR-{reg}-{m:02d}", idx=m, tier="manufacturer", region=reg,
                         headroom=float(rng.uniform(0.05, 0.40)),       # spare capacity (generic injectables run tight)
                         surge_max=float(rng.uniform(0.10, 0.60)),
                         ramp_days=float(rng.uniform(60, 180)),
                         fg_days=float(rng.uniform(7, 30)),
                         api_stock_days=float(rng.uniform(30, 90)),
                         base_reject=float(rng.uniform(0.0, 0.02))))

    # ---------- sourcing: which manufacturers make which product ----------
    # ~48% of new US shortages involve a single manufacturer -> sample n_sources accordingly
    links = []
    for p in products:
        k = int(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20]))
        k = min(k, nM)
        srcs = sorted(rng.choice(nM, size=k, replace=False).tolist())
        w = rng.dirichlet(np.ones(k) * 2.0)
        for j, m in enumerate(srcs):
            form = p["dosage_form"]
            qc = QC_RELEASE_DAYS[form]
            links.append(dict(link_id=len(links), mfr=m, product=p["product_id"], share0=float(w[j]),
                              qc_days=int(round(rng.uniform(*qc) * R["qc_mult"])),
                              batch_interval=int(rng.choice([3, 5, 7, 10, 14])),
                              batch_phase=int(rng.integers(0, 14)),
                              transit_days=int(rng.integers(*TRANSIT_DAYS[mfrs[m]["region"]]) + 1)))

    # ---------- API suppliers ----------
    nA = int(rng.integers(R["n_api"][0], R["n_api"][1] + 1))
    apis = []
    for a in range(nA):
        reg = REGIONS[int(rng.choice(len(REGIONS), p=API_REGION_P))]
        apis.append(dict(node_id=f"API-{reg}-{a:02d}", idx=a, tier="api_supplier", region=reg,
                         headroom=float(rng.uniform(0.05, 0.35)),
                         surge_max=float(rng.uniform(0.3, 0.8)),        # campaign / extra shifts after a shortage
                         ramp_days=float(rng.uniform(45, 120))))
    # each molecule offered by 1-2 API nodes; each (mfr, molecule) single-sources from one node
    mol_nodes = {mo: sorted(rng.choice(nA, size=int(min(nA, rng.choice([1, 2], p=[0.6, 0.4]))),
                                       replace=False).tolist()) for mo in range(n_mol)}
    mm = {}
    for L in links:
        mo = products[L["product"]]["molecule_id"]
        key = (L["mfr"], mo)
        if key not in mm:
            mm[key] = dict(mm_id=len(mm), mfr=L["mfr"], molecule=mo,
                           api_node=int(rng.choice(mol_nodes[mo])),
                           lead_days=int(rng.integers(*API_LEAD_DAYS)))
    mm_list = list(mm.values())

    # ---------- events ----------
    risk = float(np.exp(rng.normal(0, 0.7))) * R["rate_mult"]   # scenario-level risk heterogeneity
    years = T / 365.0
    shift = R["sev_shift"]
    events = []

    def add(ev):
        ev["event_id"] = len(events)
        events.append(ev)

    def n_events(rate):
        return int(rng.poisson(rate * years * risk))

    def start_day():
        return int(rng.integers(0, T))

    E = EVENT_TYPES
    for m in mfrs:
        forms = {products[L["product"]]["dosage_form"] for L in links if L["mfr"] == m["idx"]}
        sterile = bool(forms & {"sterile_injectable", "biologic"})
        q = E["quality_failure"]
        for _ in range(n_events(q["rate_sterile"] if sterile else q["rate_oral"])):
            s = start_day()
            add(dict(type="quality_failure", category="supply", tier="manufacturer", node=m["node_id"],
                     start=s, end=s + int(_logu(rng, q["dur"])),
                     floor=_sev(rng, q["floor"], shift), theta_down=_u(rng, q["theta_down"]),
                     theta_up=_u(rng, q["theta_up"]), reject=_u(rng, q["reject"])))
        r = E["regulatory_action"]
        for _ in range(n_events(r["rate"])):
            s = start_day()
            add(dict(type="regulatory_action", category="supply", tier="manufacturer", node=m["node_id"],
                     start=s, end=s + int(_logu(rng, r["dur"])),
                     floor=_sev(rng, r["floor"], shift), theta_down=_u(rng, r["theta_down"]),
                     theta_up=_u(rng, r["theta_up"])))
        g = E["logistics_disruption"]
        for _ in range(n_events(g["rate"])):
            s = start_day()
            add(dict(type="logistics_disruption", category="supply", tier="lane", node=m["node_id"],
                     start=s, end=s + int(_logu(rng, g["dur"])),
                     lt_mult=_u(rng, g["lt_mult"]), loss=_u(rng, g["loss"])))
        pm = E["planned_maintenance"]
        for _ in range(int(rng.poisson(pm["rate"] * years))):   # planned work is not risk-scaled
            s = start_day()
            add(dict(type="planned_maintenance", category="planned", tier="manufacturer", node=m["node_id"],
                     start=s, end=s + int(rng.uniform(*pm["dur"])),
                     floor=_u(rng, pm["floor"]), theta_down=1.0, theta_up=0.3,
                     prebuild=int(rng.uniform(*pm["prebuild"]))))
    for a in apis:
        ap = E["api_shortage"]
        for _ in range(n_events(ap["rate"])):
            s = start_day()
            add(dict(type="api_shortage", category="supply", tier="api_supplier", node=a["node_id"],
                     start=s, end=s + int(_logu(rng, ap["dur"])),
                     floor=_sev(rng, ap["floor"], shift), theta_down=_u(rng, ap["theta_down"]),
                     theta_up=_u(rng, ap["theta_up"])))
    present_regions = sorted({n["region"] for n in mfrs + apis})
    for reg in present_regions:
        nd = E["natural_disaster"]
        for _ in range(n_events(nd["rate"])):
            s = start_day()
            add(dict(type="natural_disaster", category="supply", tier="region", node=f"REGION-{reg}",
                     start=s, end=s + int(_logu(rng, nd["dur"])),
                     floor=_sev(rng, nd["floor"], shift), theta_down=_u(rng, nd["theta_down"]),
                     theta_up=_u(rng, nd["theta_up"])))
    for L in links:
        dc = E["discontinuation"]
        if rng.random() < 1 - math.exp(-dc["rate"] * years * risk):
            s = start_day()
            add(dict(type="discontinuation", category="supply", tier="product_line",
                     node=f"LINK-{L['link_id']}", start=s, end=None,
                     notice_days=int(rng.uniform(*dc["notice"]))))
    for cls in sorted({p["therapeutic_class"] for p in products}):
        ds = E["demand_surge"]
        for _ in range(int(rng.poisson(THERAPEUTIC[cls]["surge_rate"] * years * max(risk, 0.5)))):
            s = start_day()
            add(dict(type="demand_surge", category="demand", tier="therapeutic_class", node=f"CLASS-{cls}",
                     start=s, end=s + int(_logu(rng, ds["dur"])), mult=_u(rng, ds["mult"])))

    start_date = (np.datetime64("2018-01-01") + int(rng.integers(0, 6 * 365))).astype(str)
    return dict(seed=int(seed), regime=regime, T=int(T), start_date=start_date, risk=risk,
                noise_mult=R["noise_mult"], products=products, manufacturers=mfrs, api_suppliers=apis,
                links=links, mfr_molecule=mm_list, events=events)


# --------------------------------------------------------------------------------------
# Exogenous trajectories (capacities, demand) — independent of the flow state
# --------------------------------------------------------------------------------------
def _capacity_path(n_days, node_events, burn):
    """Integrate the generalised capacity ODE for one node given its events (day units)."""
    floor = np.ones(n_days)
    thd = np.zeros(n_days)
    active = np.zeros(n_days, bool)
    for ev in node_events:
        s = ev["start"] + burn
        e = n_days if ev.get("end") is None else min(n_days, ev["end"] + burn)
        if s >= n_days:
            continue
        floor[s:e] = np.minimum(floor[s:e], ev["floor"])
        thd[s:e] = np.maximum(thd[s:e], ev["theta_down"])
        active[s:e] = True
    # recovery rate in force after each event ends = that event's theta_up
    thu = np.full(n_days, 0.1)
    for ev in sorted(node_events, key=lambda e: e["start"]):
        if ev.get("end") is not None:
            thu[min(n_days, ev["end"] + burn):] = ev["theta_up"]
    c = np.ones(n_days)
    for t in range(1, n_days):
        if active[t - 1]:
            dc = -thd[t - 1] * (c[t - 1] - floor[t - 1])
        else:
            dc = thu[t - 1] * (1.0 - c[t - 1])
        c[t] = min(1.0, max(0.0, c[t - 1] + dc))
    return c, active


def _demand_paths(sc, rng, n_days, burn):
    P = sc["products"]
    days = np.arange(n_days) - burn
    start = np.datetime64(sc["start_date"])
    dates = start + days
    doy = ((dates - dates.astype("datetime64[Y]")).astype(int)).astype(float)
    dow = ((dates.astype("datetime64[D]").view("int64") - 4) % 7)  # 0=Mon
    D = np.zeros((len(P), n_days))
    mult = np.ones((len(P), n_days))
    for ev in sc["events"]:
        if ev["type"] != "demand_surge":
            continue
        cls = ev["node"].replace("CLASS-", "")
        s, e = ev["start"] + burn, min(n_days, ev["end"] + burn)
        L = e - s
        if L <= 0:
            continue
        x = np.arange(L) / L
        shape = np.minimum(1.0, x / 0.2) * np.exp(-2.0 * np.maximum(0, x - 0.4))  # ramp, plateau, decay
        for i, p in enumerate(P):
            if p["therapeutic_class"] == cls:
                mult[i, s:e] *= 1.0 + (ev["mult"] - 1.0) * shape
    for i, p in enumerate(P):
        season = 1.0 + p["season_amp"] * np.cos(2 * np.pi * (doy - p["season_peak_doy"]) / 365.0)
        trend = (1.0 + p["trend_per_year"]) ** (days / 365.0)
        weekly = np.where(dow >= 5, p["weekend_factor"], 1.0)
        mean = p["base_demand"] * season * trend * weekly * mult[i]
        k = 1.0 / p["demand_cv"] ** 2
        D[i] = rng.gamma(k, mean / k)
    return D, mult, dow


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------
def simulate(sc: dict, policy=None) -> dict:
    """Run the network. `policy(obs_dict) -> shares array (n_links,)` optionally overrides the
    buyer's sourcing shares weekly (hook for closed-loop evaluation of agents)."""
    rng = np.random.default_rng(sc["seed"] + 7919)
    burn, T = BURN_IN, sc["T"]
    N = burn + T
    H = N + MAX_LAG
    P, M, A, LK, MM = sc["products"], sc["manufacturers"], sc["api_suppliers"], sc["links"], sc["mfr_molecule"]
    nP, nM, nA, nL, nMM = len(P), len(M), len(A), len(LK), len(MM)
    ev = sc["events"]
    mfr_by_id = {m["node_id"]: m["idx"] for m in M}
    api_by_id = {a["node_id"]: a["idx"] for a in A}

    # ---- exogenous capacities ----
    def node_events(node_id, region):
        out = [e for e in ev if e["node"] == node_id and "floor" in e]
        out += [e for e in ev if e["type"] == "natural_disaster" and e["node"] == f"REGION-{region}"]
        return out
    c_m = np.zeros((nM, N)); act_m = np.zeros((nM, N), bool)
    for m in M:
        c_m[m["idx"]], act_m[m["idx"]] = _capacity_path(N, node_events(m["node_id"], m["region"]), burn)
    c_a = np.zeros((nA, N)); act_a = np.zeros((nA, N), bool)
    for a in A:
        c_a[a["idx"]], act_a[a["idx"]] = _capacity_path(N, node_events(a["node_id"], a["region"]), burn)

    reject = np.array([[m["base_reject"]] * N for m in M])
    lt_mult = np.ones((nM, N)); loss = np.zeros((nM, N))
    line = np.ones((nL, N)); disc_day = np.full(nL, -1)
    fg_mult = np.ones((nL, N))
    for e in ev:
        s = e["start"] + burn
        en = N if e.get("end") is None else min(N, e["end"] + burn)
        if e["type"] == "quality_failure":
            m = mfr_by_id[e["node"]]
            reject[m, max(0, s - 10):en] = np.maximum(reject[m, max(0, s - 10):en], e["reject"])  # failing batches precede the shutdown
        elif e["type"] == "logistics_disruption":
            m = mfr_by_id[e["node"]]
            lt_mult[m, s:en] = np.maximum(lt_mult[m, s:en], e["lt_mult"])
            loss[m, s:en] = np.maximum(loss[m, s:en], e["loss"])
        elif e["type"] == "discontinuation":
            l = int(e["node"].split("-")[1])
            line[l, s:] = 0.0
            disc_day[l] = s
        elif e["type"] == "planned_maintenance":
            m = mfr_by_id[e["node"]]
            ps = max(0, s - e["prebuild"])
            for L in LK:
                if L["mfr"] == m:
                    fg_mult[L["link_id"], ps:s] = 1.0 + (e["end"] - e["start"]) / max(1.0, M[m]["fg_days"])

    D, dmult, dow = _demand_paths(sc, rng, N, burn)

    # ---- static link arrays ----
    lm = np.array([L["mfr"] for L in LK]); lp = np.array([L["product"] for L in LK])
    share0 = np.array([L["share0"] for L in LK])
    qc = np.array([L["qc_days"] for L in LK]); bint = np.array([L["batch_interval"] for L in LK])
    bph = np.array([L["batch_phase"] for L in LK]); tr = np.array([L["transit_days"] for L in LK])
    cold = np.array([P[p]["cold_chain"] for p in lp])
    nominal = np.array([P[lp[l]]["base_demand"] * share0[l] for l in range(nL)])
    # plants size lines for seasonal peak + expected growth over the horizon, plus spare headroom
    peak = np.array([(1.0 + P[lp[l]]["season_amp"]) * (1.0 + max(0.0, P[lp[l]]["trend_per_year"])) ** (N / 365.0)
                     for l in range(nL)])
    Kcap = nominal * peak * (1.0 + np.array([M[m]["headroom"] for m in lm]))
    surge_max = np.array([M[m]["surge_max"] for m in lm]); ramp = np.array([M[m]["ramp_days"] for m in lm])
    fg_days = np.array([M[m]["fg_days"] for m in lm])
    link_mm = np.zeros(nL, int)
    mm_index = {(x["mfr"], x["molecule"]): x["mm_id"] for x in MM}
    for l in range(nL):
        link_mm[l] = mm_index[(lm[l], P[lp[l]]["molecule_id"])]
    mm_need = np.zeros(nMM)
    np.add.at(mm_need, link_mm, nominal * peak)
    mm_src = np.array([x["api_node"] for x in MM]); mm_lead = np.array([x["lead_days"] for x in MM])
    mm_stock_days = np.array([M[x["mfr"]]["api_stock_days"] for x in MM])
    Kapi = np.zeros(nA)
    np.add.at(Kapi, mm_src, mm_need)
    Kapi *= 1.0 + np.array([a["headroom"] for a in A])
    Kapi = np.maximum(Kapi, 1e-6)
    api_surge_max = np.array([a["surge_max"] for a in A]); api_ramp = np.array([a["ramp_days"] for a in A])
    api_surge = np.zeros(nA)

    base = np.array([p["base_demand"] for p in P])
    Lhat = np.zeros(nP); np.add.at(Lhat, lp, share0 * (tr + 3))
    safety = np.array([p["safety_days"] for p in P]); Rv = np.array([p["review_period"] for p in P])
    hoard = np.array([p["hoard"] for p in P])

    # ---- state ----
    I = base * (Lhat + safety)                         # buyer inventory
    FG = nominal * fg_days                             # supplier finished goods
    b = np.zeros(nL)                                   # supplier backlog (open orders not shipped)
    WIP = np.zeros(nL)
    Aapi = mm_need * mm_stock_days                     # API on hand at each (site, molecule)
    api_pipe = np.zeros(nMM)
    api_arr = np.zeros((nMM, H))
    for k in range(nMM):                               # pipeline starts full (steady state)
        api_arr[k, :mm_lead[k]] = mm_need[k] / peak.mean()
        api_pipe[k] = mm_need[k] / peak.mean() * mm_lead[k]
    outstanding = np.zeros(nL)
    fc_ord = nominal.copy()
    fc_dem = base.copy()
    share = share0.copy()
    surge = np.zeros(nL)
    rel_sched = np.zeros((nL, H)); arr_good = np.zeros((nL, H)); arr_dam = np.zeros((nL, H))
    last_stockout = np.full(nP, -10**6)

    rec = {k: np.zeros((nL, N)) for k in ["orders", "receipts", "damaged", "cancelled", "outstanding",
                                          "backlog", "fg", "wip", "production", "desired", "shipments",
                                          "cap_eff", "api_scale", "share", "in_transit", "rejected"]}
    prec = {k: np.zeros((nP, N)) for k in ["demand", "dispensed", "unmet", "inventory"]}
    mm_rec = {k: np.zeros((nMM, N)) for k in ["api_on_hand", "api_arrivals", "api_consumed"]}
    in_transit = np.zeros(nL)

    for t in range(N):
        weekday = dow[t] < 5
        # ---------------- buyer: receive ----------------
        g, d_ = arr_good[:, t], arr_dam[:, t]
        np.add.at(I, lp, g)
        outstanding -= g + d_
        in_transit -= g + d_
        # ---------------- buyer: dispense ----------------
        disp = np.minimum(I, D[:, t]); unmet = D[:, t] - disp
        I -= disp
        last_stockout[unmet > 1e-9] = t
        fc_dem += (D[:, t] - fc_dem) / 28.0
        # ---------------- discontinuation: supplier cancels open orders ----------------
        canc = np.zeros(nL)
        newly = disc_day == t
        if newly.any():
            canc[newly] = b[newly]
            outstanding[newly] -= b[newly]
            b[newly] = 0.0
        # ---------------- buyer: sourcing shares (weekly) ----------------
        if t % 7 == 0:
            # supplier reliability = days of (normal-volume) orders still unfilled; uses the link's
            # nominal volume so a supplier that is no longer ordered from can be re-qualified once
            # its outstanding orders clear (no lock-out)
            ref = np.maximum(share0 * fc_dem[lp], 1e-6)
            cover = outstanding / ref
            norm = tr + 3.0
            rel = np.exp(-1.5 * np.maximum(0.0, cover - 2.0 * norm) / norm)
            target = share0 * rel * (line[:, t] > 0)
            ssum = np.zeros(nP); np.add.at(ssum, lp, target)
            target = np.where(ssum[lp] > 1e-9, target / np.maximum(ssum[lp], 1e-9), share0 * (line[:, t] > 0))
            share += (target - share) * 0.35                   # buyer reacts with a lag (~3 weeks)
            if policy is not None:
                share = np.asarray(policy(dict(t=t - burn, outstanding=outstanding.copy(), inventory=I.copy(),
                                               share=share.copy())), float)
        # ---------------- buyer: order ----------------
        review = (t % Rv) == 0
        IP = I.copy(); np.add.at(IP, lp, outstanding)
        recent_so = (t - last_stockout) < 14
        S = fc_dem * (Lhat + Rv + safety) * (1.0 + hoard * recent_so)
        Q = np.where(review, np.maximum(0.0, S - IP), 0.0)
        Q = np.minimum(Q, fc_dem * (Rv + 21))              # wholesaler order caps limit panic buying
        q = Q[lp] * share * (line[:, t] > 0)
        outstanding += q; b += q
        fc_ord += (q - fc_ord) / 28.0
        # ---------------- API suppliers ship to sites ----------------
        arr_api = api_arr[:, t]
        Aapi += arr_api; api_pipe -= arr_api
        # API need follows what the site can physically make, not the buyer's (possibly panic) orders
        use_fc = np.zeros(nMM); np.add.at(use_fc, link_mm, np.minimum(fc_ord, Kcap * (1.0 + surge)))
        # base-stock on API inventory position (on hand + in transit)
        req = np.maximum(0.0, use_fc + (use_fc * (mm_stock_days + mm_lead) - Aapi - api_pipe) / 60.0)
        tot_req = np.zeros(nA); np.add.at(tot_req, mm_src, req)
        api_surge += (np.where(tot_req > Kapi * c_a[:, t], api_surge_max, 0.0) - api_surge) / api_ramp
        cap_api = Kapi * (1.0 + api_surge) * c_a[:, t]
        frac = np.minimum(1.0, cap_api / np.maximum(tot_req, 1e-9))
        api_ship = req * frac[mm_src]
        for k in range(nMM):
            api_arr[k, t + mm_lead[k]] += api_ship[k]
        api_pipe += api_ship
        # ---------------- manufacturer production ----------------
        tgt_surge = np.where(b > 14 * fc_ord, surge_max, 0.0)
        surge += (tgt_surge - surge) / ramp
        cap = Kcap * (1.0 + surge) * c_m[lm, t] * line[:, t]
        desired = np.maximum(0.0, fc_ord + (fc_ord * fg_days * fg_mult[:, t] + b - FG - WIP) / 10.0)
        prod = np.minimum(desired, cap)
        need = np.zeros(nMM); np.add.at(need, link_mm, prod)
        scale = np.minimum(1.0, Aapi / np.maximum(need, 1e-9))
        prod = prod * scale[link_mm]
        used = np.zeros(nMM); np.add.at(used, link_mm, prod)
        Aapi = np.maximum(0.0, Aapi - used)
        rj = prod * reject[lm, t]
        good = prod - rj
        rd = t + qc
        rd = rd + (bint - (rd - bph) % bint) % bint                  # next batch-release day
        rel_sched[np.arange(nL), rd] += good
        WIP += good
        rel = rel_sched[:, t]
        FG += rel; WIP -= rel
        # ---------------- ship ----------------
        ship = np.minimum(FG, b) if weekday else np.zeros(nL)
        FG -= ship; b -= ship
        lt = np.maximum(1, np.round(tr * lt_mult[lm, t])).astype(int)
        ad = t + lt
        wd = dow[np.minimum(ad, N - 1)] if True else None
        ad = ad + np.where(ad < N, np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), 0)   # no weekend deliveries
        lf = loss[lm, t] * np.where(cold, 1.0, 0.2)
        arr_good[np.arange(nL), ad] += ship * (1 - lf)
        arr_dam[np.arange(nL), ad] += ship * lf
        in_transit += ship
        # ---------------- record ----------------
        for k, v in (("orders", q), ("receipts", g), ("damaged", d_), ("cancelled", canc),
                     ("outstanding", outstanding), ("backlog", b), ("fg", FG), ("wip", WIP),
                     ("production", prod), ("desired", desired), ("shipments", ship),
                     ("cap_eff", cap * scale[link_mm] / np.maximum(Kcap, 1e-9)), ("api_scale", scale[link_mm]),
                     ("share", share), ("in_transit", in_transit), ("rejected", rj)):
            rec[k][:, t] = v
        prec["demand"][:, t] = D[:, t]; prec["dispensed"][:, t] = disp
        prec["unmet"][:, t] = unmet; prec["inventory"][:, t] = I
        mm_rec["api_on_hand"][:, t] = Aapi; mm_rec["api_arrivals"][:, t] = arr_api; mm_rec["api_consumed"][:, t] = used

    sl = slice(burn, N)
    return dict(
        link={k: v[:, sl] for k, v in rec.items()},
        product={k: v[:, sl] for k, v in prec.items()},
        mfr_molecule={k: v[:, sl] for k, v in mm_rec.items()},
        c_mfr=c_m[:, sl], active_mfr=act_m[:, sl], c_api=c_a[:, sl], active_api=act_a[:, sl],
        reject=reject[:, sl], lt_mult=lt_mult[:, sl], loss=loss[:, sl], line=line[:, sl],
        demand_mult=dmult[:, sl], dow=dow[sl], Kcap=Kcap, nominal=nominal,
    )
