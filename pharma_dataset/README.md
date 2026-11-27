# PharmaNet-Disrupt v1.0

A physics-grounded, multi-tier pharmaceutical supply-network dataset for **disruption detection,
root-cause localisation, shortage early warning and mitigation**, designed for evaluating
multi-agent and physics-informed ML systems.

- 3,400 independent supply networks (2 years of daily data each)
- 25,099 drug products, 42,803 supplier-product links, 31.2M link-days, 18.3M product-days
- 32,614 ground-truth events (8 types), 35,219 shortage episodes with root-cause labels
- 66,339 external text alerts (FDA-style notices, news, supplier notices, rumours)
- 1 IID split (train / val / test) + 3 out-of-distribution test sets
- Fully deterministic and regenerable (`python generate.py`, ~4 min on 2 CPU cores)

---

## 1. Why this dataset

The PIML paper this project builds on infers *when one supplier was disrupted* from the buyer's
backlog. Real pharmaceutical shortages are network problems:

- products are **multi-sourced or single-sourced**; when one manufacturer fails, others can only
  partly cover because spare capacity is limited;
- manufacturers depend on **upstream API (active ingredient) suppliers**, which are geographically
  concentrated, so one upstream failure hits several products at once (*common cause*);
- shortages have **different root causes** (quality failures, regulatory action, API shortages,
  natural disasters, discontinuations, demand surges), with different signatures;
- the buyer only sees **its own data** plus **noisy, delayed public information**.

PharmaNet-Disrupt models all of these so that a multi-agent system can be evaluated on the full
chain: *detect -> localise -> diagnose -> forecast impact -> act*.

## 2. Network structure (per scenario)

```
API suppliers (tier 2)  --API, 25-60 d freight-->  Manufacturers / FDF sites (tier 1)
                                                        | production -> QC release (batches) -> finished goods
                                                        v  transit 2-30 d
                                               Buyer (hospital network / distributor)
                                                        | dispense to patients
```

| Element | Range per scenario (IID) | Notes |
|---|---|---|
| Products (drugs) | 4-10 | 35 real generic molecules in 6 therapeutic classes |
| Manufacturers (FDF sites) | 2-5 | regions IN / CN / EU / US / PR |
| API suppliers | 2-4 | mostly India / China |
| Sources per product | 1-3 | 45% single-source |
| Horizon | 730 days | + 180-day discarded burn-in |

Product attributes: therapeutic class (anti-infective, oncology, critical care, cardiovascular,
endocrine, CNS), dosage form (sterile injectable, oral solid, biologic), criticality, cold chain,
seasonality (e.g. winter peak for anti-infectives), weekday pattern, demand growth and noise,
the buyer's safety-stock policy, and hoarding tendency.

## 3. Physics (what is conserved)

All states are non-negative and all flows are conserved (checked by `validate.py`).

**Node capacity** (every manufacturer and API supplier) follows a generalisation of the PIML
capacity ODE:

```
dc/dt = -theta_down * alpha(t) * (c - c_floor) + theta_up * (1 - alpha(t)) * (1 - c)
alpha(t) = 1 while any disruption on the node is active, else 0
```

With `c_floor = 0` this is exactly the original paper's equation. `c_floor > 0` models partial loss
(e.g. one filling line down). Overlapping events take the lowest floor and fastest drop.

**Supplier backlog** (the paper's backlog ODE, per link): `db/dt = orders(t) - shipments(t)`

**Other stocks** (explicit delay lines): API inventory at each site, QC work-in-progress
(batch release after 5-35 days), finished goods, in-transit, buyer inventory
(`dI/dt = receipts - dispensed`).

**Behaviour**
- Manufacturers produce to a finished-goods target, limited by capacity x API availability, and
  can ramp spare capacity (+10-60% over 60-180 days) when backlog persists.
- API suppliers allocate scarce API across sites pro rata and can also surge (+30-80%).
- The buyer runs an order-up-to policy (daily or weekly review), shifts sourcing shares away from
  suppliers whose orders are going unfilled (and back once they recover), hoards during
  shortages, and is limited by wholesaler order caps (bullwhip effect included).

## 4. Event catalogue (ground truth)

| Event type | Category | Hits | Key parameters (IID ranges) |
|---|---|---|---|
| `quality_failure` | supply | manufacturer site | rate 0.20/yr (sterile), 0.07/yr (oral); duration 21-365 d (log-uniform); residual capacity 0-60%; QC reject rate 10-60% starting 10 d before shutdown |
| `regulatory_action` | supply | manufacturer site | 0.04/yr; 60-720 d; residual 0-30% (warning letter / import alert) |
| `api_shortage` | supply | API supplier -> every site and product using it | 0.12/yr; 21-365 d; residual 0-50%; impact delayed by site API buffers |
| `natural_disaster` | supply | every node in a region | 0.03/yr per region; 5-180 d; residual 0-40% |
| `logistics_disruption` | supply | manufacturer -> buyer lane | 0.08/yr; 5-90 d; lead time x1.5-3; transit loss 0-30% (full for cold chain) |
| `discontinuation` | supply | one product line, permanent | 0.012/yr per link; open orders cancelled; notice 0-180 d |
| `demand_surge` | demand (confounder) | therapeutic class | 0.03-0.20/yr; 21-150 d; x1.4-3.5 demand |
| `planned_maintenance` | planned (confounder) | manufacturer site | 0.5/yr; 7-21 d; inventory pre-built 20-45 d ahead |

A scenario-level risk multiplier (log-normal, sigma 0.7) creates calm and turbulent networks;
about 9% of networks have no supply event at all.

**Calibration against real data (train split)**

| Statistic | PharmaNet-Disrupt | Real-world reference |
|---|---|---|
| Shortage root cause: manufacturing / quality (quality + regulatory + API) | 62% | 63% (FDA Drug Shortages Task Force, 2019) |
| Shortage root cause: demand increase | 16% | 12% (FDA 2019) |
| Shortage root cause: natural disaster | 9% | 5% (FDA 2019) |
| Shortage root cause: discontinuation | 5% | 3% (FDA 2019) |
| Products with a single manufacturer | 45% | 48% of new shortages (AMA / ASHP 2026) |
| Sterile injectable or biologic products | 83% | hospital formulary; 58-63% of shortages are sterile injectables |
| API sites in India / China | 50% / 32% | 50% / 32% of active API DMFs (USP Medicine Supply Map, 2023) |
| QC batch release time | 5-35 d | 12-18 d typical; sterility / mycoplasma 7-28 d |
| Median scenario fill rate | 97.6% | hospitals are normally fully supplied; shortages are episodic |

Event rates are deliberately **enriched** relative to real incidence so that each network has
enough positive examples to learn from; the risk multiplier and per-type counts allow reweighting.

## 5. Observation layer (what an agent may see)

| Table | Observed columns (`*_obs`) |
|---|---|
| `link_daily` | orders placed, receipts, damaged on arrival, cancelled, outstanding orders, current sourcing share, `data_gap` |
| `product_daily` | patient demand (+-2% recording noise), dispensed, unmet demand, inventory (+-1% count noise), `data_gap` |
| `alerts` | day, source, entity mentioned, reported cause, free text |

Realism in observations: EDI / data-feed outages (NaN blocks, ~2.2% of link-days; 6.6% in
`test_ood_noise`), no deliveries on weekends, lumpy batch receipts, transit damage, inventory
count noise.

**Alerts** are generated from the true events with type-specific visibility (e.g. 90% of
regulatory actions vs. 25% of API shortages become public) and delays (0-90 days). Entity
resolution is imperfect: about 30% name only a region or product, about 25% give no cause.
Supplier allocation notices and national shortage-list entries are derived from the true state.
About 15% of alerts are **false alarms** (unverified reports). This makes the alerts a realistic
input for an LLM / news-reading agent.

## 6. Labels (`label_*`, plus hidden `true_*` states)

| Level | Labels |
|---|---|
| Event (`events.parquet`) | type, category, tier, root node, hit nodes, start, end (cause), recovery day (capacity >= 95%), censoring, severity, theta_down, theta_up, residual capacity, affected links and products, impact start / end |
| Link-day | `label_disrupted` (a supply cause is active upstream), `label_planned`, `label_impaired` (physical supply capability < 90%), `label_event_id` (dominant event), `label_severity` |
| Product-day | `label_shortage`, `label_shortage_id`, `label_shortage_cause` (root-cause type), `label_shortage_next{14,30,60}d`, `label_onset_15_60d` (a NEW shortage starts 15-60 d ahead), `label_days_to_onset`, `label_right_censored60` |
| Shortage (`shortages.parquet`) | product, start, end, duration, unmet units, fill rate, root-cause event and type, criticality |
| Hidden states | site capacity, effective capacity, API availability, supplier backlog, finished goods, QC WIP, production, rejects, shipments, in-transit, lead-time multiplier, transit loss, true demand multiplier |

## 7. Benchmark tasks

| ID | Task | Input | Target | Suggested metrics |
|---|---|---|---|---|
| T1 | Disruption detection and timing (generalises the PIML task) | observed link / product series | onset and end per link (`events`, `label_disrupted`) | AUROC / F1 per link-day; MAE of start / end; temporal IoU (tIoU@0.5/0.7); detection delay; false alarms per year |
| T2 | Root-cause localisation | all links of a network + alerts | which node / tier caused it (`root_node`, `tier`) | top-1 / top-3 accuracy |
| T3 | Disruption-type diagnosis | as T2 | `event_type` (6 supply types vs. demand surge vs. planned) | macro-F1 |
| T4 | Severity and recovery estimation | as T1 | residual capacity, recovery day | MAE |
| T5 | Shortage early warning | observed series + alerts up to day t | `label_onset_15_60d` (evaluate where stock is still >= 14 days of cover) | AUROC, AUPRC, lead time |
| T6 | Mitigation (closed loop) | simulator with `policy` hook | re-allocate sourcing shares to reduce unmet demand | fill rate, unmet critical-drug units, cost of over-ordering |

T6 uses `pharmasim.simulate(scenario, policy=fn)`: `fn` is called weekly with the buyer's current
state and returns new sourcing shares, so agents can be evaluated in closed loop on any test
scenario.

**Naive reference baselines (test split)** — they show the tasks are non-trivial:

| Task | Naive rule | AUROC (test / OOD-severe) |
|---|---|---|
| T1 link-day disruption | outstanding orders / normal volume | 0.71 / 0.73 |
| T1 link-day disruption | 28-day receipt shortfall | 0.61 / 0.67 |
| T5 shortage onset in 15-60 d, healthy stock | inventory days of cover | 0.76 / 0.74 |
| (easy variant) any shortage in the next 30 d | inventory days of cover | 0.97 — do not use as headline |

## 8. Splits

| Split | Networks | Regime | Shift vs. train |
|---|---|---|---|
| `train` | 2,000 | IID | — |
| `val` | 400 | IID | — |
| `test` | 400 | IID | — |
| `test_ood_severe` | 200 | harsher | 2x event rate, deeper capacity loss, 1.4x longer QC release (median fill 85%) |
| `test_ood_network` | 200 | larger | 11-16 products, 6-8 manufacturers, 4-6 API suppliers (unseen size) |
| `test_ood_noise` | 200 | dirtier data | 3x data-gap rate and measurement noise |

Splits are separated by construction: each scenario is an independent network drawn from its own
seed, so no network appears in two splits.

## 9. Files

```
data/<split>/
  scenarios.parquet      one row per network (size, risk, fill rate, counts)
  nodes.parquet          manufacturers and API suppliers (region, headroom, buffers, ...)
  products.parquet       drug attributes
  links.parquet          supplier-product links (share, QC, batch, transit, capacity)
  mfr_molecule.parquet   which API supplier feeds which site for which molecule
  events.parquet         ground-truth events
  shortages.parquet      shortage episodes with root cause
  alerts.parquet         external signals with text
  link_daily/part-*.parquet     per link per day
  product_daily/part-*.parquet  per product per day
  node_daily/part-*.parquet     true capacity per node per day
  api_daily/part-*.parquet      API stock, arrivals, consumption per (site, molecule)
data/metadata.json
```

`loader.py` provides `load_split`, `observed_view` (removes every `true_` / `label_` column, i.e.
what an agent may see at inference) and `legacy_episodes` (converts links to the single-supplier
`lambda, r, b, ts, te` format so the models from Experiments 1-3 run unchanged).

## 10. Reproduce

```bash
pip install numpy pandas pyarrow matplotlib
python generate.py --out data --scale full     # ~4 min with 2 workers, 2.0 GB
python validate.py --root data --report reports
sha256sum -c MANIFEST_full.sha256              # bit-identical check (same library versions)
```

Generated with Python 3.13.18, numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.1. Scenario seeds are
`sha256("pharmanet-v1/<split>/<index>")`. Other numpy versions may change random draws; the
checksum manifest shows whether a regenerated copy is identical.

## 11. Limitations (state these in the paper)

- **Synthetic.** Dynamics and parameter ranges are grounded in public statistics, but no
  individual network corresponds to a real supply chain. Results should be presented as a
  controlled benchmark, ideally complemented by a case study on public shortage records
  (FDA / ASHP).
- **One buyer per network.** The buyer is a single hospital network or distributor; competition
  between buyers for scarce supply is not modelled.
- **No new market entrants.** Once a product's last source is discontinued, the shortage persists
  to the end of the horizon.
- **Event rates are enriched** for learnability; real incidence is lower.
- **The shortage root cause is attributed** to the most severe overlapping event, a modelling
  choice; ambiguous overlaps exist (about 7% of shortages are unattributed).

## 12. Calibration sources

- FDA Drug Shortages Task Force, *Drug Shortages: Root Causes and Potential Solutions* (2019):
  163 shortages — 63% quality / manufacturing, 12% demand, 5% natural disasters,
  3% discontinuations, 18% unknown.
- GAO-14-194 and GAO-16-595: shortages concentrated in generic sterile injectables; quality
  problems are the most frequent immediate cause; other manufacturers have constrained capacity.
- ASHP / University of Utah Drug Information Service and AMA (2026): 58% of new shortages are
  sterile injectables; 48% involve a single manufacturer.
- USP Medicine Supply Map (2023-2025): API DMF shares India 50%, China 32%, EU 10%, US 4%;
  58% of key starting materials sourced from a single country.
- Industry batch-release data: 12-18 days typical; sterility / mycoplasma testing 7-28 days;
  biologics lot release 21-35 days.
