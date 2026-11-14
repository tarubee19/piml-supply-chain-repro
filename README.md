# PIML Supply-Chain Disruption: Reproduction and Experiments

A reproduction and extension of a Physics-Informed Machine Learning (PIML) paper for detecting supplier disruptions in supply chains.

---

## The Problem

Imagine you are a company buying parts from a supplier. You can see your own inventory (backlog), but you **cannot see inside your supplier's factory**. The question is:

> Can you figure out -- just from your own inventory numbers -- **when** your supplier had a breakdown, and **for how long**?

The paper says yes, using Physics-Informed Machine Learning (PIML). We rebuilt the method from scratch, tested it, and ran three experiments to understand what actually works and why.

---

## The Physics

The supplier's situation follows simple differential equations (ODEs):

- Orders arrive at demand rate lambda(t)
- Production ships out at rate r(t) = K x c(t), where c(t) is the supplier's capacity
- Backlog b(t) builds up when a disruption hits and clears after recovery

During a disruption window [ts, te], capacity degrades and then recovers:

```
db/dt = lambda(t) - r(t)
r(t)  = K x c(t)
dc/dt = -theta_down x alpha(t) x c + theta_up x (1 - alpha) x (1 - c)
```

The goal: given observed (lambda, b, r), recover the hidden disruption timing (ts, te).

---

## What We Did -- Three Experiments

---

### Experiment 1: Clean Synthetic Data (Baseline Reproduction)

**What we built:**
A simulator following the paper's physics equations. Each episode is 300 time steps with one disruption event. Data is clean with low noise (std = 0.03).

**Models compared:**

| Model | Description |
|-------|-------------|
| GreyBox | Directly solves the physics ODE per episode using curve fitting (scipy least_squares). No machine learning. |
| Random Forest | Standard ML model trained on 15 hand-crafted summary features (backlog stats, rate drop, autocorrelations). |
| PIML | The paper's method: 2-layer neural network (32 units, tanh) + physics consistency loss. |
| LSTM | Sequence neural network reading the raw 300-step time series. |

**Results (5 seeds, 200 episodes each):**

| Model | MAE (days) | RMSE | R-squared | PCR | tIoU@0.5 | tIoU@0.7 |
|-------|-----------|------|-----------|-----|----------|----------|
| GreyBox | 0.7 | 0.9 | 1.000 | 100% | 100% | 100% |
| RF | 8.0 | 11.6 | 0.994 | 100% | 98.0% | 97.3% |
| PIML | 40.5 | 52.2 | 0.881 | 100% | 87.3% | 57.3% |
| LSTM | 87.9 | 103.8 | 0.545 | 100% | 40.7% | 17.3% |

**Metric definitions:**
- **MAE / RMSE**: average error in predicting disruption start and end, in time steps (days). Lower is better.
- **R-squared**: how much variance is explained (1.0 = perfect)
- **PCR (Physical Consistency Rate)**: are predicted end times always after start times? Should be 100%.
- **mean tIoU**: how much the predicted disruption window overlaps the real one (like IoU in object detection)
- **tIoU@0.5 / tIoU@0.7**: fraction of episodes where overlap exceeds 50% / 70%

**Key finding:** The plain curve-fitter (GreyBox) dominated the neural network (PIML) by a large margin. When data is clean and the physics is known, solving the equations directly beats a network trained to approximate the same thing. This is known as the **amortization gap**.

We also tested on 180,000 real e-commerce orders (DataCo Smart Supply Chain dataset, 2015-2017). The models could not find a clean disruption window because real shipping rates track demand elastically every week -- violating the paper's assumption that disruptions are rare, discrete events.

---

### Experiment 2: Realistic Noisy Data + Multi-Agent Ensemble

**Motivation:** Clean synthetic data was too easy. Real supply chains have multiple compounding noise sources.

**What the new simulator adds (src/realistic_simulator.py):**

1. Weekly seasonality -- demand has a 7-day cycle (busier weekdays, slower weekends)
2. Elastic demand-capacity correlation -- production partially tracks demand even outside disruptions
3. Higher variable noise -- observation noise std of 0.08 to 0.15 (vs 0.03 in Experiment 1)
4. Random-walk demand drift -- slow upward/downward trend in base demand level
5. Measurement lag -- observed production rate is smoothed over 3 days (simulating reporting delay)
6. Variable disruption severity -- theta_down ranges from mild (0.15) to catastrophic (0.90)

**New model: Multi-Agent Ensemble**

Three specialist agents (GreyBox, RF, LSTM) each produce an independent prediction of (ts, te). A coordinator aggregates them using element-wise median. No LLM or API needed. The median is robust to one bad agent.

**Results (3 seeds, 150 episodes each, T=300):**

| Model | MAE (days) | RMSE | R-squared | PCR | tIoU@0.5 | tIoU@0.7 |
|-------|-----------|------|-----------|-----|----------|----------|
| RF | 3.4 | 4.5 | 0.993 | 100% | 100% | 98.6% |
| MultiAgent | 4.2 | 5.6 | 0.989 | 100% | 100% | 97.1% |
| GreyBox | 6.1 | 9.3 | 0.971 | 100% | 92.8% | 82.6% |
| PIML | 8.5 | 11.2 | 0.957 | 100% | 95.7% | 75.4% |
| LSTM | 33.4 | 39.6 | 0.479 | 100% | 40.6% | 7.2% |

**Key findings:**
- RF overtakes GreyBox on noisy data -- curve fitting gets stuck in local minima when noise is high
- Multi-Agent ensemble is a close second, more robust than any single weaker model
- PIML improves relatively (MAE 40.5 to 8.5) but still trails RF and GreyBox
- LSTM struggles throughout -- without physics structure it cannot reliably locate narrow disruption windows

---

### Experiment 3: Physics-Informed Knowledge for All Models

**Motivation:** In Experiments 1 and 2, only PIML and GreyBox had access to the ODE structure. RF and LSTM were completely physics-blind. The paper argues that physics knowledge is a key advantage. So we gave **every** model the same physics knowledge and tested whether it helps.

**The core physics insight:**

If there were NO disruption, capacity c(t) = 1 always, so production would exactly track demand. We call this the no-disruption baseline:

```
b_baseline(t) = integral of [lambda(t) - K*lambda(t)] from 0 to t
```

The physics residual:

```
residual(t) = b_observed(t) - b_baseline(t)
```

This residual is near zero outside disruptions and dips negative when capacity is suppressed. It directly reveals WHEN and HOW MUCH the disruption depressed production -- a signal that raw statistics miss entirely.

**What changed in each model:**

| Model | Physics knowledge added |
|-------|------------------------|
| RF_Physics | Input features replaced with 35 ODE-derived residual features: residual shape segments, gradient differences, disruption window proxies (centre, spread, area), peak magnitude, drop/rise rates |
| LSTM_Physics | Added physics consistency loss during training: after predicting (ts, te), forward-simulate the ODE and penalize mismatch with observed b(t). Also added softplus duration constraint so te >= ts is mathematically guaranteed |
| PIML_v2 | Same ODE residual features as RF_Physics + deeper network (3 layers, 64 units) + softplus duration constraint |
| MultiAgent_Physics | Runs GreyBox + RF_Physics + LSTM_Physics, takes their median as a warm start, then runs a short GreyBox refinement from that warm start instead of a fixed prior |

**Results (same dataset as Experiment 2, 3 seeds, 150 episodes each):**

| Model | MAE (days) | RMSE | R-squared | PCR | tIoU@0.5 | tIoU@0.7 |
|-------|-----------|------|-----------|-----|----------|----------|
| RF_Physics (NEW) | **3.2** | **4.5** | **0.993** | 100% | **100%** | **100%** |
| RF (Exp 2 baseline) | 3.4 | 4.5 | 0.993 | 100% | 100% | 98.6% |
| MultiAgent (Exp 2) | 4.2 | 5.6 | 0.989 | 100% | 100% | 97.1% |
| PIML_v2 (NEW) | 5.9 | 8.0 | 0.979 | 100% | 100% | 89.9% |
| MultiAgent_Physics (NEW) | 5.8 | 8.5 | 0.976 | 100% | 95.7% | 81.2% |
| GreyBox (Exp 2) | 6.1 | 9.3 | 0.971 | 100% | 92.8% | 82.6% |
| PIML (Exp 2 baseline) | 8.5 | 11.2 | 0.957 | 100% | 95.7% | 75.4% |
| LSTM_Physics (NEW) | 29.1 | 34.5 | 0.599 | 100% | 43.5% | 10.1% |
| LSTM (Exp 2 baseline) | 33.4 | 39.6 | 0.479 | 100% | 40.6% | 7.2% |

**Key findings:**

1. **Physics features help RF most** -- RF_Physics is the only model across all three experiments to achieve tIoU@0.7 = 100% (perfect precision detection). The ODE residual is a near-sufficient statistic for disruption timing, removing seasonal and elastic noise that confounds raw statistics.

2. **PIML_v2 improves significantly over PIML** -- MAE drops from 8.5 to 5.9. The softplus constraint eliminates physically impossible predictions (te < ts). But with only 105 training episodes, RF still generalizes more efficiently.

3. **Physics consistency loss helps LSTM slightly** -- MAE improves from 33.4 to 29.1. But LSTM's core problem is architectural: a 2-layer LSTM encoding 300 timesteps loses the temporal resolution needed to pinpoint a narrow disruption window. This is an attention/transformer problem, not a loss function problem.

4. **Physics warm-start hurts MultiAgent** -- MultiAgent_Physics (MAE 5.8) is worse than plain MultiAgent (MAE 4.2). The final GreyBox refinement step pulls RF_Physics's accurate answer toward what the ODE fitting surface suggests, introducing error when RF was already right. When a data-driven model is correct, adding a physics correction can make things worse.

5. **Overall winner: RF_Physics** -- combining RF's pattern recognition with physics-derived input features gives the best of both worlds. This validates the paper's central argument -- physics knowledge helps -- but the benefit goes to the simplest model, not the complex neural architecture the paper proposed.

---

## Overall Conclusion Across All 3 Experiments

The paper's claim that physics knowledge is a key advantage is validated. But the biggest benefit goes to Random Forest getting better input features, not to the PIML neural architecture.

| What we found | What it means |
|--------------|---------------|
| GreyBox beats PIML on clean data | Solving physics directly beats learning to approximate it when data is clean |
| RF beats GreyBox on noisy data | Data-driven learning beats pure physics when real-world noise is present |
| RF_Physics beats both | Physics knowledge as input features to a simple model is the winning combination |
| LSTM always struggles | Without a way to pinpoint a narrow time window, sequence models cannot locate disruptions precisely |

---

## Repository Layout

```
piml-supply-chain-repro/
|
|-- src/
|   |-- simulator.py                    # Experiment 1: clean physics simulator
|   |-- realistic_simulator.py          # Experiments 2+3: realistic noisy simulator
|   |-- models.py                       # Exp 1+2 models (PIML, GreyBox, RF, LSTM, MultiAgent)
|   |-- models_v2.py                    # Exp 3 models (RF_Physics, LSTM_Physics, PIML_v2, MultiAgent_Physics)
|   |-- run_experiment.py               # Experiment 1 runner
|   |-- run_realistic_experiment.py     # Experiment 2 runner
|   `-- run_physics_experiment.py       # Experiment 3 runner
|
|-- results/
|   `-- table7_8.csv                    # Experiment 1 results
|
|-- results_realistic/
|   `-- realistic_comparison.csv        # Experiment 2 results
|
|-- results_physics/
|   `-- physics_comparison.csv          # Experiment 3 results (9 models)
|
|-- real_data/
|   |-- build_series.py                 # Converts DataCo CSV to daily lambda/r/b series
|   |-- build_series_weekly.py          # Same at weekly resolution
|   `-- README.md                       # Why the real-data experiment found nothing
|
|-- requirements.txt
`-- README.md
```

---

## How to Run

```bash
pip install -r requirements.txt

# Experiment 1: clean synthetic data
cd src && python run_experiment.py

# Experiment 2: realistic noisy data + multi-agent
cd src && python run_realistic_experiment.py

# Experiment 3: physics-informed models for all
cd src && python run_physics_experiment.py
```

Results are saved to the respective results/ folders as CSV files.

---

## Further Improvements

- **Full-length time series**: All experiments used T=300 steps for speed. Running T=900 would more closely match the paper and may change relative rankings.
- **More seeds**: 3 seeds (Exp 2+3) vs 5 seeds (Exp 1) -- increase to 5 for fairer comparison.
- **Tune LSTM_Physics**: The physics consistency loss weight (0.3) was not tuned. A grid search might find a better balance between supervised and physics losses.
- **Better LSTM architecture**: Replace the 2-layer LSTM with a Temporal Fusion Transformer or attention mechanism that can better locate narrow disruption windows in long sequences.
- **Learned physics features**: Replace hand-designed ODE residual features with features learned end-to-end via Neural ODE or differentiable simulation. This removes the assumption that the governing equations are known.
- **TFT baseline**: The paper also compares against Temporal Fusion Transformers (not yet implemented here).
- **Capacity-sharing optimization**: The second half of the paper (using the disruption estimate to share capacity across a network of suppliers) -- not yet attempted.
- **LLM-backed agent**: A proper LLM-based agent that reasons about the time series in natural language -- requires an API key.

---

## Honesty Notes

- This is our own rebuild based on reading the paper's equations -- not the authors' actual code.
- Our simulated data is ours, not theirs. Real comparisons would need the original dataset.
- We have not tuned PIML as heavily as the paper describes.
- The multi-agent approach here is a simple rule-based ensemble, not a full LLM-backed agent system.
- Experiment 3 physics features are hand-designed from the ODE structure. Learned physics features (via Neural ODE or differentiable simulation) would be a natural next step.
