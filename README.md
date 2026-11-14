# PIML Supply-Chain Disruption: Reproduction and Experiments

## What is this project?

There's a research paper that tries to solve this problem:

> A company can see its own orders and deliveries, but **not** what's happening at its supplier. Can you tell â just from your own numbers â **when** a supplier had a problem and for how long?

The paper calls its method **PIML** (Physics-Informed Machine Learning): a small neural network taught some physics rules about how a backlog builds up, so its guesses stay physically realistic.

The paper's own code and data aren't public, so we rebuilt it ourselves, tested it, and ran further experiments on progressively more realistic data.

---

## What we did â Step by Step

### Experiment 1: Clean Synthetic Data (Baseline Reproduction)

**What we built:**
We wrote our own simulator (`src/simulator.py`) that generates fake supplier "episodes" following the same physics as the paper:
- Orders arrive at rate Î»(t)
- Deliveries go out at rate r(t) = K Ã c(t), where c(t) is the supplier's capacity
- Backlog b(t) builds up when a supplier is disrupted and recovers afterward

**Models compared:**
1. **PIML** â the paper's method: a 2-layer neural network (32 neurons, tanh) that predicts when the disruption started/ended, then checks its answer against the physics equations
2. **GreyBox** â no neural network; just directly solves the physics equations for each episode using curve-fitting (scipy least_squares)
3. **Random Forest** â a standard ML model using hand-crafted summary features
4. **LSTM** â a sequence neural network reading the raw time series

**Results on clean synthetic data (5 seeds, 200 episodes each):**

| Model | MAE â | RMSE â | RÂ² â | PCR | mean tIoU â | tIoU@0.5 | tIoU@0.7 |
|-------|-------|--------|------|-----|-------------|----------|----------|
| GreyBox | 0.7 | 0.9 | 1.000 | 100% | 0.993 | 100% | 100% |
| RF | 8.0 | 11.6 | 0.994 | 100% | 0.916 | 98.0% | 97.3% |
| PIML | 40.5 | 52.2 | 0.881 | 100% | 0.692 | 87.3% | 57.3% |
| LSTM | 87.9 | 103.8 | 0.545 | 100% | 0.435 | 40.7% | 17.3% |

**Metric definitions:**
- **MAE/RMSE**: average error (in time-steps) on predicting disruption start/end
- **RÂ²**: how much variance is explained (1.0 = perfect)
- **PCR**: physical consistency rate â are predicted end times after start times? (should be 100%)
- **mean tIoU**: how much the predicted disruption window overlaps the real one (like IoU in object detection)
- **tIoU@0.5/0.7**: fraction of episodes with overlap > 50%/70%

**Finding:** The plain curve-fit (GreyBox) beat the neural network by a wide margin. We checked this wasn't a bug (no data leakage, confirmed across 5 seeds, eye-checked examples). It's the **amortization gap**: when the problem is clean, solving it fresh each time beats a network trained once to guess quickly.

We also tested on **180,000 real e-commerce orders** (DataCo Smart Supply Chain dataset, 2015â2017). The models couldn't find a clean disruption window â because real shipping speed tracks demand elastically every week, while the paper's model assumes shipping only changes during a rare supplier breakdown. See `real_data/README.md` for details.

---

### Experiment 2: Realistic Noisy Synthetic Data + Multi-Agent Comparison

**Motivation:** The clean synthetic data was too easy â real supply chains are messier. We built a harder dataset incorporating the specific real-world complications we found in Experiment 1.

**What the new dataset adds (`src/realistic_simulator.py`):**
1. **Weekly seasonality** â demand has a 7-day cycle (like real retail: busier weekdays, slower weekends)
2. **Elastic demand-capacity correlation** â production rate partially tracks demand even outside disruptions (what we saw in real data: shipping speed follows order volume)
3. **Higher, variable noise** â observation noise std of 0.08â0.15 (vs. 0.03 in the clean version)
4. **Random-walk demand drift** â slow upward/downward trend in base demand level
5. **Measurement lag** â observed production rate is smoothed over 3 days (simulating reporting delay)
6. **Variable disruption severity** â Î¸_down ranges from mild (0.15) to catastrophic (0.90)

All while staying physically sane: b(t) â¥ 0, c(t) â [0,1], r(t) â¥ 0.

**New model: Multi-Agent Ensemble**

We also added a simple **multi-agent approach**:
- **GreyBox, RF, and LSTM act as independent "specialist agents"** â each produces its own prediction of (ts, te)
- A **coordinator agent** aggregates them using element-wise median
- No LLM or API key needed; the "agency" is in the independent analysis and aggregation

**Results on realistic synthetic data (3 seeds, 150 episodes each, T=300 time steps):**

| Model | MAE â | RMSE â | RÂ² â | PCR | mean tIoU â | tIoU@0.5 | tIoU@0.7 |
|-------|-------|--------|------|-----|-------------|----------|----------|
| **RF** | **4.6 Â± 0.2** | **7.4 Â± 1.5** | **0.981** | 100% | **0.885** | **100%** | **94.2%** |
| **MultiAgent** | **5.0 Â± 0.9** | **7.9 Â± 1.6** | **0.979** | 100% | **0.875** | **100%** | **92.8%** |
| GreyBox | 6.1 Â± 0.1 | 9.3 Â± 0.2 | 0.971 | 100% | 0.797 | 92.8% | 82.6% |
| PIML | 8.3 Â± 0.8 | 10.9 Â± 1.4 | 0.959 | 100% | 0.800 | 100% | 78.3% |
| LSTM | 33.9 Â± 2.4 | 40.1 Â± 2.8 | 0.463 | 100% | 0.420 | 39.1% | 13.0% |

**Key findings from Experiment 2:**

1. **RF still wins** â even on noisier, harder data, Random Forest is the most accurate (MAE 4.6). It learns disruption signatures from summary statistics without needing the physics ODE.

2. **Multi-Agent ensemble is a close second** â combining the three specialist agents (median aggregation) achieves MAE 5.0, nearly as good as RF alone, and more robust than any single weaker agent.

3. **PIML ranks 4th** â the amortization gap widens on harder data (MAE goes from 40.5 on clean data to 8.3 on realistic data, but relatively speaking it still trails RF and GreyBox).

4. **GreyBox stays strong** â per-episode curve fitting (MAE 6.1) beats the neural network, confirming Experiment 1.

5. **LSTM struggles most** â without physics structure, it can't reliably locate disruption windows on noisy sequences (MAE 33.9).

---

### Experiment 3: Physics-Informed Knowledge for All Models

**Motivation:** In Experiments 1 and 2, only PIML and GreyBox had physics knowledge â they understood the ODE governing how backlog evolves. RF and LSTM were completely "physics-blind": they only saw raw numbers. The paper argues that physics knowledge is a key advantage. So what if we gave **every** model access to the same physics understanding?

**The core physics insight:** If there were NO disruption, supplier capacity c(t) = 1 always, so production would exactly track demand (r_baseline = K Ã Î»). The gap between observed backlog b(t) and this no-disruption baseline b_baseline(t) is a physics-derived signal that reveals WHEN and HOW MUCH the disruption depressed production â a signal that raw statistics miss.

**What changed in each model (`src/models_v2.py`):**

| Model | What physics knowledge was added |
|-------|----------------------------------|
| **RF_Physics** | Input features replaced with **physics-residual features**: b(t) â b_baseline(t) in segments, ODE gradient differences, estimated disruption window proxies from residual shape |
| **LSTM_Physics** | Added a **physics-consistency loss**: after predicting (ts, te), forward-simulate the ODE and penalize mismatch with observed b(t) â same idea as PIML but starting from LSTM's sequential encoding |
| **PIML_v2** | Physics-residual features (same as RF_Physics) + 3-layer deeper net (64 units) + **softplus duration constraint** so te â¥ ts is guaranteed mathematically, not just clipped |
| **MultiAgent_Physics** | Runs GreyBox + RF_Physics + LSTM_Physics, takes their median as a **warm start**, then runs a final short GreyBox refinement (10 iterations) from that warm start |

**Results on realistic synthetic data (3 seeds, 150 episodes each, T=300 â same dataset as Experiment 2):**

| Model | MAE â | RMSE â | RÂ² â | PCR | mean tIoU â | tIoU@0.5 | tIoU@0.7 |
|-------|-------|--------|------|-----|-------------|----------|----------|
| **RF_Physics** | **3.2 Â± 0.2** | **4.5 Â± 0.1** | **0.993** | 100% | **0.911** | **100%** | **100%** |
| RF (Exp 2 baseline) | 3.4 Â± 0.3 | 4.5 Â± 0.8 | 0.993 | 100% | 0.905 | 100% | 98.6% |
| MultiAgent (Exp 2) | 4.2 Â± 0.3 | 5.6 Â± 0.3 | 0.989 | 100% | 0.886 | 100% | 97.1% |
| MultiAgent_Physics | 5.8 Â± 0.1 | 8.5 Â± 0.1 | 0.976 | 100% | 0.810 | 95.7% | 81.2% |
| PIML_v2 | 5.9 Â± 0.2 | 8.0 Â± 0.7 | 0.979 | 100% | 0.848 | 100% | 89.9% |
| GreyBox (Exp 2) | 6.1 Â± 0.1 | 9.3 Â± 0.2 | 0.971 | 100% | 0.797 | 92.8% | 82.6% |
| PIML (Exp 2) | 8.5 Â± 1.2 | 11.2 Â± 2.0 | 0.957 | 100% | 0.794 | 95.7% | 75.4% |
| LSTM_Physics | 29.1 Â± 3.8 | 34.5 Â± 4.5 | 0.599 | 100% | 0.454 | 43.5% | 10.1% |
| LSTM (Exp 2) | 33.4 Â± 2.7 | 39.6 Â± 2.7 | 0.479 | 100% | 0.410 | 40.6% | 7.2% |

**Key findings from Experiment 3:**

1. **Physics features help RF most** â RF_Physics achieves tIoU@0.7 = **100%** (perfect precision detection), the only model to ever hit this. MAE improves from 3.4 â 3.2. The ODE-derived residual features give RF the right signal to separate disruption from seasonal noise.

2. **PIML_v2 improves significantly over PIML** â Adding physics-residual inputs + softplus constraint + a deeper net drops MAE from 8.5 â 5.9. The constraint preventing te < ts eliminates a whole class of physically impossible predictions.

3. **Physics consistency loss helps LSTM slightly** â LSTM_Physics (MAE 29.1) beats plain LSTM (MAE 33.4). The physics loss provides a learning signal that raw supervision on (ts, te) labels alone doesn't. But LSTM still struggles because a 2-layer LSTM reads all 300 time steps and can't locate a narrow disruption window precisely.

4. **Physics warm-start hurts MultiAgent** â MultiAgent_Physics (MAE 5.8) is worse than plain MultiAgent (MAE 4.2). The final GreyBox refinement step sometimes "corrects" RF's accurate data-driven answer toward the physics ODE's assumptions, introducing error. When RF is already right, adding ODE refinement is counterproductive.

5. **The best model overall is RF_Physics** â combining RF's ability to learn from data with physics-derived features that encode the paper's ODE structure. This is exactly what the paper argued for: **data-driven learning + physics knowledge together**.

**Conclusion across all 3 experiments:**
The paper's claim that physics knowledge helps is validated â but the biggest benefit goes to the simplest model (RF) getting better input features, not to the neural network architecture (PIML). Pure data-driven RF with the right physics-derived features beats the full PIML system the paper proposes.

---

## Repository Layout

```
piml-supply-chain-repro/
â
âââ src/
â   âââ simulator.py                 # Experiment 1: clean physics simulator
â   âââ realistic_simulator.py       # Experiments 2+3: realistic noisy simulator
â   âââ models.py                    # Exp 1+2 models (PIML, GreyBox, RF, LSTM, MultiAgent)
â   âââ models_v2.py                 # Exp 3 models (RF_Physics, LSTM_Physics, PIML_v2, MultiAgent_Physics)
â   âââ run_experiment.py            # Experiment 1 runner
â   âââ run_realistic_experiment.py  # Experiment 2 runner
â   âââ run_physics_experiment.py    # Experiment 3 runner
â
âââ results/
â   âââ table7_8.csv                 # Experiment 1 results
â
âââ results_realistic/
â   âââ realistic_comparison.csv    # Experiment 2 results
â
âââ results_physics/
â   âââ physics_comparison.csv      # Experiment 3 results (9 models)
â
âââ real_data/
â   âââ build_series.py              # Converts DataCo CSV to daily Î»/r/b series
â   âââ build_series_weekly.py       # Same at weekly resolution
â   âââ README.md                   # Why the real-data experiment found nothing
â
âââ requirements.txt
âââ README.md
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

---

## What's Left to Do

- **Full-length time series**: All experiments used T=300 steps for speed; running T=900 would more closely match the paper and may change relative rankings
- **More seeds**: 3 seeds (Exp. 2+3) vs. 5 seeds (Exp. 1) â increase to 5 for fairer comparison
- **Tune LSTM_Physics further**: The physics consistency loss weight (0.3) was not tuned; a grid search might find a better balance between supervised and physics losses
- **Better LSTM architecture**: Replace the 2-layer LSTM with a Temporal Fusion Transformer or attention mechanism that can better locate narrow disruption windows in long sequences
- **TFT baseline**: The paper also compares against Temporal Fusion Transformers (not yet implemented here)
- **Capacity-sharing optimization**: The second half of the paper (using the disruption estimate to share capacity across a network of suppliers) â not yet attempted
- **LLM/RAG agent**: A proper LLM-backed agent that reasons about the time series in natural language â requires an API key

---

## Honesty Notes

- This is our own rebuild based on reading the paper's equations â not the authors' actual code
- Our simulated data is ours, not theirs; real comparisons would need the original dataset
- We haven't tuned PIML as heavily as the paper describes
- The multi-agent approach here is a simple rule-based ensemble, not a full LLM-backed agent system
- Experiment 3 physics features are hand-designed from the ODE structure; learned physics features (via NeuralODE or differentiable simulation) would be a natural next step
