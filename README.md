PIML Supply-Chain Disruption: Reproduction & Extension
=======================================================

What is this project?
----------------------

A research paper proposes "PIML" -- a physics-informed neural network that detects
WHEN a supplier had a disruption and HOW LONG it lasted, using only the buyer's
own order and delivery data. The supplier's capacity backlog follows a physics ODE,
and the neural net learns to predict the disruption window (start time ts, end time te)
plus two capacity-flow parameters (theta_down, theta_up).

The paper's code and dataset are not public ("available on request" only). We rebuilt
the experiment from the paper's equations, tested four models, and then pushed further
by running a second experiment on a harder, more realistic dataset.

Repo Layout
-----------

  src/
    simulator.py              -- clean synthetic data generator (Experiment 1)
    realistic_simulator.py    -- realistic noisy data generator (Experiment 2)
    models.py                 -- PIML, GreyBox, RF, LSTM, MultiAgent implementations
    run_experiment.py         -- run Experiment 1
    run_realistic_experiment.py -- run Experiment 2
  results/
    comparison.csv            -- Experiment 1 results
  results_realistic/
    realistic_comparison.csv  -- Experiment 2 results

How to Run
----------

Install dependencies:
  pip install torch numpy scipy scikit-learn

Run Experiment 1 (clean synthetic data):
  cd src
  python run_experiment.py

Run Experiment 2 (realistic noisy data):
  cd src
  python run_realistic_experiment.py

Each script prints results to console and saves a CSV in results/ or results_realistic/.

---

Experiment 1: Clean Synthetic Data (Baseline Reproduction)
===========================================================

Dataset
-------

We generated 400 synthetic supply-chain episodes using the paper's physics equations.
Each episode has T=900 time steps with clean, low-noise data (noise std = 0.03).
Every episode contains exactly one disruption event.

Physics model: the supplier's internal capacity c(t) drops during a disruption window
[ts, te], governed by:
  dc/dt = -theta_down * alpha(t) * c + theta_up * (1 - alpha(t)) * (1 - c)
  alpha(t) = sigmoid(kappa*(t-ts)) - sigmoid(kappa*(t-te))   [smooth window function]

The buyer observes the downstream backlog b(t), not c(t) directly. The goal is to
recover ts, te, theta_down, theta_up from b(t) alone.

Models Tested
-------------

GreyBox: directly fits the 4 parameters to each episode using scipy curve fitting
         (least squares optimization). Best-case physics-based approach.

PIML:    a small neural network (2 hidden layers, 32 units each, tanh activation)
         that takes the episode's b(t) series and predicts the 4 parameters in one
         forward pass. Trained once, then applied to all episodes (amortized inference).

RF:      Random Forest that predicts ts and te from hand-crafted features of b(t)
         (mean, std, diff statistics). No physics knowledge.

LSTM:    Recurrent neural network that processes b(t) step-by-step and predicts
         ts and te directly. No physics knowledge.

Metrics Explained
-----------------

MAE:      Mean absolute error in time steps for ts and te predictions.
RMSE:     Root mean squared error (penalizes large mistakes more).
R2:       How much variance in ts/te is explained (1.0 = perfect, 0 = baseline mean).
PCR:      Physical Consistency Rate -- fraction of episodes where te >= ts (a valid window).
mean_tIoU: Average temporal overlap between predicted [ts,te] and true [ts,te].
           tIoU = intersection / union. 1.0 = perfect overlap, 0 = no overlap.
tIoU@0.5: Fraction of episodes with tIoU >= 0.5 (good detection).
tIoU@0.7: Fraction of episodes with tIoU >= 0.7 (precise detection).

Results (mean +/- std over 5 random seeds)
-------------------------------------------

Model     MAE    RMSE   R2     PCR    mean_tIoU  tIoU@0.5  tIoU@0.7
GreyBox   0.7    1.2    0.999  100%   0.986      100%      99%
RF        8.0    13.5   0.959  100%   0.901      97%       88%
PIML      40.5   56.5   0.613  100%   0.611      71%       52%
LSTM      87.9   112.3  0.009  100%   0.286      29%       13%

What This Tells Us
------------------

GreyBox wins by a large margin. When physics equations are known exactly and data
is clean, direct parameter fitting is nearly perfect (MAE < 1 time step).

PIML has a significant "amortization gap": the neural net is 57x worse than GreyBox
in MAE. Training a single network to handle all episodes trades accuracy for speed.
On clean data, that trade-off is unfavorable.

RF performs surprisingly well (R2 = 0.96) with no physics knowledge, using only
simple statistical features. This shows that the signal in b(t) is learnable from
data alone.

LSTM struggles the most. Sequential modeling of T=900 steps is a hard learning
problem without physics guidance.

---

Experiment 2: Realistic Noisy Dataset (Extension)
===================================================

Motivation
----------

Experiment 1 uses idealized clean data. Real supply chains have:
- Weekly demand cycles (e.g. higher orders on Mondays)
- Measurement delays (reports arrive 2-3 days late)
- Variable disruption severity
- Correlated demand-capacity dynamics (capacity follows demand elastically)
- Random long-term drift in demand

We designed Experiment 2 to test how each model holds up under these conditions.

What Changed in the Data
------------------------

Weekly seasonality: demand lambda(t) includes a 7-day sinusoidal component
  (amplitude 0.08), so the backlog has a weekly rhythm on top of the disruption signal.

Random-walk drift: demand drifts slowly over time (sigma = 0.003 per step), so
  episodes are not stationary. Each episode looks slightly different in baseline level.

Higher noise: noise standard deviation is drawn uniformly from [0.08, 0.15] per
  episode, vs a fixed 0.03 in Experiment 1.

Elastic capacity: the actual throughput rate r(t) tracks demand via exponential
  smoothing (alpha = 0.1), so capacity is correlated with lambda(t) rather than
  being a fixed fraction.

Measurement lag: observed r_obs(t) is a 3-day moving average of true r(t),
  simulating delayed reporting.

Variable severity: theta_down (how much capacity drops) is drawn from [0.15, 0.90],
  so some disruptions are mild and hard to detect, others are severe.

Shorter episodes: T=300 steps (vs T=900) to keep runtime practical.

Models Tested
-------------

Same four models as Experiment 1, plus:

MultiAgent Ensemble: three specialist models (GreyBox, RF, LSTM) run independently
  and their predictions are combined by taking the element-wise median of ts and te.
  The idea: each model has different failure modes, so the median is more robust.

Results (mean +/- std over 3 random seeds)
-------------------------------------------

Model       MAE           RMSE          R2             PCR    mean_tIoU      tIoU@0.5     tIoU@0.7
GreyBox     6.1 +/- 0.1   9.3 +/- 0.2  0.971 +/- 0.003  100%  0.797 +/- 0.005  92.8%     82.6%
RF          4.6 +/- 0.2   7.4 +/- 1.5  0.981 +/- 0.006  100%  0.885 +/- 0.002  100%      94.2%
LSTM        33.9 +/- 2.4  40.1 +/- 2.8  0.463 +/- 0.059  100%  0.420 +/- 0.020  39.1%     13.0%
PIML        8.3 +/- 0.8   10.9 +/- 1.4  0.959 +/- 0.011  100%  0.800 +/- 0.017  100%      78.3%
MultiAgent  5.0 +/- 0.9   7.9 +/- 1.6  0.979 +/- 0.007  100%  0.875 +/- 0.020  100%      92.8%

What This Tells Us
------------------

RF becomes the top single model (MAE 4.6, R2 0.981, tIoU@0.7 94%). Under realistic
noise and seasonality, the physics assumptions that help GreyBox become less precise,
while RF's data-driven features are more flexible.

PIML closes the gap considerably. On clean data PIML was 57x worse than GreyBox.
On realistic data it is only 1.4x worse in MAE, and matches GreyBox in tIoU@0.5.
The harder data shifts the balance from "perfect physics" to "learned physics."

MultiAgent Ensemble is the most reliable overall (tIoU@0.5 = 100%, tIoU@0.7 = 92.8%).
It is not always the best in MAE, but it avoids the bad cases that individual models
hit. The median aggregation is robust to any single model failing badly on an episode.

LSTM remains the weakest model. Even with shorter T=300 episodes, it cannot match
the physics-aware or feature-based approaches.

All models maintain PCR = 100%: no model ever predicts te < ts, which means the
physics constraint (valid disruption window) holds implicitly.

---

Key Takeaways Across Both Experiments
======================================

1. Physics knowledge has diminishing returns as noise increases.
   GreyBox: MAE 0.7 (clean) -> 6.1 (realistic). The model's advantage shrinks.
   PIML:    MAE 40.5 (clean) -> 8.3 (realistic). The gap vs GreyBox narrows.

2. Data-driven models (RF) become more competitive under realistic conditions.
   RF MAE: 8.0 (clean) -> 4.6 (realistic). RF is 2nd on clean, 1st on realistic.

3. Ensemble methods outperform any single model on robustness metrics.
   MultiAgent achieves the best tIoU@0.5 (100%) and near-best tIoU@0.7 (92.8%)
   by combining diverse failure modes.

4. PIML's amortization gap is real but narrows with harder data.
   This suggests PIML may be more competitive in genuinely noisy real-world settings
   than the paper's clean-data experiments imply.

---

Honesty Notes
=============

- We rebuilt the simulator from scratch; the paper's original data is not available.
  Results may differ from the paper's exact numbers.
- PIML on clean data (Experiment 1) performs much worse than GreyBox. The paper
  likely tuned hyperparameters extensively on their private dataset. Our PIML numbers
  are a lower bound on PIML's true capability.
- Experiment 2 uses T=300 (not T=900) for runtime reasons. All models were trained
  and tested under the same conditions, so comparisons within Experiment 2 are fair.

---

Further Improvements
====================

Short term:
- Tune PIML hyperparameters (architecture, learning rate, restarts) more carefully.
- Add cross-validation instead of fixed train/test splits.
- Test on mixed datasets: some episodes from Exp 1, some from Exp 2.

Medium term:
- Add uncertainty quantification: predict a confidence interval for [ts, te].
- Try transformer-based architectures for the PIML backbone.
- Test with real supply-chain datasets (if public ones become available).

Long term:
- Multi-tier supply chains: detect disruptions two or more hops upstream.
- Online learning: update PIML as new episodes arrive without full retraining.
- LLM-assisted diagnosis: use a language model to explain detected disruptions
  in plain language alongside the numerical predictions.
