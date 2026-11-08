PIML Supply-Chain Disruption: Open Reproduction
====================================================

A simple, open reproduction and critique of: Matsuno, Watanabe, Lin, Tanimizu (2026). PIML-Driven Supply Chain Disruption Estimation and Capacity-Sharing Optimization. ASME J. Comput. Inf. Sci. Eng. 26(11), 111003.

The original paper infers a hidden upstream supplier's disruption window (ts, te) from a focal firm's own order/backlog/delivery records, using a physics-informed neural network (PIML) trained with a hybrid loss. It reports PIML beating LSTM/RF/TFT baselines on a private simulator.

We could not get the original data or code (available only "on reasonable request"), so we built an open simulator of the same physics (backlog conservation law, sigmoid disruption window, capacity ramp-up ODE -- Eqs 1 to 7 of the paper) and reproduced the comparison ourselves, adding one baseline the paper is missing: a grey-box per-episode curve fit of the same 4 parameters with scipy.optimize.least_squares, no neural network at all.

Result
------

5 seeds, 200 episodes, 900 steps each. Mean absolute error (MAE) in time units, lower is better; R2 and mean tIoU closer to 1 is better.

| Model | MAE | R2 | mean tIoU |
|-------|-----|----|-----------|
| Grey-box curve fit, no neural net | 0.7 | 1.000 | 0.993 |
| Random Forest | 8.0 | 0.994 | 0.916 |
| PIML, our reimplementation, fixed training | 40.5 | 0.881 | 0.692 |
| PIML, first pass, before the fix | 54.7 | 0.699 | 0.616 |
| LSTM | 87.9 | 0.545 | 0.435 |

See results/table7_8.csv for the exact numbers and results/run_log_v2.txt for the full per-seed log.

Headline finding, opposite of the paper's claim: in our open noiseless-physics simulator, a plain per-episode least-squares fit of the same 4 unknown parameters essentially solves the problem exactly, and clearly beats our reimplementation of the PIML neural network. Random Forest, with no physics at all, also beats PIML. The amortized neural network is harder to train than a direct per-episode physics fit when there are only 4 real unknowns and the forward model is exact. This is the "amortization gap": a network trained once across many episodes trades per-case accuracy for the ability to do one fast forward pass later, while the grey-box fit re-solves a small, well-posed optimization from scratch per episode. This is a known, expected result in machine learning, not a bug.

Verification
------------

Before trusting this result we checked for a bug. The grey-box fitting function only reads an episode's own observed order rate and backlog, never the ground-truth disruption window it is trying to recover, confirmed by inspecting its source. Spot-checking individual test episodes shows grey-box lands within one or two time units of the true window every time, while PIML is off by tens of units on the same episodes. The pattern holds on every one of the 5 random seeds, not just on average.

What this means for next steps
-------------------------------

The grey-box fit is clearly the right method for a single, clean disruption with noiseless physics. PIML's case has to be made where a plain fit cannot go: multiple overlapping disruptions, partial observability, or needing one fast forward pass instead of re-optimizing per supplier every epoch. We already improved PIML's training once (normalizing the loss scale and using 3 random restarts), which cut its error by about 25 percent and its across-seed variance by about 3.5x, but it still does not close the gap. Next steps: add observation noise and overlapping disruptions to see whether grey-box's advantage holds once the physics stops being exact, and test real-world data.

Repo layout
-----------

src/simulator.py is the open simulator. src/models.py has the PIML network, the grey-box fit, and the Random Forest and LSTM baselines. src/run_experiment.py runs the 5-seed comparison and writes results/table7_8.csv.

Reproduce
---------

Install torch, scikit-learn, scipy, numpy and pandas, then run src/run_experiment.py. Takes about 20 minutes on a laptop CPU, no GPU needed.

Honesty notes
--------------

This is our own reimplementation of the paper's method based on its equations, not the authors' code, so results here are not a critique of their actual implementation. All data is synthetic, from our own simulator, not the paper's private dataset. We have not yet tuned PIML's training as carefully as the paper reports doing.
