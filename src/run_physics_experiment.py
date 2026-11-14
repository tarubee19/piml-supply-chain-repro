"""
Experiment 3: Physics-Informed Models on Realistic Dataset

Compares ALL models side by side:
  - GreyBox            (baseline physics solver, same as Exp 2)
  - RF                 (plain RF, same as Exp 2)
  - LSTM               (plain LSTM, same as Exp 2)
  - PIML               (original PIML, same as Exp 2)
  - MultiAgent         (median of GB+RF+LSTM, same as Exp 2)

NEW in Experiment 3:
  - RF_Physics         (RF with physics-residual features)
  - LSTM_Physics       (LSTM with physics-consistency loss)
  - PIML_v2            (PIML with physics features + softplus duration + deeper net)
  - MultiAgent_Physics (median of GB+RF_Physics+LSTM_Physics, then physics warm-start)

Dataset: same realistic noisy dataset as Experiment 2
  - Weekly seasonality, random-walk drift, 3-day lag, variable severity
  - T=300, N=150, 3 seeds

Results saved to results_physics/physics_comparison.csv
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from realistic_simulator import make_realistic_dataset
from models import (fit_greybox, train_predict_rf, train_predict_lstm, train_piml,
                    predict_piml)
from models_v2 import (train_predict_rf_physics, train_predict_lstm_physics,
                        train_piml_v2, predict_piml_v2,
                        predict_multiagent_physics)

# ââ Config ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
N_EPISODES = 150
T          = 300
SEEDS      = [0, 1, 2]
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15   # rest = test

OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "results_physics")
OUT_FILE   = os.path.join(OUT_DIR, "physics_comparison.csv")

# ââ Metrics âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def tiou(ts_p, te_p, ts_t, te_t):
    inter = max(0.0, min(te_p, te_t) - max(ts_p, ts_t))
    union = max(te_p, te_t) - min(ts_p, ts_t)
    return inter / union if union > 0 else 0.0


def compute_metrics(ts_pred, te_pred, ts_true, te_true):
    n = len(ts_true)
    errs = []
    for i in range(n):
        errs += [abs(ts_pred[i] - ts_true[i]), abs(te_pred[i] - te_true[i])]
    mae  = np.mean(errs)
    rmse = np.sqrt(np.mean(np.array(errs)**2))
    all_t = np.concatenate([ts_true, te_true])
    all_p = np.concatenate([ts_pred, te_pred])
    ss_res = np.sum((all_t - all_p)**2)
    ss_tot = np.sum((all_t - all_t.mean())**2)
    r2   = 1 - ss_res / (ss_tot + 1e-10)
    pcr  = np.mean(te_pred >= ts_pred) * 100
    tious = [tiou(ts_pred[i], te_pred[i], ts_true[i], te_true[i]) for i in range(n)]
    mean_tiou = np.mean(tious)
    tiou50 = np.mean([v >= 0.5 for v in tious]) * 100
    tiou70 = np.mean([v >= 0.7 for v in tious]) * 100
    return mae, rmse, r2, pcr, mean_tiou, tiou50, tiou70


# ââ Runner ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def run_seed(seed):
    print(f"\n{'='*55}")
    print(f"  SEED {seed}")
    print(f"{'='*55}")

    eps = make_realistic_dataset(n_episodes=N_EPISODES, T=T, seed=seed)

    n_train = int(N_EPISODES * TRAIN_FRAC)
    n_val   = int(N_EPISODES * VAL_FRAC)
    train_eps = eps[:n_train]
    val_eps   = eps[n_train:n_train + n_val]
    test_eps  = eps[n_train + n_val:]

    ts_true = np.array([e["ts"] for e in test_eps])
    te_true = np.array([e["te"] for e in test_eps])
    n_test  = len(test_eps)

    results = {}

    # ââ 1. GreyBox (baseline) ââââââââââââââââââââââââââââââââââââââââââââââ
    t0 = time.time()
    gb_ts, gb_te = [], []
    for ep in test_eps:
        ts_p, te_p, _, _ = fit_greybox(ep, T0_prior=T*0.25)
        gb_ts.append(ts_p); gb_te.append(te_p)
    gb_ts, gb_te = np.array(gb_ts), np.array(gb_te)
    results["GreyBox"] = compute_metrics(gb_ts, gb_te, ts_true, te_true)
    print(f"  GreyBox done ({time.time()-t0:.1f}s)  MAE={results['GreyBox'][0]:.2f}")

    # ââ 2. RF (plain, Exp 2 baseline) âââââââââââââââââââââââââââââââââââââ
    t0 = time.time()
    rf_ts, rf_te = train_predict_rf(train_eps + val_eps, test_eps, seed=seed)
    rf_te = np.maximum(rf_te, rf_ts)
    results["RF"] = compute_metrics(rf_ts, rf_te, ts_true, te_true)
    print(f"  RF done ({time.time()-t0:.1f}s)  MAE={results['RF'][0]:.2f}")

    # ââ 3. LSTM (plain, Exp 2 baseline) âââââââââââââââââââââââââââââââââââ
    t0 = time.time()
    lstm_ts, lstm_te = train_predict_lstm(train_eps + val_eps, test_eps,
                                           T=T, epochs=25, seed=seed)
    lstm_te = np.maximum(lstm_te, lstm_ts)
    results["LSTM"] = compute_metrics(lstm_ts, lstm_te, ts_true, te_true)
    print(f"  LSTM done ({time.time()-t0:.1f}s)  MAE={results['LSTM'][0]:.2f}")

    # ââ 4. PIML (original, Exp 2 baseline) ââââââââââââââââââââââââââââââââ
    t0 = time.time()
    piml_net = train_piml(train_eps, val_eps, T=T, epochs=120,
                           n_restarts=2, seed=seed, T0_prior=T*0.25)
    piml_ts, piml_te, _, _ = predict_piml(piml_net, test_eps)
    piml_te = np.maximum(piml_te, piml_ts)
    results["PIML"] = compute_metrics(piml_ts, piml_te, ts_true, te_true)
    print(f"  PIML done ({time.time()-t0:.1f}s)  MAE={results['PIML'][0]:.2f}")

    # ââ 5. MultiAgent (plain median, Exp 2 baseline) ââââââââââââââââââââââ
    ma_ts = np.median(np.stack([gb_ts, rf_ts, lstm_ts]), axis=0)
    ma_te = np.median(np.stack([gb_te, rf_te, lstm_te]), axis=0)
    ma_te = np.maximum(ma_te, ma_ts)
    results["MultiAgent"] = compute_metrics(ma_ts, ma_te, ts_true, te_true)
    print(f"  MultiAgent done  MAE={results['MultiAgent'][0]:.2f}")

    # ââ 6. RF_Physics âââââââââââââââââââââââââââââââââââââââââââââââââââââ
    t0 = time.time()
    rfp_ts, rfp_te = train_predict_rf_physics(train_eps + val_eps, test_eps, seed=seed)
    results["RF_Physics"] = compute_metrics(rfp_ts, rfp_te, ts_true, te_true)
    print(f"  RF_Physics done ({time.time()-t0:.1f}s)  MAE={results['RF_Physics'][0]:.2f}")

    # ââ 7. LSTM_Physics âââââââââââââââââââââââââââââââââââââââââââââââââââ
    t0 = time.time()
    lstmp_ts, lstmp_te = train_predict_lstm_physics(
        train_eps + val_eps, test_eps, T=T,
        epochs=60, seed=seed, physics_weight=0.3)
    results["LSTM_Physics"] = compute_metrics(lstmp_ts, lstmp_te, ts_true, te_true)
    print(f"  LSTM_Physics done ({time.time()-t0:.1f}s)  MAE={results['LSTM_Physics'][0]:.2f}")

    # ââ 8. PIML_v2 ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    t0 = time.time()
    piml2_net = train_piml_v2(train_eps, val_eps, T=T, epochs=150,
                               n_restarts=2, seed=seed)
    piml2_ts, piml2_te, _, _ = predict_piml_v2(piml2_net, test_eps)
    results["PIML_v2"] = compute_metrics(piml2_ts, piml2_te, ts_true, te_true)
    print(f"  PIML_v2 done ({time.time()-t0:.1f}s)  MAE={results['PIML_v2'][0]:.2f}")

    # ââ 9. MultiAgent_Physics âââââââââââââââââââââââââââââââââââââââââââââ
    t0 = time.time()
    map_ts, map_te = predict_multiagent_physics(
        test_eps,
        gb_ts, gb_te,
        rfp_ts, rfp_te,
        lstmp_ts, lstmp_te,
        T=T, refine=True)
    results["MultiAgent_Physics"] = compute_metrics(map_ts, map_te, ts_true, te_true)
    print(f"  MultiAgent_Physics done ({time.time()-t0:.1f}s)  MAE={results['MultiAgent_Physics'][0]:.2f}")

    return results


# ââ Main ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model_names = ["GreyBox", "RF", "LSTM", "PIML", "MultiAgent",
                   "RF_Physics", "LSTM_Physics", "PIML_v2", "MultiAgent_Physics"]
    metric_names = ["MAE", "RMSE", "R2", "PCR", "mean_tIoU", "tIoU_50", "tIoU_70"]

    all_results = {m: [] for m in model_names}

    for seed in SEEDS:
        res = run_seed(seed)
        for m in model_names:
            all_results[m].append(res[m])

    print(f"\n{'='*70}")
    print("EXPERIMENT 3 SUMMARY (mean +/- std over seeds)")
    print(f"{'='*70}")
    header = f"{'Model':<22}" + "".join(f"{mn:>12}" for mn in metric_names)
    print(header)
    print("-" * len(header))

    csv_lines = ["Model," + ",".join(metric_names)]
    for m in model_names:
        vals = np.array(all_results[m])   # (n_seeds, 7)
        means = vals.mean(axis=0)
        stds  = vals.std(axis=0)
        row   = f"{m:<22}"
        csv_parts = [m]
        for i, mn in enumerate(metric_names):
            if mn in ("PCR", "tIoU_50", "tIoU_70"):
                row += f"{means[i]:>11.1f}%"
                csv_parts.append(f"{means[i]:.1f} +/- {stds[i]:.1f}")
            elif mn == "mean_tIoU":
                row += f"{means[i]:>12.3f}"
                csv_parts.append(f"{means[i]:.3f} +/- {stds[i]:.3f}")
            elif mn == "R2":
                row += f"{means[i]:>12.3f}"
                csv_parts.append(f"{means[i]:.3f} +/- {stds[i]:.3f}")
            else:
                row += f"{means[i]:>12.1f}"
                csv_parts.append(f"{means[i]:.1f} +/- {stds[i]:.1f}")
        print(row)
        csv_lines.append(",".join(csv_parts))

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
