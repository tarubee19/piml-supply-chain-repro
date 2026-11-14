"""
Full model comparison on the realistic synthetic dataset.

Models compared:
  1. PIML (paper's architecture: 2x32-tanh MLP + physics ODE forward sim)
  2. GreyBox (per-episode scipy least-squares curve fit)
  3. RandomForest (hand-crafted features -> ts, te)
  4. LSTM (raw series -> ts, te)
  5. MultiAgent Ensemble (GreyBox + RF + LSTM as specialist agents,
     median-based coordinator aggregation)

Metrics (matching the paper):
  - MAE, RMSE, R^2 on (ts, te) predictions
  - PCR: physical consistency rate (te >= ts)
  - mean tIoU, tIoU@0.5, tIoU@0.7
"""
import sys, os
import numpy as np
import time

sys.path.insert(0, os.path.dirname(__file__))

from realistic_simulator import make_realistic_dataset
from models import (train_piml, predict_piml, fit_greybox,
                    train_predict_rf, train_predict_lstm)

# Reduced from T=900 to T=300 for computational feasibility
# (the ODE forward sim in PyTorch is O(T) sequential per sample per epoch)
N_EPISODES = 150
T = 300
SEEDS = [0, 1, 2]


def temporal_iou(ts_pred, te_pred, ts_true, te_true):
    inter_start = max(ts_pred, ts_true)
    inter_end = min(te_pred, te_true)
    intersection = max(0, inter_end - inter_start)
    union = max(te_pred, te_true) - min(ts_pred, ts_true)
    if union <= 0:
        return 0.0
    return intersection / union


def evaluate(ts_pred, te_pred, ts_true, te_true):
    n = len(ts_true)
    pred = np.concatenate([ts_pred, te_pred])
    true = np.concatenate([ts_true, te_true])

    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot

    pcr = np.mean(te_pred >= ts_pred) * 100.0

    tious = [temporal_iou(ts_pred[i], te_pred[i], ts_true[i], te_true[i])
             for i in range(n)]
    mean_tiou = np.mean(tious)
    tiou50 = np.mean([t >= 0.5 for t in tious]) * 100.0
    tiou70 = np.mean([t >= 0.7 for t in tious]) * 100.0

    return dict(MAE=mae, RMSE=rmse, R2=r2, PCR=pcr,
                mean_tIoU=mean_tiou, tIoU_50=tiou50, tIoU_70=tiou70)


def run_one_seed(seed, verbose=True):
    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"  SEED {seed}", flush=True)
        print(f"{'='*60}", flush=True)

    t0 = time.time()
    episodes = make_realistic_dataset(n_episodes=N_EPISODES, T=T, seed=seed)
    n_train = int(0.7 * N_EPISODES)
    n_val = int(0.15 * N_EPISODES)
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train:n_train + n_val]
    test_eps = episodes[n_train + n_val:]

    ts_true = np.array([e["ts"] for e in test_eps])
    te_true = np.array([e["te"] for e in test_eps])

    results = {}

    # --- 1. GreyBox ---
    print("  Running GreyBox...", flush=True)
    gb_ts, gb_te = [], []
    for ep in test_eps:
        ts_g, te_g, _, _ = fit_greybox(ep, T0_prior=T * 0.28)
        gb_ts.append(ts_g)
        gb_te.append(te_g)
    gb_ts, gb_te = np.array(gb_ts), np.array(gb_te)
    results["GreyBox"] = evaluate(gb_ts, gb_te, ts_true, te_true)
    print(f"    GreyBox MAE={results['GreyBox']['MAE']:.1f}", flush=True)

    # --- 2. RandomForest ---
    print("  Running RF...", flush=True)
    rf_ts, rf_te = train_predict_rf(train_eps, test_eps, seed=seed)
    results["RF"] = evaluate(rf_ts, rf_te, ts_true, te_true)
    print(f"    RF MAE={results['RF']['MAE']:.1f}", flush=True)

    # --- 3. LSTM ---
    print("  Running LSTM...", flush=True)
    lstm_ts, lstm_te = train_predict_lstm(train_eps, test_eps, T, seed=seed,
                                          epochs=30, stride=2)
    results["LSTM"] = evaluate(lstm_ts, lstm_te, ts_true, te_true)
    print(f"    LSTM MAE={results['LSTM']['MAE']:.1f}", flush=True)

    # --- 4. PIML ---
    print("  Running PIML...", flush=True)
    piml_net = train_piml(train_eps, val_eps, T, epochs=120, seed=seed,
                          n_restarts=2, T0_prior=T * 0.28, verbose=False)
    piml_ts, piml_te, _, _ = predict_piml(piml_net, test_eps)
    results["PIML"] = evaluate(piml_ts, piml_te, ts_true, te_true)
    print(f"    PIML MAE={results['PIML']['MAE']:.1f}", flush=True)

    # --- 5. MultiAgent Ensemble ---
    # Concept: GreyBox, RF, LSTM as independent specialist agents,
    # coordinator aggregates via element-wise median
    print("  Running MultiAgent Ensemble...", flush=True)
    ma_ts = np.median(np.stack([gb_ts, rf_ts, lstm_ts]), axis=0)
    ma_te = np.median(np.stack([gb_te, rf_te, lstm_te]), axis=0)
    results["MultiAgent"] = evaluate(ma_ts, ma_te, ts_true, te_true)
    print(f"    MultiAgent MAE={results['MultiAgent']['MAE']:.1f}", flush=True)

    elapsed = time.time() - t0
    print(f"  Seed {seed} done in {elapsed:.0f}s", flush=True)

    return results


def main():
    all_results = {m: [] for m in ["GreyBox", "RF", "LSTM", "PIML", "MultiAgent"]}

    for seed in SEEDS:
        seed_results = run_one_seed(seed)
        for model, metrics in seed_results.items():
            all_results[model].append(metrics)

    # Aggregate across seeds
    print("\n" + "=" * 80, flush=True)
    print("  FINAL RESULTS - Realistic Synthetic Dataset (mean +/- std over 3 seeds)", flush=True)
    print("=" * 80, flush=True)

    metric_names = ["MAE", "RMSE", "R2", "PCR", "mean_tIoU", "tIoU_50", "tIoU_70"]
    header = f"{'Model':<12}" + "".join(f"{m:>14}" for m in metric_names)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    csv_lines = ["Model,MAE,RMSE,R2,PCR,mean_tIoU,tIoU_50,tIoU_70"]

    for model in ["GreyBox", "RF", "LSTM", "PIML", "MultiAgent"]:
        vals = {m: [r[m] for r in all_results[model]] for m in metric_names}
        means = {m: np.mean(vals[m]) for m in metric_names}
        stds = {m: np.std(vals[m]) for m in metric_names}

        row = f"{model:<12}"
        csv_parts = [model]
        for m in metric_names:
            if m == "PCR":
                row += f"{'':>4}{means[m]:.1f}"
                csv_parts.append(f"{means[m]:.1f}")
            elif m in ("R2", "mean_tIoU"):
                row += f"  {means[m]:.3f}+/-{stds[m]:.3f}"
                csv_parts.append(f"{means[m]:.3f} +/- {stds[m]:.3f}")
            else:
                row += f"  {means[m]:.1f}+/-{stds[m]:.1f}"
                csv_parts.append(f"{means[m]:.1f} +/- {stds[m]:.1f}")
        print(row, flush=True)
        csv_lines.append(",".join(csv_parts))

    # Save CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results_realistic")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "realistic_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\nResults saved to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
