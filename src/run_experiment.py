"""
X1: Reproduce the paper's core comparison (its Table 7 & 8) on our OPEN
simulator, plus X2's grey-box baseline (our addition / critique test).

Usage: python3 run_experiment.py
Writes results/table7_8.csv and prints a formatted summary.
"""
import time
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(4)  # avoid CPU oversubscription slowing the LSTM/PIML rollouts

from simulator import make_dataset
from models import train_piml, predict_piml, fit_greybox, train_predict_rf, train_predict_lstm

T = 900
N_EPISODES = 200   # 200 labelled episodes (vs ~1 window/supplier in the original paper)
SEEDS = [0, 1, 2, 3, 4]


def metrics(ts_true, te_true, ts_pred, te_pred):
    dur_true = te_true - ts_true
    dur_pred = te_pred - ts_pred
    err_ts = ts_pred - ts_true
    err_te = te_pred - te_true
    err_dur = dur_pred - dur_true
    all_err = np.concatenate([err_ts, err_te])
    mae = np.mean(np.abs(all_err))
    rmse = np.sqrt(np.mean(all_err ** 2))

    y_true = np.concatenate([ts_true, te_true])
    y_pred = np.concatenate([ts_pred, te_pred])
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    pcr = np.mean(te_pred >= ts_pred)  # physical consistency: end after start

    # temporal IoU
    inter = np.maximum(0, np.minimum(te_true, te_pred) - np.maximum(ts_true, ts_pred))
    union = np.maximum(te_true, te_pred) - np.minimum(ts_true, ts_pred)
    union = np.where(union <= 0, 1e-6, union)
    tiou = inter / union
    mean_tiou = tiou.mean()
    tiou_50 = (tiou >= 0.5).mean() * 100
    tiou_70 = (tiou >= 0.7).mean() * 100

    return dict(MAE=mae, RMSE=rmse, R2=r2, PCR=pcr * 100,
                mean_tIoU=mean_tiou, tIoU_50=tiou_50, tIoU_70=tiou_70)


def run_seed(seed):
    eps = make_dataset(n_episodes=N_EPISODES, T=T, seed=seed)
    n = len(eps)
    idx = np.random.default_rng(seed).permutation(n)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    train_eps = [eps[i] for i in idx[:n_train]]
    val_eps = [eps[i] for i in idx[n_train:n_train + n_val]]
    test_eps = [eps[i] for i in idx[n_train + n_val:]]

    ts_true = np.array([e["ts"] for e in test_eps])
    te_true = np.array([e["te"] for e in test_eps])

    results = {}

    # RF
    ts_p, te_p = train_predict_rf(train_eps, test_eps, seed=seed)
    results["RF"] = metrics(ts_true, te_true, ts_p, te_p)

    # LSTM
    ts_p, te_p = train_predict_lstm(train_eps, test_eps, T, seed=seed)
    results["LSTM"] = metrics(ts_true, te_true, ts_p, te_p)

    # Grey-box (per test episode, no training set needed)
    gb = [fit_greybox(e) for e in test_eps]
    ts_p = np.array([g[0] for g in gb])
    te_p = np.array([g[1] for g in gb])
    results["GreyBox"] = metrics(ts_true, te_true, ts_p, te_p)

    # PIML
    net = train_piml(train_eps, val_eps, T, seed=seed)
    ts_p, te_p, _, _ = predict_piml(net, test_eps)
    results["PIML"] = metrics(ts_true, te_true, ts_p, te_p)

    return results


def main():
    t0 = time.time()
    all_results = {m: [] for m in ["RF", "LSTM", "GreyBox", "PIML"]}
    for s in SEEDS:
        print(f"--- seed {s} ---")
        res = run_seed(s)
        for m, v in res.items():
            all_results[m].append(v)
            print(f"  {m:8s} MAE={v['MAE']:.1f} RMSE={v['RMSE']:.1f} R2={v['R2']:.3f} "
                  f"PCR={v['PCR']:.1f}% tIoU={v['mean_tIoU']:.3f}")

    rows = []
    for m, runs in all_results.items():
        df = pd.DataFrame(runs)
        row = {"Model": m}
        for col in df.columns:
            row[col] = f"{df[col].mean():.1f} ± {df[col].std():.1f}" if col in ("MAE", "RMSE") \
                else f"{df[col].mean():.3f} ± {df[col].std():.3f}" if col in ("R2", "mean_tIoU") \
                else f"{df[col].mean():.1f}"
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\n=== Reproduction: window-estimation comparison (5 seeds) ===")
    print(table.to_string(index=False))
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "table7_8.csv"), index=False)
    print(f"\nSaved to results/table7_8.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
