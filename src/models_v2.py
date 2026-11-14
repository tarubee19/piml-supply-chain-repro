"""
Experiment 3 â Physics-Informed Models.

Every model now has access to physics knowledge from the paper's ODE.
The key idea from the paper: supply-chain backlog follows a physics ODE,
so any model that ignores this is leaving signal on the table.

Changes vs models.py (Experiment 2):

RF_Physics  : RF trained on physics-derived features (residuals vs no-disruption
              baseline, estimated alpha window, ODE gradient features) instead of
              just raw statistical bins.

LSTM_Physics: LSTM trained with an extra physics-consistency loss: after predicting
              (ts, te), the ODE is forward-simulated and penalises mismatch with
              observed b(t). Same idea as PIML, but starting from LSTM's sequential
              encoding rather than a static feature vector.

PIML_v2     : Same as before but with better features (physics-residual input) and
              an extra te>=ts hard constraint via softplus duration.

MultiAgent_Physics: Runs GreyBox + RF_Physics + LSTM_Physics, then uses their
              median as a warm-start for a final short GreyBox refinement (10 iters).
              Best of data-driven + physics-corrected.

Shared physics utilities are at the top.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor

from simulator import alpha_window, K_CAP, KAPPA
from models import (forward_simulate, fit_greybox, soft_ode_forward_torch,
                    PIMLNet, _train_piml_once, train_piml, predict_piml,
                    train_predict_lstm, LSTMNet)


# âââââââââââââââââââââââââââââââââââââââââââââ
# PHYSICS UTILITIES
# âââââââââââââââââââââââââââââââââââââââââââââ

def physics_residual_features(ep, n_segments=10):
    """
    Compute features that encode the physics ODE structure.

    Key idea: if there were NO disruption, capacity c(t)=1 always, so
      r_baseline(t) = K_CAP * lambda_obs(t)   (demand-matched production)
      b_baseline(t) = integral of (lambda - r_baseline) = ~0

    The difference between observed b(t) and b_baseline reveals WHEN and
    HOW MUCH the disruption depressed production.

    Features:
    - Residual b(t) - b_baseline(t) in segments   â WHERE backlog diverges
    - d/dt(b_obs) - d/dt(b_baseline) in segments  â RATE of divergence
    - Estimated alpha(t) proxy from b-gradient     â WINDOW shape
    - Peak residual location (normalized)          â ts estimate
    - Recovery location (where residual shrinks)   â te estimate
    - Residual area (integral)                     
    - Max single-step backlog increase             
    - Max single-step backlog decrease             
    """
    t      = ep["t"]
    b      = ep["b"]
    lam    = ep["lambda_obs"]
    r_obs  = ep["r_obs"]
    T      = len(b)

    r_base = np.clip(lam, 0, None)
    b_base = np.zeros(T)
    for i in range(1, T):
        db = lam[i-1] - r_base[i-1]
        b_base[i] = max(0.0, b_base[i-1] + db)

    resid  = b - b_base
    db_obs = np.diff(b, prepend=b[0])
    db_bas = np.diff(b_base, prepend=b_base[0])
    grad_diff = db_obs - db_bas

    nb = n_segments
    seg_resid = [resid[i*T//nb:(i+1)*T//nb].mean() for i in range(nb)]
    seg_grad  = [grad_diff[i*T//nb:(i+1)*T//nb].mean() for i in range(nb)]

    pos_resid = np.clip(resid, 0, None)
    if pos_resid.sum() > 1e-6:
        weights = pos_resid / pos_resid.sum()
        centre  = float(np.dot(np.arange(T), weights))
        spread  = float(np.sqrt(np.dot((np.arange(T) - centre)**2, weights)))
    else:
        centre, spread = T/2, T/4

    thresh = 0.20 * max(pos_resid.max(), 1e-6)
    above  = np.where(pos_resid > thresh)[0]
    ts_proxy = float(above[0])  / T if len(above) else 0.3
    te_proxy = float(above[-1]) / T if len(above) else 0.7

    area         = float(resid.sum()) / T
    peak_resid   = float(pos_resid.max())
    max_drop     = float((-grad_diff).clip(0).max())
    max_rise     = float(grad_diff.clip(0).max())

    r_drop  = float((lam - r_obs).clip(0).mean())
    r_ratio = float(r_obs.min() / (lam.mean() + 1e-6))

    feats = (seg_resid + seg_grad +
             [centre/T, spread/T, ts_proxy, te_proxy,
              area, peak_resid, max_drop, max_rise,
              r_drop, r_ratio,
              b.max(), b.mean(), lam.mean(), r_obs.mean(), r_obs.min()])
    return np.array(feats, dtype=np.float32)


def train_predict_rf_physics(train_eps, test_eps, seed=0):
    Xtr = np.stack([physics_residual_features(e) for e in train_eps])
    ytr = np.stack([[e["ts"], e["te"]] for e in train_eps])
    Xte = np.stack([physics_residual_features(e) for e in test_eps])
    rf = RandomForestRegressor(n_estimators=300, max_depth=10,
                                min_samples_leaf=2, random_state=seed)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    ts_pred = pred[:, 0]
    te_pred = np.maximum(pred[:, 1], ts_pred)
    return ts_pred, te_pred


class LSTMPhysicsNet(nn.Module):
    def __init__(self, T, hidden=48):
        super().__init__()
        self.T = T
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden,
                            num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        raw = self.head(h[-1])
        ts = torch.sigmoid(raw[:, 0]) * self.T
        dur = torch.nn.functional.softplus(raw[:, 1]) * (self.T / 4)
        te = ts + dur
        return ts, te


def train_predict_lstm_physics(train_eps, test_eps, T, epochs=60,
                               lr=3e-3, seed=0, stride=3,
                               physics_weight=0.3):
    torch.manual_seed(seed)
    def to_seq(eps):
        arrs = []
        for e in eps:
            lam_n = e["lambda_obs"] / (e["lambda_obs"].mean() + 1e-6)
            r_n   = e["r_obs"]      / (e["lambda_obs"].mean() + 1e-6)
            b_n   = e["b"]          / (e["b"].max() + 1e-6)
            arrs.append(np.stack([lam_n[::stride], r_n[::stride], b_n[::stride]], axis=-1))
        return torch.tensor(np.stack(arrs), dtype=torch.float32)
    Xtr = to_seq(train_eps)
    ytr = torch.tensor(np.stack([[e["ts"], e["te"]] for e in train_eps]), dtype=torch.float32)
    Xte = to_seq(test_eps)
    lam_tr_np = np.stack([e["lambda_obs"] for e in train_eps])
    t_np = np.arange(T, dtype=np.float32)
    scale = max(e["b"].max() for e in train_eps) + 1e-6
    b_tr = torch.tensor(np.stack([e["b"] for e in train_eps]), dtype=torch.float32) / scale
    net = LSTMPhysicsNet(T)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss, best_state = float('inf'), None
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        ts_p, te_p = net(Xtr)
        l_sup = ((ts_p - ytr[:, 0])**2 + (te_p - ytr[:, 1])**2).mean()
        theta_d = torch.full((len(train_eps),), 0.5)
        theta_u = torch.full((len(train_eps),), 0.5)
        b_hat = soft_ode_forward_torch(t_np, lam_tr_np, ts_p, te_p, theta_d, theta_u) / scale
        l_phys = ((b_hat - b_tr)**2).mean()
        loss = l_sup + physics_weight * l_phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        ts_te, te_te = net(Xte)
    return ts_te.numpy(), te_te.numpy()


class PIMLNetV2(nn.Module):
    def __init__(self, in_dim, T):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),     nn.Tanh(),
            nn.Linear(64, 32),     nn.Tanh(),
            nn.Linear(32, 4),
        )
    def forward(self, x):
        raw = self.net(x)
        ts = torch.sigmoid(raw[:, 0]) * self.T
        dur = torch.nn.functional.softplus(raw[:, 1]) * (self.T / 4)
        te = ts + dur
        theta_down = 0.05 + 0.95 * torch.sigmoid(raw[:, 2])
        theta_up = 0.05 + 0.95 * torch.sigmoid(raw[:, 3])
        return ts, te, theta_down, theta_up


def _train_piml_v2_once(train_eps, val_eps, T, epochs, lr, T0_prior,
                         seed, scale, verbose=False):
    torch.manual_seed(seed)
    X_train = torch.tensor(np.stack([physics_residual_features(e) for e in train_eps]))
    X_val = torch.tensor(np.stack([physics_residual_features(e) for e in val_eps]))
    b_train = torch.tensor(np.stack([e["b"] for e in train_eps]), dtype=torch.float32) / scale
    b_val = torch.tensor(np.stack([e["b"] for e in val_eps]), dtype=torch.float32) / scale
    lam_train = np.stack([e["lambda_obs"] for e in train_eps])
    lam_val = np.stack([e["lambda_obs"] for e in val_eps])
    t = np.arange(T, dtype=np.float32)
    net = PIMLNetV2(in_dim=X_train.shape[1], T=T)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, bad = float('inf'), None, 0
    patience = 25
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        ts, te, th_d, th_u = net(X_train)
        b_hat = soft_ode_forward_torch(t, lam_train, ts, te, th_d, th_u) / scale
        l_data = ((b_hat - b_train)**2).mean()
        l_prior = ((te - ts - T0_prior)**2).mean() * 1e-5
        loss = l_data + l_prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            ts_v, te_v, thd_v, thu_v = net(X_val)
            b_hat_v = soft_ode_forward_torch(t, lam_val, ts_v, te_v, thd_v, thu_v) / scale
            val_loss = ((b_hat_v - b_val)**2).mean().item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                break
    net.load_state_dict(best_state)
    return net, best_val


def train_piml_v2(train_eps, val_eps, T, epochs=150, lr=3e-3,
                   T0_prior=None, seed=0, n_restarts=3, verbose=False):
    if T0_prior is None:
        T0_prior = T * 0.25
    scale = max(e["b"].max() for e in train_eps + val_eps) + 1e-6
    best_net, best_val = None, float('inf')
    for r in range(n_restarts):
        net, val = _train_piml_v2_once(train_eps, val_eps, T, epochs, lr,
                                        T0_prior, seed=seed*100+r, scale=scale, verbose=verbose)
        if val < best_val:
            best_net, best_val = net, val
    return best_net


def predict_piml_v2(net, eps):
    X = torch.tensor(np.stack([physics_residual_features(e) for e in eps]))
    with torch.no_grad():
        ts, te, th_d, th_u = net(X)
    return ts.numpy(), te.numpy(), th_d.numpy(), th_u.numpy()


def fit_greybox_warm(ep, ts_init, te_init, max_nfev=100, T0_prior=None):
    t, lam, b_obs = ep["t"], ep["lambda_obs"], ep["b"]
    T = len(t)
    if T0_prior is None:
        T0_prior = T * 0.25
    def resid(p):
        ts, te, th_d, th_u = p
        te = max(te, ts + 1)
        b_hat, _, _ = forward_simulate(t, lam, ts, te, max(th_d, 0.01), max(th_u, 0.01))
        return np.concatenate([b_hat - b_obs, [0.001 * (te - ts - T0_prior)]])
    ts0 = float(np.clip(ts_init, 0.05*T, 0.80*T))
    te0 = float(np.clip(te_init, ts0+1, 1.50*T))
    x0 = [ts0, te0, 0.5, 0.5]
    bounds = ([0, 0, 0.01, 0.01], [T, 2*T, 1.0, 1.0])
    res = least_squares(resid, x0, bounds=bounds, max_nfev=max_nfev)
    ts, te, th_d, th_u = res.x
    return ts, max(te, ts+1), th_d, th_u


def predict_multiagent_physics(test_eps, ts_gb, te_gb, ts_rf, te_rf,
                                ts_lstm, te_lstm, T, refine=True):
    n = len(test_eps)
    ts_stack = np.stack([ts_gb, ts_rf, ts_lstm], axis=0)
    te_stack = np.stack([te_gb, te_rf, te_lstm], axis=0)
    ts_med = np.median(ts_stack, axis=0)
    te_med = np.median(te_stack, axis=0)
    if not refine:
        return ts_med, te_med
    ts_out, te_out = np.zeros(n), np.zeros(n)
    T0_prior = T * 0.25
    for i, ep in enumerate(test_eps):
        ts_r, te_r, _, _ = fit_greybox_warm(ep, ts_med[i], te_med[i],
                                              max_nfev=80, T0_prior=T0_prior)
        ts_out[i] = ts_r
        te_out[i] = te_r
    return ts_out, te_out
