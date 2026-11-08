"""
Models compared in the reproduction:

1. PIML          - small MLP predicts (ts, te, theta_down, theta_up) from the
                    observed series; we FORWARD-SIMULATE b_hat(t) with the
                    known ODE (Eqs 1,2,6,7) from those parameters and train
                    on reconstruction error. This bakes the physics in by
                    construction (same spirit as the paper's hybrid loss),
                    and also gives trivial PCR=100% (te>=ts, c in [0,1] always).
2. GreyBox       - no neural net at all: per-episode scipy least-squares fit
                    of the same 4 parameters directly against the observed b(t).
                    Tests whether PIML's neural net adds anything over plain
                    curve-fitting (our critique #2).
3. RandomForest  - sklearn RF on hand-crafted summary features -> (ts, te) directly.
                    No physics, can predict te<ts.
4. LSTM          - small PyTorch LSTM over the raw series -> (ts, te) directly.
                    No physics.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor

from simulator import alpha_window, K_CAP, KAPPA


# ---------- shared forward simulator (used by PIML and GreyBox) ----------

def forward_simulate(t, lambda_obs, ts, te, theta_down, theta_up, kappa=KAPPA):
    """Given lambda(t) and candidate (ts, te, theta_down, theta_up), simulate
    c(t) and b_hat(t) via the same Euler scheme as the ground truth."""
    T = len(t)
    alpha = alpha_window(t, ts, te, kappa)
    c = np.zeros(T)
    c[0] = 1.0
    for i in range(1, T):
        dc = -theta_down * alpha[i - 1] * c[i - 1] + theta_up * (1 - alpha[i - 1]) * (1 - c[i - 1])
        c[i] = np.clip(c[i - 1] + dc, 0.0, 1.0)
    r_hat = K_CAP * c
    b_hat = np.zeros(T)
    for i in range(1, T):
        db = lambda_obs[i - 1] - r_hat[i - 1]
        b_hat[i] = max(0.0, b_hat[i - 1] + db)
    return b_hat, r_hat, c


# ---------- 1. PIML: amortized inference network + physics forward-sim ----------

class PIMLNet(nn.Module):
    """2 hidden layers x 32 units, tanh - matches the paper's architecture.
    Input: summary features of (lambda_obs, r_obs, b); Output: 4 raw params,
    squashed into valid ranges (ts,te in [0,T], thetas in [0.05,1.0])."""

    def __init__(self, in_dim, T):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        raw = self.net(x)
        ts = torch.sigmoid(raw[:, 0]) * self.T
        dur = torch.sigmoid(raw[:, 1]) * self.T          # te = ts + dur  (enforces te>=ts: PCR=100% by construction)
        te = ts + dur
        theta_down = 0.05 + 0.95 * torch.sigmoid(raw[:, 2])
        theta_up = 0.05 + 0.95 * torch.sigmoid(raw[:, 3])
        return ts, te, theta_down, theta_up


def make_features(ep):
    """Hand-crafted summary features (same ones used by RF baseline)."""
    b, lam, r = ep["b"], ep["lambda_obs"], ep["r_obs"]
    T = len(b)
    # coarse-binned backlog + delivery profile (10 bins) + global stats
    nb = 10
    bins_b = [b[i * T // nb:(i + 1) * T // nb].mean() for i in range(nb)]
    bins_r = [r[i * T // nb:(i + 1) * T // nb].mean() for i in range(nb)]
    feats = bins_b + bins_r + [b.max(), b.mean(), lam.mean(), r.mean(), r.min()]
    return np.array(feats, dtype=np.float32)


def soft_ode_forward_torch(t, lambda_obs, ts, te, theta_down, theta_up, kappa=KAPPA):
    """Differentiable (torch) version of forward_simulate, batched, for training.
    Builds the trajectory with list-append + stack (no in-place slice writes)
    so autograd can backprop through the whole Euler rollout."""
    B = ts.shape[0]
    T = len(t)
    t_t = torch.as_tensor(t, dtype=torch.float32)
    lam_t = torch.as_tensor(lambda_obs, dtype=torch.float32)  # (B,T)
    alpha = torch.sigmoid(kappa * (t_t[None, :] - ts[:, None])) - torch.sigmoid(kappa * (t_t[None, :] - te[:, None]))

    c_list = [torch.ones(B)]
    for i in range(1, T):
        c_prev = c_list[-1]
        dc = -theta_down * alpha[:, i - 1] * c_prev + theta_up * (1 - alpha[:, i - 1]) * (1 - c_prev)
        c_list.append(torch.clamp(c_prev + dc, 0.0, 1.0))
    c = torch.stack(c_list, dim=1)  # (B,T)
    r_hat = K_CAP * c

    b_list = [torch.zeros(B)]
    for i in range(1, T):
        b_prev = b_list[-1]
        db = lam_t[:, i - 1] - r_hat[:, i - 1]
        b_list.append(torch.clamp(b_prev + db, min=0.0))
    b_hat = torch.stack(b_list, dim=1)
    return b_hat


def _train_piml_once(train_eps, val_eps, T, epochs, lr, T0_prior, seed, scale, verbose=False):
    torch.manual_seed(seed)
    X_train = torch.tensor(np.stack([make_features(e) for e in train_eps]))
    X_val = torch.tensor(np.stack([make_features(e) for e in val_eps]))
    b_train = torch.tensor(np.stack([e["b"] for e in train_eps]), dtype=torch.float32) / scale
    b_val = torch.tensor(np.stack([e["b"] for e in val_eps]), dtype=torch.float32) / scale
    lam_train = np.stack([e["lambda_obs"] for e in train_eps])
    lam_val = np.stack([e["lambda_obs"] for e in val_eps])
    t = np.arange(T, dtype=np.float32)

    net = PIMLNet(in_dim=X_train.shape[1], T=T)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_state, patience, bad = np.inf, None, 20, 0
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        ts, te, th_d, th_u = net(X_train)
        b_hat = soft_ode_forward_torch(t, lam_train, ts, te, th_d, th_u) / scale
        l_data = ((b_hat - b_train) ** 2).mean()                       # Ldata (+ Lode: same term, since b_hat obeys the ODE by construction)
        l_prior = ((te - ts - T0_prior) ** 2).mean() * 1e-5            # Lprior: soft duration prior
        loss = l_data + l_prior
        loss.backward()
        opt.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            ts_v, te_v, thd_v, thu_v = net(X_val)
            b_hat_v = soft_ode_forward_torch(t, lam_val, ts_v, te_v, thd_v, thu_v) / scale
            val_loss = ((b_hat_v - b_val) ** 2).mean().item()
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                break
        if verbose and ep % 20 == 0:
            print(f"  [seed {seed}] epoch {ep}: train {loss.item():.5f} val {val_loss:.5f}")

    net.load_state_dict(best_state)
    return net, best_val


def train_piml(train_eps, val_eps, T, epochs=200, lr=5e-3, T0_prior=250.0, seed=0,
                n_restarts=3, verbose=False):
    """Trains PIML with multiple random restarts (fixing the scale of the
    reconstruction loss for stable gradients) and keeps the restart with the
    lowest validation loss. This is still 'one simple net', just given a fair
    chance to escape a bad initialization, which our first pass showed is
    the real problem (PIML's error varied 3x across seeds -- an optimization
    issue, not a representational one)."""
    scale = max(e["b"].max() for e in train_eps + val_eps) + 1e-6
    best_net, best_val = None, np.inf
    for r in range(n_restarts):
        net, val_loss = _train_piml_once(train_eps, val_eps, T, epochs, lr, T0_prior,
                                          seed=seed * 100 + r, scale=scale, verbose=verbose)
        if val_loss < best_val:
            best_net, best_val = net, val_loss
    return best_net


def predict_piml(net, eps):
    X = torch.tensor(np.stack([make_features(e) for e in eps]))
    with torch.no_grad():
        ts, te, th_d, th_u = net(X)
    return ts.numpy(), te.numpy(), th_d.numpy(), th_u.numpy()


# ---------- 2. Grey-box: per-episode scipy least-squares (no NN at all) ----------

def fit_greybox(ep, T0_prior=250.0):
    t, lam, b_obs = ep["t"], ep["lambda_obs"], ep["b"]

    def resid(p):
        ts, te, th_d, th_u = p
        te = max(te, ts)  # keep feasible
        b_hat, _, _ = forward_simulate(t, lam, ts, te, max(th_d, 0.01), max(th_u, 0.01))
        data_resid = (b_hat - b_obs)
        prior_resid = np.array([0.003 * (te - ts - T0_prior)])
        return np.concatenate([data_resid, prior_resid])

    T = len(t)
    x0 = [T * 0.3, T * 0.3 + T0_prior, 0.5, 0.5]
    bounds = ([0, 0, 0.01, 0.01], [T, 2 * T, 1.0, 1.0])
    res = least_squares(resid, x0, bounds=bounds, max_nfev=300)
    ts, te, th_d, th_u = res.x
    return ts, max(te, ts), th_d, th_u


# ---------- 3. Random Forest (direct, no physics) ----------

def train_predict_rf(train_eps, test_eps, seed=0):
    Xtr = np.stack([make_features(e) for e in train_eps])
    ytr = np.stack([[e["ts"], e["te"]] for e in train_eps])
    Xte = np.stack([make_features(e) for e in test_eps])
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    return pred[:, 0], pred[:, 1]


# ---------- 4. LSTM (direct, no physics) ----------

class LSTMNet(nn.Module):
    def __init__(self, T, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        self.T = T

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        raw = self.head(h[-1])
        ts = torch.sigmoid(raw[:, 0]) * self.T
        te = torch.sigmoid(raw[:, 1]) * self.T
        return ts, te


def train_predict_lstm(train_eps, test_eps, T, epochs=25, lr=5e-3, seed=0, stride=3):
    """Downsample the sequence (stride) before feeding the LSTM: a 900-step
    CPU LSTM rollout is the dominant cost in this whole experiment, and the
    disruption window is smooth enough that every 3rd point is plenty of
    signal for this baseline."""
    torch.manual_seed(seed)

    def to_seq(eps):
        arrs = [np.stack([e["lambda_obs"][::stride], e["r_obs"][::stride], e["b"][::stride] / 300.0], axis=-1)
                for e in eps]
        return torch.tensor(np.stack(arrs), dtype=torch.float32)

    Xtr = to_seq(train_eps)
    ytr = torch.tensor(np.stack([[e["ts"], e["te"]] for e in train_eps]), dtype=torch.float32)
    Xte = to_seq(test_eps)

    net = LSTMNet(T)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        ts, te = net(Xtr)
        loss = ((ts - ytr[:, 0]) ** 2 + (te - ytr[:, 1]) ** 2).mean()
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        ts, te = net(Xte)
    return ts.numpy(), te.numpy()
