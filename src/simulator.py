"""
Open simulator reproducing the backlog/disruption dynamics of
Matsuno et al. (2026), "PIML-Driven Supply Chain Disruption Estimation
and Capacity-Sharing Optimization," ASME JCISE 26(11).

Physics (from the paper):
    db/dt   = lambda(t) - r(t),            b(t) >= 0        (Eq. 1)
    r(t)    = (K / tau) * c(t) * a(t)                        (Eq. 2)
    alpha(t)= sigmoid(kappa*(t-ts)) - sigmoid(kappa*(t-te))   (Eq. 6)
    dc/dt   = -theta_down*alpha(t)*c(t) + theta_up*(1-alpha(t))*(1-c(t))  (Eq. 7)

We simulate many independent "supplier episodes" with randomised
(ts, te, theta_down, theta_up), a noisy order-inflow process lambda(t),
and Euler-integrate b(t) and c(t). Each episode is one labelled example:
    X = {lambda(t), r_obs(t), b(t)}_t          (what the focal firm sees)
    y = (ts, te, theta_down, theta_up)          (hidden ground truth)
"""
import numpy as np

KAPPA = 8.0          # sigmoid sharpness (paper's best value, Table 6)
K_CAP = 1.0           # normalized production capacity K/tau (a(t)=1, single product)
NOISE_STD = 0.03      # observation noise on lambda and r


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def alpha_window(t, ts, te, kappa=KAPPA):
    return sigmoid(kappa * (t - ts)) - sigmoid(kappa * (t - te))


def simulate_episode(T=900, dt=1.0, rng=None, has_disruption=True):
    """Simulate one supplier's transaction-level time series.

    Returns dict with time series (lambda_obs, r_obs, b) and
    ground-truth parameters (ts, te, theta_down, theta_up, c_true).
    """
    rng = rng or np.random.default_rng()
    t = np.arange(T) * dt

    # Base order inflow: steady demand + mild seasonality + noise.
    base_lambda = 0.8 + 0.1 * np.sin(2 * np.pi * t / 180.0)
    lambda_true = base_lambda + rng.normal(0, 0.03, size=T)
    lambda_true = np.clip(lambda_true, 0.05, None)

    if has_disruption:
        # Disruption window: random onset after a short warm-up,
        # random duration consistent with Table 9/10-style magnitudes.
        ts = rng.uniform(0.15, 0.55) * T
        duration = rng.uniform(0.10, 0.40) * T
        te = ts + duration
        theta_down = rng.uniform(0.2, 0.8)   # matches paper's init range
        theta_up = rng.uniform(0.2, 0.8)
    else:
        ts, te = -1e6, -1e6  # never triggers -> c(t) stays at 1
        theta_down = rng.uniform(0.2, 0.8)
        theta_up = rng.uniform(0.2, 0.8)

    c = np.zeros(T)
    c[0] = 1.0
    alpha = alpha_window(t, ts, te)
    for i in range(1, T):
        dc = -theta_down * alpha[i - 1] * c[i - 1] + theta_up * (1 - alpha[i - 1]) * (1 - c[i - 1])
        c[i] = np.clip(c[i - 1] + dc * dt, 0.0, 1.0)

    r_true = K_CAP * c  # a(t)=1, single aggregated product
    b = np.zeros(T)
    for i in range(1, T):
        db = lambda_true[i - 1] - r_true[i - 1]
        b[i] = max(0.0, b[i - 1] + db * dt)

    # What the focal firm actually observes (small sensor noise).
    lambda_obs = np.clip(lambda_true + rng.normal(0, NOISE_STD, size=T), 0, None)
    r_obs = np.clip(r_true + rng.normal(0, NOISE_STD, size=T), 0, None)

    return dict(
        t=t, lambda_obs=lambda_obs, r_obs=r_obs, b=b, c_true=c,
        ts=ts, te=te, theta_down=theta_down, theta_up=theta_up,
        duration=te - ts if has_disruption else 0.0, has_disruption=has_disruption,
    )


def make_dataset(n_episodes=400, T=900, seed=0, frac_no_disruption=0.0):
    rng = np.random.default_rng(seed)
    episodes = []
    n_no = int(n_episodes * frac_no_disruption)
    for i in range(n_episodes):
        has_dis = i >= n_no
        episodes.append(simulate_episode(T=T, rng=rng, has_disruption=has_dis))
    return episodes


if __name__ == "__main__":
    eps = make_dataset(n_episodes=5, T=900, seed=42)
    for e in eps:
        print(f"ts={e['ts']:.1f} te={e['te']:.1f} dur={e['duration']:.1f} "
              f"theta_down={e['theta_down']:.2f} theta_up={e['theta_up']:.2f} "
              f"max_b={e['b'].max():.2f}")
