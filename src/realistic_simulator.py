"""
Realistic synthetic dataset generator for supply chain disruption estimation.

Improvements over the basic simulator (based on findings from applying the
paper's models to the real DataCo Smart Supply Chain dataset):

1. Weekly seasonality -- real order data shows strong day-of-week patterns
2. Elastic demand-capacity correlation -- in real supply chains, production
   rate r(t) tracks demand lambda(t) elastically even outside disruptions,
   rather than being a fixed constant K*c(t)
3. Higher, heterogeneous noise -- real data is much noisier than NOISE_STD=0.03
4. Demand trend / drift -- slow random-walk drift in base demand level
5. Partial observability -- r_obs has measurement lag/smoothing
6. Variable disruption severity -- some disruptions are mild, some catastrophic

All while remaining physically sane: b(t) >= 0, c(t) in [0,1], r(t) >= 0.
"""
import numpy as np
from simulator import sigmoid, alpha_window, KAPPA, K_CAP


def simulate_realistic_episode(T=900, dt=1.0, rng=None, has_disruption=True):
    """Generate one episode with realistic real-world complications."""
    rng = rng or np.random.default_rng()
    t = np.arange(T) * dt

    # --- 1. Demand process with weekly seasonality + trend drift ---
    base_lambda = 0.8 + 0.1 * np.sin(2 * np.pi * t / 180.0)
    weekly = 0.08 * np.sin(2 * np.pi * t / 7.0) + 0.04 * np.cos(4 * np.pi * t / 7.0)
    base_lambda += weekly
    drift = np.cumsum(rng.normal(0, 0.002, size=T))
    drift = drift - drift.mean()
    base_lambda += drift
    noise_level = rng.uniform(0.08, 0.15)
    lambda_true = base_lambda + rng.normal(0, noise_level, size=T)
    lambda_true = np.clip(lambda_true, 0.05, None)

    # --- 2. Disruption parameters ---
    if has_disruption:
        ts = rng.uniform(0.15, 0.55) * T
        duration = rng.uniform(0.10, 0.40) * T
        te = ts + duration
        theta_down = rng.uniform(0.15, 0.90)
        theta_up = rng.uniform(0.15, 0.90)
    else:
        ts, te = -1e6, -1e6
        theta_down = rng.uniform(0.2, 0.8)
        theta_up = rng.uniform(0.2, 0.8)

    # --- 3. Capacity dynamics (same ODE as paper) ---
    c = np.zeros(T)
    c[0] = 1.0
    alpha = alpha_window(t, ts, te)
    for i in range(1, T):
        dc = (-theta_down * alpha[i-1] * c[i-1]
              + theta_up * (1 - alpha[i-1]) * (1 - c[i-1]))
        c[i] = np.clip(c[i-1] + dc * dt, 0.0, 1.0)

    # --- 4. Production rate with elastic demand tracking ---
    # r(t) = K * c(t) * elastic_factor(t) where elastic_factor tracks demand
    elastic_target = lambda_true / K_CAP
    elastic_factor = np.ones(T)
    elasticity = rng.uniform(0.3, 0.7)
    smooth_tau = rng.uniform(5, 20)
    for i in range(1, T):
        alpha_e = 1.0 - np.exp(-dt / smooth_tau)
        elastic_factor[i] = (elastic_factor[i-1]
                             + alpha_e * (elastic_target[i] - elastic_factor[i-1]))
    effective_factor = (1 - elasticity) + elasticity * elastic_factor
    r_true = K_CAP * c * np.clip(effective_factor, 0.3, 1.5)

    # --- 5. Backlog (same ODE as paper) ---
    b = np.zeros(T)
    for i in range(1, T):
        db = lambda_true[i-1] - r_true[i-1]
        b[i] = max(0.0, b[i-1] + db * dt)

    # --- 6. Noisy observations ---
    obs_noise = rng.uniform(0.04, 0.10)
    lambda_obs = np.clip(lambda_true + rng.normal(0, obs_noise, size=T), 0, None)
    r_noise = rng.normal(0, obs_noise, size=T)
    r_obs_raw = r_true + r_noise
    kernel = np.ones(3) / 3.0
    r_obs = np.clip(np.convolve(r_obs_raw, kernel, mode='same'), 0, None)

    return dict(
        t=t, lambda_obs=lambda_obs, r_obs=r_obs, b=b, c_true=c,
        ts=ts, te=te, theta_down=theta_down, theta_up=theta_up,
        duration=te - ts if has_disruption else 0.0, has_disruption=has_disruption,
    )


def make_realistic_dataset(n_episodes=400, T=900, seed=0, frac_no_disruption=0.0):
    """Generate a dataset of realistic episodes."""
    rng = np.random.default_rng(seed)
    episodes = []
    n_no = int(n_episodes * frac_no_disruption)
    for i in range(n_episodes):
        has_dis = i >= n_no
        episodes.append(simulate_realistic_episode(T=T, rng=rng, has_disruption=has_dis))
    return episodes


if __name__ == "__main__":
    eps = make_realistic_dataset(n_episodes=5, T=900, seed=42)
    for e in eps:
        print(f"ts={e['ts']:.1f} te={e['te']:.1f} dur={e['duration']:.1f} "
              f"theta_down={e['theta_down']:.2f} theta_up={e['theta_up']:.2f} "
              f"max_b={e['b'].max():.2f} noise_b_std={e['b'].std():.2f}")
