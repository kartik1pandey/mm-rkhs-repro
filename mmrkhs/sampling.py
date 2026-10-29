"""Monte-Carlo advantage estimates for the sample-based variants (Sec. 6.5).

The paper uses 5 episodes of 50,000 samples per iteration and estimates the
advantage following Sutton & Barto, Sec. 5.2 (every-visit Monte Carlo). Returns
are truncated at the end of each episode; the last `burn` steps of every episode
are excluded from the averages because their returns are cut short
(0.95^300 ~ 2e-7, so the remaining bias is negligible).
State-action pairs that are never visited get advantage 0 (no update there).
"""

import numpy as np
from scipy.signal import lfilter


def rollout(env, pi, n_episodes, length, rng):
    """Simulate n_episodes parallel chains under pi; returns (states, actions) (E, T)."""
    cum_pi = np.cumsum(pi, axis=1)
    cum_p = np.cumsum(env.probs, axis=2)
    s = rng.choice(env.n_states, size=n_episodes, p=env.rho)
    states = np.empty((n_episodes, length), dtype=np.int64)
    actions = np.empty((n_episodes, length), dtype=np.int64)
    u_a = rng.random((length, n_episodes))
    u_s = rng.random((length, n_episodes))
    last_a, last_b = env.n_actions - 1, env.branching - 1
    for t in range(length):
        a = np.minimum((cum_pi[s] < u_a[t, :, None]).sum(axis=1), last_a)
        j = np.minimum((cum_p[s, a] < u_s[t, :, None]).sum(axis=1), last_b)
        states[:, t], actions[:, t] = s, a
        s = env.next_states[s, a, j]
    return states, actions


def mc_advantage(env, pi, rng, n_episodes=5, length=50_000, burn=300):
    """Every-visit Monte-Carlo estimate of A_pi; also returns empirical visitation."""
    states, actions = rollout(env, pi, n_episodes, length, rng)
    costs = env.cost[states, actions]
    # Discounted reverse cumulative sum: G_t = c_t + gamma * G_{t+1}.
    returns = lfilter([1.0], [1.0, -env.gamma], costs[:, ::-1], axis=1)[:, ::-1]
    keep = slice(0, length - burn)
    s, a, g = states[:, keep].ravel(), actions[:, keep].ravel(), returns[:, keep].ravel()
    n, m = env.n_states, env.n_actions
    sa = s * m + a
    sa_count = np.bincount(sa, minlength=n * m).reshape(n, m)
    sa_sum = np.bincount(sa, weights=g, minlength=n * m).reshape(n, m)
    s_count = np.bincount(s, minlength=n)
    s_sum = np.bincount(s, weights=g, minlength=n)
    v_hat = np.divide(s_sum, s_count, out=np.zeros(n), where=s_count > 0)
    q_hat = np.divide(sa_sum, sa_count, out=np.zeros((n, m)), where=sa_count > 0)
    adv = np.where(sa_count > 0, q_hat - v_hat[:, None], 0.0)
    visitation = s_count / s_count.sum()
    return adv, visitation
