"""Policy-update rules compared in the experiments.

All algorithms minimise discounted cost and keep a tabular policy pi (n x m).
Mirror-descent style methods are run in log-space so that large step sizes
(eta_k = 1.15 (k + 1) reaches 57.5 at k = 49) never underflow to exact zeros.

Equation numbers refer to Gupta & Mahajan, "Operator-Theoretic Foundations and
Policy Gradient Methods for General MDPs with Unbounded Costs", arXiv:2603.17875v3.
"""

import numpy as np
from scipy.special import logsumexp


def _normalise_log(log_p):
    return log_p - logsumexp(log_p, axis=1, keepdims=True)


# --------------------------------------------------------------------------- MM-RKHS
def mmrkhs_update(log_pi, adv, eta, beta, R=None, inner_steps=1, clip=None):
    """One outer iteration of MM-RKHS, eqs. (6.4)-(6.5).

    theta_{k,l}(s,a) = -eta_k * (A(s,a) - beta_k(s) * R_a^T (pi_k(s) - p_l))
    p_{l+1,a}        = p_{l,a} exp(theta_{k,l}(s,a)) / Z,   p_0 = pi_k(s)

    R=None means the identity matrix (the paper's setting). `clip` bounds theta,
    as in the paper's sample-based variant (clip = 1.5).
    """
    pi_k = np.exp(log_pi)
    log_p = log_pi.copy()
    beta = np.broadcast_to(np.asarray(beta, dtype=float), (log_pi.shape[0],))
    for _ in range(inner_steps):
        diff = pi_k - np.exp(log_p)                 # zero on the first inner step
        penalty = diff if R is None else diff @ R   # row s holds R_a^T (pi_k(s) - p_l)
        theta = -eta * (adv - beta[:, None] * penalty)
        if clip is not None:
            theta = np.clip(theta, -clip, clip)
        log_p = _normalise_log(log_p + theta)
    return log_p


def paper_beta(adv, k):
    """beta_k(s) = ||A_{pi_k}(s, .)||_inf / sqrt(k + 1)  (Sec. 6.5)."""
    return np.max(np.abs(adv), axis=1) / np.sqrt(k + 1)


def paper_eta(k, eta0=1.15):
    """eta_k = eta_0 (k + 1)  (Sec. 6.5)."""
    return eta0 * (k + 1)


# ------------------------------------------------------------------------------ PPO
def ppo_update(logits, adv, weights, eps=0.2, lr=0.8, n_steps=10):
    """Clipped-surrogate PPO for cost minimisation, eqs. (6.1a)-(6.1c).

    Tabular softmax parameterisation. Each outer iteration takes `n_steps`
    gradient-descent steps of size `lr` on

        L(theta) = sum_s w(s) sum_a pi_k(a|s) max(r A, clip(r, 1-eps, 1+eps) A),
        r = pi_theta(a|s) / pi_k(a|s).

    `weights` is w(s): the discounted visitation d_{pi_k} (the pairing with
    sigma*_pi rho in (6.1a)) or all-ones (unweighted, per-state update).
    """
    pi_k = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    theta = logits.copy()
    for _ in range(n_steps):
        pi = np.exp(theta - logsumexp(theta, axis=1, keepdims=True))
        r = pi / pi_k
        unclipped = r * adv
        clipped = np.clip(r, 1 - eps, 1 + eps) * adv
        g = np.where(unclipped >= clipped, adv, 0.0)          # dL/dr per (s, a)
        # d/dtheta_b sum_a pi_k(a) h(r_a) = pi(b) (g_b - sum_a g_a pi(a))
        grad = pi * (g - np.sum(g * pi, axis=1, keepdims=True))
        theta -= lr * weights[:, None] * grad
    return theta


# ------------------------------------------------------------------- reference methods
def policy_iteration_update(adv):
    """Greedy policy improvement: pi_{k+1}(s) = argmin_a q_k(s, a)."""
    n, m = adv.shape
    pi = np.zeros((n, m))
    pi[np.arange(n), np.argmin(adv, axis=1)] = 1.0
    return pi
