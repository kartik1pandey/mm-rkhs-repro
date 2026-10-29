"""Run one algorithm for K iterations and record J(pi_k) = <v_{pi_k}, rho> each step."""

import numpy as np

from .algorithms import mmrkhs_update, paper_beta, paper_eta, ppo_update, policy_iteration_update
from .sampling import mc_advantage


def run(env, method, K=50, sample_based=False, seed=0, **cfg):
    """Return a dict with the cost curve J_0..J_K (length K + 1) and the final policy.

    method:
      "mmrkhs" - eqs. (6.4)-(6.5). cfg: inner_steps (1), beta_scale (1.0),
                 eta0 (1.15), eta_schedule ("linear" | "constant"), R (None = I)
      "ppo"    - eqs. (6.1). cfg: eps (0.2), lr (0.8), n_steps (10),
                 weighting ("visitation" | "uniform")
      "pi"     - exact policy iteration (reference)
    Sample-based runs replace the exact advantage with a Monte-Carlo estimate;
    J is always evaluated exactly so curves are comparable.
    """
    rng = np.random.default_rng(seed)
    n, m = env.n_states, env.n_actions
    log_pi = np.full((n, m), -np.log(m))  # uniform initial policy
    pi = np.exp(log_pi)
    J = []
    for k in range(K + 1):
        ev = env.evaluate(pi)
        J.append(ev["J"])
        if k == K:
            break
        if sample_based:
            adv, visitation = mc_advantage(env, pi, rng)
        else:
            adv, visitation = ev["adv"], None

        if method == "mmrkhs":
            if cfg.get("eta_schedule", "linear") == "linear":
                eta = paper_eta(k, cfg.get("eta0", 1.15))
            else:
                eta = cfg.get("eta0", 1.15)
            beta = cfg.get("beta_scale", 1.0) * paper_beta(adv, k)
            log_pi = mmrkhs_update(log_pi, adv, eta, beta, R=cfg.get("R"),
                                   inner_steps=cfg.get("inner_steps", 1),
                                   clip=1.5 if sample_based else None)
            pi = np.exp(log_pi)
        elif method == "ppo":
            if cfg.get("weighting", "visitation") == "visitation":
                w = visitation if sample_based else env.visitation(ev["P"])
            else:
                w = np.ones(n)
            log_pi = ppo_update(log_pi, adv, w, eps=cfg.get("eps", 0.2),
                                lr=cfg.get("lr", 0.8), n_steps=cfg.get("n_steps", 10))
            log_pi -= np.log(np.sum(np.exp(log_pi), axis=1, keepdims=True))
            pi = np.exp(log_pi)
        elif method == "pi":
            pi = policy_iteration_update(adv)
        else:
            raise ValueError(method)
    return {"J": np.array(J), "pi": pi}
