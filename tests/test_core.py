import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmrkhs import Garnet, run  # noqa: E402
from mmrkhs.algorithms import mmrkhs_update, paper_beta, ppo_update  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return Garnet.generate(n_states=30, n_actions=8, branching=4, gamma=0.9, seed=3)


def random_log_policy(env, seed=0):
    x = np.random.default_rng(seed).normal(size=(env.n_states, env.n_actions))
    return x - logsumexp(x, axis=1, keepdims=True)


def test_garnet_is_valid(env):
    assert np.allclose(env.probs.sum(axis=2), 1)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            assert len(set(env.next_states[s, a])) == env.branching


def test_exact_evaluation_satisfies_bellman(env):
    pi = np.exp(random_log_policy(env))
    ev = env.evaluate(pi)
    bellman = np.sum(pi * (env.cost + env.gamma * env.expected_next(ev["v"])), axis=1)
    assert np.allclose(ev["v"], bellman)
    assert np.allclose(np.sum(pi * ev["adv"], axis=1), 0)  # E_pi[A] = 0


def test_single_inner_step_ignores_beta_and_R(env):
    """With l=1 and p_0 = pi_k, the RKHS term in (6.4) is identically zero."""
    log_pi = random_log_policy(env)
    adv = env.evaluate(np.exp(log_pi))["adv"]
    rng = np.random.default_rng(1)
    M = rng.normal(size=(env.n_actions, env.n_actions))
    R = M @ M.T + np.eye(env.n_actions)
    base = mmrkhs_update(log_pi, adv, eta=3.0, beta=0.0)
    for beta in (paper_beta(adv, 0), 1e3):
        for R_ in (None, R):
            assert np.array_equal(base, mmrkhs_update(log_pi, adv, eta=3.0, beta=beta, R=R_))


def test_single_inner_step_is_mirror_descent(env):
    """l=1 MM-RKHS == pi_{k+1} proportional to pi_k exp(-eta A)."""
    log_pi = random_log_policy(env)
    adv = env.evaluate(np.exp(log_pi))["adv"]
    eta = 2.5
    expected = np.exp(log_pi) * np.exp(-eta * adv)
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(np.exp(mmrkhs_update(log_pi, adv, eta, beta=1.0)), expected)


def test_more_inner_steps_do_use_beta(env):
    log_pi = random_log_policy(env)
    adv = env.evaluate(np.exp(log_pi))["adv"]
    a = mmrkhs_update(log_pi, adv, eta=3.0, beta=0.0, inner_steps=5)
    b = mmrkhs_update(log_pi, adv, eta=3.0, beta=paper_beta(adv, 0), inner_steps=5)
    assert not np.allclose(a, b)


def test_ppo_gradient_matches_finite_differences(env):
    log_pi = random_log_policy(env)
    adv = env.evaluate(np.exp(log_pi))["adv"]
    w = np.ones(env.n_states)
    pi_k = np.exp(log_pi)

    def surrogate(theta):
        pi = np.exp(theta - logsumexp(theta, axis=1, keepdims=True))
        r = pi / pi_k
        return np.sum(w[:, None] * pi_k * np.maximum(r * adv, np.clip(r, 0.8, 1.2) * adv))

    lr = 1e-6
    theta1 = ppo_update(log_pi, adv, w, lr=lr, n_steps=1)
    grad = (log_pi - theta1) / lr
    s, a, h = 4, 2, 1e-6
    e = np.zeros_like(log_pi)
    e[s, a] = h
    fd = (surrogate(log_pi + e) - surrogate(log_pi - e)) / (2 * h)
    assert grad[s, a] == pytest.approx(fd, rel=1e-4, abs=1e-8)


def test_policy_iteration_reaches_optimum(env):
    j_star = env.rho @ env.optimal_value()
    J = run(env, method="pi", K=20)["J"]
    assert J[-1] == pytest.approx(j_star, abs=1e-8)
    assert np.all(np.diff(J) <= 1e-10)  # monotone improvement
