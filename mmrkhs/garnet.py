"""GARNET MDPs (Bhatnagar et al., 2009) with sparse transitions and exact evaluation.

A GARNET(n, m, b) instance has n states and m actions; every (s, a) pair can reach
exactly b next states, with probabilities given by a random partition of [0, 1].
Costs are drawn i.i.d. from U[0, 1]. Transitions are stored sparsely as
(next-state indices, probabilities), each of shape (n, m, b), so a 1000 x 200 x 20
instance needs ~32 MB instead of the 1.6 GB a dense kernel would.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Garnet:
    n_states: int
    n_actions: int
    branching: int
    gamma: float
    next_states: np.ndarray  # (n, m, b) int
    probs: np.ndarray        # (n, m, b) float, rows sum to 1
    cost: np.ndarray         # (n, m) float
    rho: np.ndarray          # (n,) initial distribution

    # ------------------------------------------------------------------ build
    @classmethod
    def generate(cls, n_states=1000, n_actions=200, branching=20, gamma=0.95, seed=0):
        rng = np.random.default_rng(seed)
        n, m, b = n_states, n_actions, branching
        # b distinct next states per (s, a): argsort of random keys, take first b.
        next_states = np.empty((n, m, b), dtype=np.int64)
        for s in range(n):
            next_states[s] = np.argpartition(rng.random((m, n)), b, axis=1)[:, :b]
        # Random partition of [0, 1] into b pieces (the classic GARNET recipe).
        cuts = np.sort(rng.random((n, m, b - 1)), axis=2)
        edges = np.concatenate([np.zeros((n, m, 1)), cuts, np.ones((n, m, 1))], axis=2)
        probs = np.diff(edges, axis=2)
        cost = rng.random((n, m))
        rho = np.full(n, 1.0 / n)
        return cls(n, m, b, gamma, next_states, probs, cost, rho)

    # ------------------------------------------------------------ evaluation
    def transition_matrix(self, pi):
        """Dense P_pi[s, s'] = sum_a pi(a|s) P(s'|s, a), built with one bincount."""
        n = self.n_states
        rows = np.broadcast_to(np.arange(n)[:, None, None], self.next_states.shape)
        flat = (rows * n + self.next_states).ravel()
        weights = (pi[:, :, None] * self.probs).ravel()
        return np.bincount(flat, weights=weights, minlength=n * n).reshape(n, n)

    def expected_next(self, v):
        """E[v(s') | s, a] for every (s, a); shape (n, m)."""
        return np.einsum("smb,smb->sm", self.probs, v[self.next_states])

    def evaluate(self, pi):
        """Exact v_pi, q_pi, advantage A_pi = q_pi - v_pi, and J(pi) = <v_pi, rho>."""
        P = self.transition_matrix(pi)
        c_pi = np.sum(pi * self.cost, axis=1)
        v = np.linalg.solve(np.eye(self.n_states) - self.gamma * P, c_pi)
        q = self.cost + self.gamma * self.expected_next(v)
        return {"v": v, "q": q, "adv": q - v[:, None], "J": float(self.rho @ v), "P": P}

    def visitation(self, P):
        """Normalised discounted state-visitation d_pi = (1-g) rho^T (I - g P_pi)^-1."""
        d = np.linalg.solve((np.eye(self.n_states) - self.gamma * P).T, self.rho)
        return (1 - self.gamma) * d

    def optimal_value(self, tol=1e-12, max_iter=100_000):
        """v* via value iteration (cost minimisation)."""
        v = np.zeros(self.n_states)
        for _ in range(max_iter):
            v_new = np.min(self.cost + self.gamma * self.expected_next(v), axis=1)
            if np.max(np.abs(v_new - v)) < tol:
                return v_new
            v = v_new
        return v

    def uniform_policy(self):
        return np.full((self.n_states, self.n_actions), 1.0 / self.n_actions)
