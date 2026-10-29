"""Run every experiment and save raw curves to results/.

    python experiments/run_all.py            # everything (~25 min on a laptop CPU)
    python experiments/run_all.py --only fig1 beta

Experiments
  fig1     Reproduce Fig. 1: MM-RKHS vs PPO, closed-form and sample-based (seed 0)
  beta     l=1 ablation: paper's beta_k vs beta_k = 0 (is the RKHS term active?)
  inner    l in {1, 5, 20}, larger beta, and a non-identity R
  seeds    5 GARNET seeds: MM-RKHS, PPO, NPG with constant step, policy iteration
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmrkhs import Garnet, run  # noqa: E402

RESULTS = ROOT / "results"
K = 50


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(name, payload):
    RESULTS.mkdir(exist_ok=True)
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in payload.items()}
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    log(f"saved results/{name}.json")


def env_and_opt(seed):
    env = Garnet.generate(seed=seed)
    return env, float(env.rho @ env.optimal_value())


def exp_fig1():
    env, j_star = env_and_opt(0)
    curves = {}
    for label, kw in [
        ("MM-RKHS", dict(method="mmrkhs")),
        ("PPO", dict(method="ppo", weighting="uniform")),
        ("PPO (visitation-weighted)", dict(method="ppo", weighting="visitation")),
        ("Sample MM-RKHS", dict(method="mmrkhs", sample_based=True)),
        ("Sample PPO", dict(method="ppo", weighting="uniform", sample_based=True)),
    ]:
        t = time.time()
        curves[label] = run(env, K=K, seed=1, **kw)["J"]
        log(f"fig1 {label}: J_K = {curves[label][-1]:.4f} ({time.time() - t:.0f}s)")
    save("fig1", {"J_star": j_star, "curves": {k: v.tolist() for k, v in curves.items()}})


def exp_beta():
    env, j_star = env_and_opt(0)
    a = run(env, method="mmrkhs", K=K)
    b = run(env, method="mmrkhs", K=K, beta_scale=0.0)
    c = run(env, method="mmrkhs", K=K, beta_scale=100.0)
    save("beta", {
        "J_star": j_star,
        "paper_beta": a["J"], "beta_zero": b["J"], "beta_x100": c["J"],
        "max_abs_J_diff_zero": float(np.max(np.abs(a["J"] - b["J"]))),
        "max_abs_pi_diff_zero": float(np.max(np.abs(a["pi"] - b["pi"]))),
        "max_abs_J_diff_x100": float(np.max(np.abs(a["J"] - c["J"]))),
    })


def action_rbf_kernel(env):
    """A positive-definite R built from action similarity (cost profiles across states)."""
    X = env.cost.T                                   # (m, n): one feature vector per action
    sq = np.sum(X**2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    h2 = np.median(d2[d2 > 0])
    R = np.exp(-d2 / h2)
    return R + 1e-6 * np.eye(env.n_actions)


def exp_inner():
    env, j_star = env_and_opt(0)
    R_rbf = action_rbf_kernel(env)
    curves = {}
    for label, kw in [
        ("l=1 (paper)", dict(inner_steps=1)),
        ("l=5", dict(inner_steps=5)),
        ("l=20", dict(inner_steps=20)),
        ("l=20, 10x beta", dict(inner_steps=20, beta_scale=10.0)),
        ("l=20, RBF R", dict(inner_steps=20, R=R_rbf)),
        ("l=20, RBF R, 10x beta", dict(inner_steps=20, R=R_rbf, beta_scale=10.0)),
    ]:
        t = time.time()
        curves[label] = run(env, method="mmrkhs", K=K, **kw)["J"]
        log(f"inner {label}: J_K = {curves[label][-1]:.5f} ({time.time() - t:.0f}s)")
    save("inner", {"J_star": j_star, "curves": {k: v.tolist() for k, v in curves.items()}})


def exp_seeds(seeds=(0, 1, 2, 3, 4)):
    methods = {
        "MM-RKHS (paper)": dict(method="mmrkhs"),
        "NPG, constant eta=1.15": dict(method="mmrkhs", beta_scale=0.0, eta_schedule="constant", eta0=1.15),
        "NPG, constant eta=10": dict(method="mmrkhs", beta_scale=0.0, eta_schedule="constant", eta0=10.0),
        "PPO": dict(method="ppo", weighting="uniform"),
        "Policy iteration": dict(method="pi"),
    }
    gaps = {name: [] for name in methods}
    for seed in seeds:
        env, j_star = env_and_opt(seed)
        for name, kw in methods.items():
            gaps[name].append((run(env, K=K, **kw)["J"] - j_star).tolist())
        log(f"seeds: seed {seed} done (J* = {j_star:.4f})")
    save("seeds", {"seeds": list(seeds), "gaps": gaps})


EXPERIMENTS = {"fig1": exp_fig1, "beta": exp_beta, "inner": exp_inner, "seeds": exp_seeds}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    args = p.parse_args()
    for name in args.only:
        log(f"=== {name} ===")
        EXPERIMENTS[name]()
