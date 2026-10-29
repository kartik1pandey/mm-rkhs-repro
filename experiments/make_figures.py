"""Turn results/*.json into figures/*.png.   python experiments/make_figures.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"

# Categorical slots in fixed order (validated reference palette), plus neutrals.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}")


def fig1_reproduction():
    r = load("fig1")
    c, k = r["curves"], np.arange(len(r["curves"]["MM-RKHS"]))
    styles = [("Sample MM-RKHS", AQUA, "o"), ("Sample PPO", YELLOW, "o"),
              ("MM-RKHS", BLUE, None), ("PPO", ORANGE, None)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, log in zip(axes, (False, True)):
        for name, color, marker in styles:
            ax.plot(k, c[name], color=color, marker=marker, markersize=3, label=name)
        ax.axhline(r["J_star"], color=INK2, ls="--", lw=1, label="Optimal J*")
        ax.set_xlabel("Iteration k")
        ax.set_ylabel(r"Expected discounted cost $\langle v_{\pi_k}, \rho\rangle$")
        if log:
            ax.set_yscale("log")
        ax.set_title("Convergence: MM-RKHS vs PPO" + (" (log scale)" if log else ""))
    axes[0].legend(loc="center right")
    fig.suptitle("Reproduction of Fig. 1, arXiv:2603.17875 - GARNET(1000 states, 200 actions, "
                 "branching 20), gamma = 0.95", fontsize=10, color=INK2)
    save(fig, "fig1_reproduction.png")


def fig_key_finding():
    """The figure to attach to the email: (a) beta-invariance, (b) where the speed comes from."""
    b, s = load("beta"), load("seeds")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    k = np.arange(len(b["paper_beta"]))
    gap = lambda J: np.maximum(np.array(J) - b["J_star"], 1e-12)  # noqa: E731
    ax.plot(k, gap(b["paper_beta"]), color=BLUE, lw=5, alpha=0.35, label=r"paper's $\beta_k$")
    ax.plot(k, gap(b["beta_zero"]), color=ORANGE, lw=2, label=r"$\beta_k = 0$")
    ax.plot(k, gap(b["beta_x100"]), color=INK, lw=1, ls=":", label=r"$100\times\beta_k$")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"Suboptimality $J(\pi_k) - J^*$")
    ax.set_title(r"(a) With $\bar l = 1$, $\beta_k$ has no effect")
    ax.legend(loc="upper right")
    ax.text(0.03, 0.04, f"max |ΔJ| (β vs 0) = {b['max_abs_J_diff_zero']:.0e}",
            transform=ax.transAxes, color=INK2, fontsize=9)

    ax = axes[1]
    styles = {"Policy iteration": GREEN, "MM-RKHS (paper)": BLUE,
              "NPG, constant eta=10": VIOLET, "NPG, constant eta=1.15": AQUA, "PPO": ORANGE}
    for name, color in styles.items():
        g = np.maximum(np.array(s["gaps"][name]), 1e-12)
        med = np.median(g, axis=0)
        kk = np.arange(len(med))
        ax.plot(kk, med, color=color, label=name.replace("eta", "η"))
        ax.fill_between(kk, g.min(axis=0), g.max(axis=0), color=color, alpha=0.12, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-9)
    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"Suboptimality $J(\pi_k) - J^*$")
    ax.set_title(f"(b) Step-size schedule drives the speed-up ({len(s['seeds'])} seeds)")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "fig2_key_finding.png")


def fig_inner():
    r = load("inner")
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (name, J), color in zip(r["curves"].items(), colors):
        ax.plot(np.maximum(np.array(J) - r["J_star"], 1e-12), color=color, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"Suboptimality $J(\pi_k) - J^*$")
    ax.set_title(r"Activating the RKHS term: $\bar l > 1$ and non-identity $R$")
    ax.legend(fontsize=8.5)
    save(fig, "fig3_inner_steps.png")


if __name__ == "__main__":
    for f in (fig1_reproduction, fig_key_finding, fig_inner):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skip {f.__name__}: {e.filename} missing")
