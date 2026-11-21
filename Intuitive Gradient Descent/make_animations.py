"""
Generates the GIF animations used in README.md.

    python make_animations.py

Needs: numpy, matplotlib, pillow.  Output goes to ./assets/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#6b7280"


# ---------------------------------------------------------------------------
# 1) One parameter, one step at a time, with the actual numbers on screen
# ---------------------------------------------------------------------------
def anim_1d_steps():
    J  = lambda w: (w - 2.0) ** 2 + 1.0        # cost function
    dJ = lambda w: 2.0 * (w - 2.0)             # its slope
    mu, w, n_iter = 0.15, -2.0, 12

    hist = [w]
    for _ in range(n_iter):
        w = w - mu * dJ(w)
        hist.append(w)
    hist = np.array(hist)

    # two "phases" per iteration: 0 = measure slope, 1 = take step
    frames = [(k, ph) for k in range(n_iter) for ph in (0, 1)] + [(n_iter, 0)] * 6

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.3),
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    ww = np.linspace(-3, 6, 400)

    def draw(f):
        k, ph = f
        ax.clear(); ax2.clear()
        ax.plot(ww, J(ww), color=GREY, lw=2)
        ax.set_xlim(-3, 6); ax.set_ylim(0, 27)
        ax.set_xlabel("parameter  w"); ax.set_ylabel("cost  J(w)")
        ax.set_title("J(w) = (w − 2)² + 1     μ = 0.15")

        # breadcrumb trail of past points
        ax.plot(hist[:k + 1], J(hist[:k + 1]), "o", color=BLUE, alpha=0.25, ms=6)

        wk = hist[k]
        g = dJ(wk)
        ax.plot(wk, J(wk), "o", color=BLUE, ms=11, zorder=5)

        if k < n_iter:
            # tangent line = the local "slope" we just measured
            t = np.linspace(wk - 1.5, wk + 1.5, 2)
            ax.plot(t, J(wk) + g * (t - wk), color=ORANGE, lw=2.2)
            step = -mu * g
            if ph == 1:
                ax.annotate("", xy=(wk + step, 0.6), xytext=(wk, 0.6),
                            arrowprops=dict(arrowstyle="->", color=RED, lw=2.5))
                ax.plot([wk + step] * 2, [0.6, J(wk + step)], ":", color=RED)
                ax.plot(wk + step, J(wk + step), "o", mfc="none", mec=RED, ms=11, mew=2)
            lines = [
                f"iteration k = {k}",
                f"w_k        = {wk: .4f}",
                f"J(w_k)     = {J(wk): .4f}",
                f"slope J'   = {g: .4f}",
            ]
            if ph == 1:
                lines += ["", "w_k+1 = w_k − μ·J'",
                          f"      = {wk:.3f} − 0.15·({g:.3f})",
                          f"      = {wk + step:.4f}"]
            else:
                lines += ["", "slope < 0  →  move right" if g < 0 else
                          "slope > 0  →  move left"]
        else:
            lines = [f"after {n_iter} iterations", f"w = {wk:.4f}",
                     f"J = {J(wk):.4f}", "", "true minimum: w* = 2, J = 1"]
        ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="top", family="monospace", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="white", ec="#d1d5db"))

        # right panel: cost vs iteration
        ax2.plot(range(k + 1), J(hist[:k + 1]), "o-", color=BLUE)
        ax2.axhline(1.0, color=GREEN, ls="--", lw=1.2)
        ax2.text(n_iter, 1.4, "J_min = 1", color=GREEN, ha="right", fontsize=9)
        ax2.set_xlim(-0.5, n_iter + 0.5); ax2.set_ylim(0, 18)
        ax2.set_xlabel("iteration k"); ax2.set_title("cost going down")
        fig.tight_layout()

    a = FuncAnimation(fig, draw, frames=frames, interval=700)
    a.save(os.path.join(OUT, "gd_1d_steps.gif"), writer=PillowWriter(fps=1.4), dpi=85)
    plt.close(fig)
    return hist


# ---------------------------------------------------------------------------
# 2) Same function, four step sizes
# ---------------------------------------------------------------------------
def anim_learning_rate():
    J  = lambda w: (w - 2.0) ** 2 + 1.0
    dJ = lambda w: 2.0 * (w - 2.0)
    mus = [(0.05, "too small: crawls"), (0.30, "good"),
           (0.90, "too big: zig-zags"), (1.05, "μ > 1: diverges")]
    n_iter = 15
    paths = []
    for mu, _ in mus:
        w, p = -1.0, [-1.0]
        for _ in range(n_iter):
            w = w - mu * dJ(w); p.append(w)
        paths.append(np.array(p))

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, sharey=True)
    ww = np.linspace(-3, 7, 400)

    def draw(k):
        for ax, (mu, lab), p in zip(axes.flat, mus, paths):
            ax.clear()
            ax.plot(ww, J(ww), color=GREY, lw=1.8)
            seg = p[:k + 1]
            ax.plot(seg, J(seg), "-", color=BLUE, alpha=0.35, lw=1)
            ax.plot(seg, J(seg), "o", color=BLUE, ms=4, alpha=0.4)
            ax.plot(seg[-1], J(seg[-1]), "o", color=RED if mu > 1 else BLUE, ms=9)
            ax.set_xlim(-3, 7); ax.set_ylim(0, 26)
            ax.set_title(f"μ = {mu}   ({lab})", fontsize=10.5)
        fig.suptitle(f"Step size matters — iteration {k}", fontsize=12)
        fig.tight_layout()

    frames = list(range(n_iter + 1)) + [n_iter] * 5
    a = FuncAnimation(fig, draw, frames=frames, interval=500)
    a.save(os.path.join(OUT, "gd_learning_rate.gif"), writer=PillowWriter(fps=2), dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) Non-convex cost: where you start decides where you end
# ---------------------------------------------------------------------------
def anim_nonconvex():
    J  = lambda w: 0.1 * w**4 - w**2 + 0.5 * w + 3.0
    dJ = lambda w: 0.4 * w**3 - 2.0 * w + 0.5
    mu, n_iter = 0.08, 30
    starts = [(-0.6, GREEN), (0.6, ORANGE)]
    paths = []
    for w0, _ in starts:
        w, p = w0, [w0]
        for _ in range(n_iter):
            w = w - mu * dJ(w); p.append(w)
        paths.append(np.array(p))

    fig, ax = plt.subplots(figsize=(8, 4.3))
    ww = np.linspace(-3.4, 3.4, 400)

    def draw(k):
        ax.clear()
        ax.plot(ww, J(ww), color=GREY, lw=2)
        for (w0, c), p in zip(starts, paths):
            seg = p[:k + 1]
            ax.plot(seg, J(seg), "o", color=c, ms=4, alpha=0.35)
            ax.plot(seg[-1], J(seg[-1]), "o", color=c, ms=11,
                    label=f"start w₀ = {w0:+.1f}  →  w = {seg[-1]:+.3f},  J = {J(seg[-1]):.3f}")
        ax.set_xlim(-3.4, 3.4); ax.set_ylim(-1, 6)
        ax.set_xlabel("w"); ax.set_ylabel("J(w)")
        ax.set_title(f"Non-convex J(w) = 0.1w⁴ − w² + 0.5w + 3     iteration {k}")
        ax.legend(loc="upper center", fontsize=9, frameon=True)
        ax.text(-2.35, -0.95, "global min", fontsize=9, color=GREEN, ha="center")
        ax.text(2.1, 0.9, "local min", fontsize=9, color=ORANGE, ha="center")
        fig.tight_layout()

    frames = list(range(n_iter + 1)) + [n_iter] * 6
    a = FuncAnimation(fig, draw, frames=frames, interval=250)
    a.save(os.path.join(OUT, "gd_nonconvex.gif"), writer=PillowWriter(fps=5), dpi=85)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Two parameters: the MSE surface of a 2-tap Wiener filter
# ---------------------------------------------------------------------------
R = np.array([[1.0, 0.8], [0.8, 1.0]])   # input autocorrelation matrix
p = np.array([0.8, 0.2])                  # cross-correlation vector
sigma_d2 = 2.0                            # power of desired signal

def J2(w):   # w shape (..., 2)
    return sigma_d2 - 2 * w @ p + np.einsum("...i,ij,...j->...", w, R, w)

def grad2(w):
    return 2 * (R @ w - p)

w_opt = np.linalg.solve(R, p)             # closed-form Wiener solution

def anim_2d():
    mu, n_iter = 0.25, 40
    w = np.array([-2.5, -2.2]); path = [w.copy()]
    for _ in range(n_iter):
        w = w - mu * grad2(w); path.append(w.copy())
    path = np.array(path)

    g1, g2 = np.meshgrid(np.linspace(-3, 3.5, 200), np.linspace(-2.8, 2.5, 200))
    Z = J2(np.stack([g1, g2], -1))
    fig, ax = plt.subplots(figsize=(6.6, 5.6))

    def draw(k):
        ax.clear()
        ax.contour(g1, g2, Z, levels=Z.min() + np.geomspace(0.02, Z.max() - Z.min(), 18),
                   colors="#9ca3af", linewidths=1.0)
        ax.plot(*w_opt, "*", color=GREEN, ms=18, zorder=6,
                label=f"closed form  w° = R⁻¹p = ({w_opt[0]:.3f}, {w_opt[1]:.3f})")
        seg = path[:k + 1]
        ax.plot(seg[:, 0], seg[:, 1], "o-", color=BLUE, ms=4, lw=1.2, alpha=0.7)
        ax.plot(*seg[-1], "o", color=BLUE, ms=10, zorder=7,
                label=f"gradient descent, k={k}: ({seg[-1,0]:.3f}, {seg[-1,1]:.3f})")
        if k < n_iter:
            g = grad2(seg[-1])
            ax.annotate("", xy=seg[-1] - mu * g, xytext=seg[-1],
                        arrowprops=dict(arrowstyle="->", color=RED, lw=2))
        ax.set_xlabel("w₁ (tap 1)"); ax.set_ylabel("w₂ (tap 2)")
        ax.set_title("MSE surface of a 2-tap filter (contours)\n"
                     "each step is perpendicular to the contour it is on", fontsize=10.5)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.set_aspect("equal")
        fig.tight_layout()

    frames = list(range(n_iter + 1)) + [n_iter] * 6
    a = FuncAnimation(fig, draw, frames=frames, interval=350)
    a.save(os.path.join(OUT, "gd_2d_wiener.gif"), writer=PillowWriter(fps=3), dpi=85)
    plt.close(fig)
    return path


if __name__ == "__main__":
    # anim_1d_steps()  # older 1-D version, replaced by make_3d_zoom.py
    anim_learning_rate();    print("learning-rate done")
    anim_nonconvex();        print("non-convex done")
    pth = anim_2d();         print("2d done, final:", pth[-1], "w_opt:", w_opt)
