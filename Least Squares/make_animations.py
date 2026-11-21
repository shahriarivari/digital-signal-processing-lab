"""
Generates every animation / figure used in the README.
Run:  python make_animations.py      (needs numpy, matplotlib, pillow)
Outputs go to ./assets/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

os.makedirs("assets", exist_ok=True)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY = "#2563eb", "#dc2626", "#16a34a", "#6b7280"

# ---------------------------------------------------------------- data
rng = np.random.default_rng(42)
N = 20
x = np.sort(rng.uniform(0, 10, N))
y = 0.8 * x + 2.0 + rng.normal(0, 0.8, N)

def cost(w, b):
    return np.mean((y - (w * x + b)) ** 2)

w_ls = np.cov(x, y, bias=True)[0, 1] / np.var(x)
b_ls = y.mean() - w_ls * x.mean()
J_ls = cost(w_ls, b_ls)

wg = np.linspace(-1, 3, 201)
bg = np.linspace(-4, 8, 201)
W, B = np.meshgrid(wg, bg)
Jgrid = np.mean((y - (W[..., None] * x + B[..., None])) ** 2, axis=-1)
levels = np.geomspace(J_ls * 1.05, Jgrid.max(), 25)


def draw_fit(ax, w, b, color=RED, squares=True, title=None, extra_line=None):
    ax.clear()
    ax.grid(True, alpha=0.3)
    xx = np.array([-1, 11])
    if extra_line is not None:                       # best-so-far line
        ax.plot(xx, extra_line[0] * xx + extra_line[1], color=GREEN, lw=2, alpha=0.9)
    ax.plot(xx, w * xx + b, color=color, lw=2.2)
    e = y - (w * x + b)
    for xi, yi, ei in zip(x, y, e):
        yh = yi - ei
        ax.plot([xi, xi], [yh, yi], color=GREY, lw=1)
        if squares:
            ax.add_patch(Rectangle((xi, min(yi, yh)), abs(ei), abs(ei),
                                   color=color, alpha=0.16, lw=0))
    ax.scatter(x, y, color=BLUE, zorder=3, s=28)
    ax.set_xlim(-1, 12); ax.set_ylim(-2, 13); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title or f"w = {w:.2f}   b = {b:.2f}   J = {cost(w, b):.2f}")


# ================================================================
# 1) HERO: squares shrinking while gradient descent fits the line
# ================================================================
def gradient_descent(w, b, eta=0.02, iters=1500):
    hist = [(w, b, cost(w, b))]
    for _ in range(iters):
        e = y - (w * x + b)
        w -= eta * (-2 * np.mean(e * x))
        b -= eta * (-2 * np.mean(e))
        hist.append((w, b, cost(w, b)))
    return np.array(hist)

hist = gradient_descent(-0.5, 7.5)
frames = np.unique(np.concatenate([np.arange(0, 25),
                                   np.geomspace(25, len(hist) - 1, 35).astype(int)]))
frames = np.concatenate([frames, [frames[-1]] * 12])          # hold the last frame

fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=90)
def upd(k):
    i = frames[k]; w, b, J = hist[i]
    draw_fit(ax, w, b, title=f"iteration {i:4d}   |   total area of squares J = {J:.2f}")
    fig.tight_layout()
FuncAnimation(fig, upd, frames=len(frames)).save("assets/hero_squares.gif", writer=PillowWriter(fps=10))
plt.close(fig)
print("hero_squares.gif done")

# ================================================================
# shared helpers for the "two worlds" figures
# ================================================================
from matplotlib.colors import LogNorm

def eq(w, b):
    sign = "+" if b >= 0 else "−"
    return f"y = {w:.2f}·x {sign} {abs(b):.2f}"

def param_space(ax, title="PARAMETER SPACE: every point is one line", annotate=True):
    cf = ax.contourf(W, B, Jgrid, levels=np.geomspace(J_ls, Jgrid.max(), 30),
                     norm=LogNorm(), cmap="viridis", alpha=0.85)
    ax.plot(w_ls, b_ls, "*", color="white", ms=16, mec="k", zorder=5)
    if annotate:
        ax.annotate("lowest error = best line", (w_ls, b_ls), xytext=(1.0, -3.5), fontsize=9,
                    color="white", weight="bold", arrowprops=dict(arrowstyle="->", color="white"))
    ax.set_xlabel("slope w   (how steep the line is)")
    ax.set_ylabel("intercept b   (where it crosses x = 0)")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(-1, 3); ax.set_ylim(-4, 8); ax.grid(False)
    return cf

def data_space(ax, title="DATA SPACE: the line on the data"):
    ax.clear(); ax.grid(True, alpha=0.3)
    ax.scatter(x, y, color=BLUE, s=22, zorder=3)
    ax.set_xlim(-1, 11); ax.set_ylim(-2, 13)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title, fontsize=10.5)

def draw_line(ax, w, b, color, lw=2.2, residuals=False, label=None, ls="-"):
    xx = np.array([-1, 11])
    ax.plot(xx, w * xx + b, color=color, lw=lw, ls=ls, label=label)
    if residuals:
        for xi, yi in zip(x, y):
            ax.plot([xi, xi], [w * xi + b, yi], color=color, lw=0.8, alpha=0.5)

# ================================================================
# 2a) STATIC: anatomy of a line  y = w x + b
# ================================================================
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), dpi=90)
xx = np.linspace(-1, 10, 2)
ax = axs[0]
ax.plot(xx, 0.8 * xx + 2, color=RED, lw=2.5)
ax.plot(0, 2, "o", color="k", ms=7, zorder=4)
ax.annotate("b = 2\nthe height where the line\ncrosses the y-axis (x = 0)", (0, 2), xytext=(0.6, 7.6),
            fontsize=9, arrowprops=dict(arrowstyle="->"))
ax.plot([4, 7], [5.2, 5.2], "k--", lw=1.2); ax.plot([7, 7], [5.2, 7.6], "k--", lw=1.2)
ax.text(5.5, 4.5, "run = 3", ha="center", fontsize=9)
ax.text(7.2, 6.3, "rise = 2.4", fontsize=9)
ax.text(3.3, 0.2, "w = rise / run = 0.8\n“+1 step in x → +0.8 in y”", fontsize=9)
ax.set_title("y = w·x + b   →   y = 0.8·x + 2", fontsize=10.5)
for a in axs:
    a.set_xlim(-1, 10); a.set_ylim(-1, 11); a.axvline(0, color="k", lw=0.8); a.axhline(0, color="k", lw=0.8)
for wv, c in zip([0.2, 0.8, 1.4], ["#fca5a5", RED, "#7f1d1d"]):
    axs[1].plot(xx, wv * xx + 2, color=c, lw=2, label=f"w = {wv}")
axs[1].plot(0, 2, "ko"); axs[1].legend(fontsize=9, loc="upper left")
axs[1].set_title("change w → the line TILTS\n(pivots around the intercept)", fontsize=10.5)
for bv, c in zip([0, 2, 4], ["#93c5fd", BLUE, "#1e3a8a"]):
    axs[2].plot(xx, 0.8 * xx + bv, color=c, lw=2, label=f"b = {bv}")
axs[2].legend(fontsize=9, loc="upper left")
axs[2].set_title("change b → the line SLIDES up/down\n(same steepness)", fontsize=10.5)
plt.tight_layout(); plt.savefig("assets/line_anatomy.png"); plt.close()
print("line_anatomy.png done")

# ================================================================
# 2b) STATIC: how to read the right-hand plot (3 lines = 3 points)
# ================================================================
examples = [("A", 0.2, 6.0, "#f97316"), ("B", 1.8, -1.5, "#a21caf"), ("C", w_ls, b_ls, GREEN)]
fig, (aL, aR) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=90)
data_space(aL, "DATA SPACE: three candidate lines")
param_space(aR, "PARAMETER SPACE: the same three lines as points", annotate=False)
for name, w, b, c in examples:
    draw_line(aL, w, b, c, label=f"{name}: {eq(w, b)}   error J = {cost(w, b):.2f}")
    aR.plot(w, b, "o", color=c, ms=11, mec="white", mew=1.5, zorder=6)
    aR.text(w + 0.08, b + 0.35, name, color="white", fontsize=12, weight="bold", zorder=6)
aL.legend(fontsize=8.5, loc="upper left")
plt.tight_layout(); plt.savefig("assets/how_to_read.png"); plt.close()
print("how_to_read.png done")

# ================================================================
# 2c) THE FOR LOOP: brute force grid search, best-so-far tracking
# ================================================================
w_try = np.linspace(-0.6, 2.2, 9)
b_try = np.linspace(-2, 7, 8)
trials = [(w, b) for w in w_try for b in b_try]

fig, (aL, aR) = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=72)
param_space(aR)
tried_pts, = aR.plot([], [], "o", color="white", ms=3, alpha=0.7)
cur_pt, = aR.plot([], [], "o", color=RED, ms=10, mec="white", mew=1.5, zorder=7)
best_pt, = aR.plot([], [], "o", color=GREEN, ms=10, mec="white", mew=1.5, zorder=6)
cur_lbl = aR.text(0, 0, "", color="white", fontsize=8.5, zorder=8,
                  bbox=dict(boxstyle="round,pad=0.2", fc=RED, ec="none"))

state = {"best": (np.inf, 0, 0)}
def upd2(k):
    k = min(k, len(trials) - 1)
    if k == 0:
        state["best"] = (np.inf, 0, 0)
    w, b = trials[k]; J = cost(w, b)
    if J < state["best"][0]:
        state["best"] = (J, w, b)
    Jb, wb, bb = state["best"]
    data_space(aL, f"try {k+1}/{len(trials)}")
    draw_line(aL, wb, bb, GREEN, lw=2)
    draw_line(aL, w, b, RED, residuals=True)
    aL.text(-0.6, 12.1, f"trying: {eq(w, b)}   J = {J:.2f}", color=RED, fontsize=9, weight="bold")
    aL.text(-0.6, 10.9, f"best:   {eq(wb, bb)}   J = {Jb:.2f}", color=GREEN, fontsize=9, weight="bold")
    tw, tb = zip(*trials[:k + 1])
    tried_pts.set_data(tw, tb); cur_pt.set_data([w], [b]); best_pt.set_data([wb], [bb])
    cur_lbl.set_position((min(w + 0.1, 2.0), b + 0.45)); cur_lbl.set_text(f"w={w:.2f}, b={b:.2f}")
    fig.tight_layout()

FuncAnimation(fig, upd2, frames=len(trials) + 12).save("assets/for_loop_search.gif",
                                                        writer=PillowWriter(fps=5))
plt.close(fig)
print("for_loop_search.gif done")

# ================================================================
# 3) GRADIENT DESCENT: line on data  +  same line as a point on the bowl
# ================================================================
fig, (aL, aR) = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=72)
param_space(aR, "PARAMETER SPACE: rolling downhill")
aR.plot(hist[0, 0], hist[0, 1], "s", color="white", ms=7, zorder=6)
aR.text(hist[0, 0] - 0.15, hist[0, 1] - 0.9, "start", color="white", fontsize=9)
path, = aR.plot([], [], "-", color=RED, lw=1.6, zorder=6)
head, = aR.plot([], [], "o", color=RED, ms=10, mec="white", mew=1.5, zorder=7)
def upd3(k):
    i = frames[k]; w, b, J = hist[i]
    data_space(aL, f"iteration {i}")
    draw_line(aL, w_ls, b_ls, GREEN, lw=1.5, ls="--")
    draw_line(aL, w, b, RED, residuals=True)
    aL.text(-0.6, 12.1, f"{eq(w, b)}   J = {J:.2f}", color=RED, fontsize=9, weight="bold")
    path.set_data(hist[:i + 1, 0], hist[:i + 1, 1]); head.set_data([w], [b])
    fig.tight_layout()
FuncAnimation(fig, upd3, frames=len(frames)).save("assets/gd_bowl.gif", writer=PillowWriter(fps=8))
plt.close(fig)
print("gd_bowl.gif done")

# ================================================================
# 4) LMS: the line updating one sample at a time
# ================================================================
mu, wl, bl = 0.008, -0.5, 7.5
lms = [(wl, bl, -1)]
order = np.concatenate([rng.permutation(N) for _ in range(400)])
for n in order:
    e = y[n] - (wl * x[n] + bl)
    wl += mu * e * x[n]; bl += mu * e
    lms.append((wl, bl, n))
lms_idx = np.unique(np.concatenate([np.arange(0, 40), np.geomspace(40, len(lms) - 1, 45).astype(int)]))

fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=90)
def upd4(k):
    i = lms_idx[min(k, len(lms_idx) - 1)]
    w, b, n = lms[i]
    draw_fit(ax, w, b, squares=False, title=f"LMS update n = {i}:  w = {w:.2f}, b = {b:.2f}")
    ax.plot([-1, 11], [w_ls * -1 + b_ls, w_ls * 11 + b_ls], "--", color=GREEN, lw=1.5)
    if n >= 0:
        ax.scatter([x[n]], [y[n]], s=180, facecolors="none", edgecolors=RED, lw=2, zorder=4)
    fig.tight_layout()
FuncAnimation(fig, upd4, frames=len(lms_idx) + 10).save("assets/lms_sample_by_sample.gif",
                                                         writer=PillowWriter(fps=10))
plt.close(fig)
print("lms_sample_by_sample.gif done")

# ================================================================
# 5) STATIC: non-linear data -> residual plot diagnosis
# ================================================================
x2 = np.sort(rng.uniform(0, 10, 30))
y2 = 0.15 * x2 ** 2 - 0.6 * x2 + 3 + rng.normal(0, 0.5, x2.size)
xx = np.linspace(0, 10, 200)
fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), dpi=90, sharex=True)
for c, deg in enumerate([1, 2]):
    th = np.polyfit(x2, y2, deg); r = y2 - np.polyval(th, x2)
    axs[0, c].scatter(x2, y2, s=20, color=BLUE)
    axs[0, c].plot(xx, np.polyval(th, xx), color=RED, lw=2)
    axs[0, c].set_title(["straight line: y = w x + b", "same least squares, extra column x²"][c])
    axs[1, c].scatter(x2, r, s=20, color="#f59e0b"); axs[1, c].axhline(0, color="k", lw=1)
    axs[1, c].set_title(["residuals: U-shape → wrong model", "residuals: random noise → good"][c])
plt.tight_layout(); plt.savefig("assets/nonlinear_residuals.png"); plt.close()
print("nonlinear_residuals.png done")

# ================================================================
# 6) STATIC: why it's convex - every point adds a U, U + U = U
# ================================================================
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), dpi=90)
ws = np.linspace(-0.4, 2.0, 300)
for xi, yi in zip(x, y):
    axs[0].plot(ws, (yi - ws * xi - b_ls) ** 2 / N, color=GREY, lw=0.8, alpha=0.7)
Jw = [cost(w, b_ls) for w in ws]
axs[0].plot(ws, Jw, color=RED, lw=3, label="total error J")
axs[0].plot(w_ls, J_ls, "*", color=GREEN, ms=16, mec="k")
axs[0].set_ylim(0, 6); axs[0].set_xlabel("slope w  (b kept fixed)"); axs[0].set_ylabel("error")
axs[0].set_title("grey: error of each single point\nred: their sum, still one U", fontsize=10.5)
axs[0].legend(fontsize=9)

bs = np.linspace(-4, 8, 300)
for xi, yi in zip(x, y):
    axs[1].plot(bs, (yi - w_ls * xi - bs) ** 2 / N, color=GREY, lw=0.8, alpha=0.7)
axs[1].plot(bs, [cost(w_ls, b) for b in bs], color=RED, lw=3)
axs[1].plot(b_ls, J_ls, "*", color=GREEN, ms=16, mec="k")
axs[1].set_ylim(0, 6); axs[1].set_xlabel("intercept b  (w kept fixed)")
axs[1].set_title("same story for b:\nsum of U's = one U", fontsize=10.5)

t = np.linspace(-3, 3, 400)
bad = 0.3 * t ** 4 - 1.6 * t ** 2 + 0.5 * t + 2.5
axs[2].plot(t, bad, color="#a21caf", lw=3)
i1 = np.argmin(np.where(t < 0, bad, np.inf)); i2 = np.argmin(np.where(t > 0, bad, np.inf))
axs[2].plot(t[i1], bad[i1], "*", color=GREEN, ms=16, mec="k")
axs[2].plot(t[i2], bad[i2], "o", color=RED, ms=9)
axs[2].annotate("trap: looks like a bottom,\nbut is not the best", (t[i2], bad[i2]), xytext=(0.3, 4.2),
                fontsize=9, arrowprops=dict(arrowstyle="->"))
axs[2].set_title("what least squares does NOT have:\na wavy curve with several dips", fontsize=10.5)
axs[2].set_xlabel("some parameter"); axs[2].set_ylim(0, 6)
plt.tight_layout(); plt.savefig("assets/convexity.png"); plt.close()
print("convexity.png done")

# ================================================================
# 7) STATIC: LMS identifying an unknown FIR system
# ================================================================
h = np.array([0.8, -0.4, 0.2, 0.1])                       # unknown system
M, Nn, runs = len(h), 2000, 100

def lms_run(mu, seed):
    r = np.random.default_rng(seed)
    xs = r.normal(0, 1, Nn)
    d = np.convolve(xs, h)[:Nn] + r.normal(0, 0.05, Nn)    # desired = system output + noise
    wv = np.zeros(M); W_h = np.zeros((Nn, M)); E = np.zeros(Nn)
    for n in range(Nn):
        xv = xs[max(0, n - M + 1):n + 1][::-1]
        xv = np.pad(xv, (0, M - len(xv)))                  # [x[n], x[n-1], ..., x[n-M+1]]
        e = d[n] - wv @ xv                                 # 1) output and error
        wv = wv + mu * e * xv                              # 2) LMS update
        W_h[n] = wv; E[n] = e ** 2
    return W_h, E

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), dpi=90)
W_h, _ = lms_run(0.01, 0)
for k in range(M):
    axs[0].plot(W_h[:, k], lw=1.6, label=f"w{k} → {h[k]}")
    axs[0].axhline(h[k], color="k", ls=":", lw=0.8)
axs[0].set_title("LMS taps learning the unknown system (μ = 0.01)", fontsize=10.5)
axs[0].set_xlabel("sample n"); axs[0].legend(fontsize=8.5, ncol=2)
for mu, c in [(0.002, BLUE), (0.01, GREEN), (0.08, RED)]:
    E = np.mean([lms_run(mu, s)[1] for s in range(runs)], axis=0)
    axs[1].plot(10 * np.log10(E), color=c, lw=1.4, label=f"μ = {mu}")
axs[1].axhline(10 * np.log10(0.05 ** 2), color="k", ls="--", lw=1, label="noise floor")
axs[1].set_title("learning curve (mean of e², 100 runs)", fontsize=10.5)
axs[1].set_xlabel("sample n"); axs[1].set_ylabel("dB"); axs[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("assets/lms_system_id.png"); plt.close()
print("lms_system_id.png done")
