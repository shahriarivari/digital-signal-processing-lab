"""
Shared pieces for the 3-D gradient-descent visuals:
the cost function, the descent path, and the two drawing functions
(main full view + zoomed view) used by both the GIF and the interactive page.
"""
import numpy as np

# ----------------------------------------------------------------- cost function
def F(x, y):
    return 0.25 * x**4 - x**2 + 0.3 * x + 0.25 * y**4 - 0.8 * y**2 + 0.15 * y

def Fx(x, y):   # dF/dx
    return x**3 - 2 * x + 0.3

def Fy(x, y):   # dF/dy
    return y**3 - 1.6 * y + 0.15

F_TEXT = "F(x,y) = 0.25x⁴ − x² + 0.3x + 0.25y⁴ − 0.8y² + 0.15y"
MU = 0.2
START = (-0.3, 1.9)
GLOBAL_MIN = (-1.484, -1.309)
FULL = 2.3          # half-width of the main view
FLOOR = -2.6        # z of the "floor" in the main view

BLUE, ORANGE, RED, GREEN, DARK = "#2563eb", "#ea580c", "#dc2626", "#16a34a", "#111827"

# ----------------------------------------------------------------- run gradient descent
pts = [np.array(START)]
while True:
    x, y = pts[-1]
    step = -MU * np.array([Fx(x, y), Fy(x, y)])
    pts.append(pts[-1] + step)
    if np.linalg.norm(step) < 0.01 or len(pts) > 15:
        break
pts = np.array(pts)
N_IT = len(pts) - 1

# phases inside one iteration
EVAL, SLOPE, NUDGE = 1, 2, 3
FINAL = 99


def zoom_title(k):
    c, h = zoom_window(k)
    return f"ZOOMED VIEW  (×{FULL / h:.1f}, the orange box)"


def zoom_window(k):
    """centre and half-width of the zoomed view for iteration k"""
    k = min(k, N_IT - 1)
    step = np.linalg.norm(pts[k + 1] - pts[k])
    return 0.5 * (pts[k] + pts[k + 1]), max(0.25, 0.65 * step + 0.15)


def _style(ax):
    ax.set_xlabel("x", labelpad=-2); ax.set_ylabel("y", labelpad=-2); ax.set_zlabel("F(x, y)", labelpad=-2)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=34, azim=-58)
    ax.set_box_aspect(None, zoom=0.88)


# ----------------------------------------------------------------- main (full) view
_main_grid = None

def draw_main(ax, k, phase, t=1.0):
    """Static full surface; only dots and lines change.
    k = iteration, phase = EVAL/SLOPE/NUDGE/FINAL, t = progress of the slide during NUDGE."""
    global _main_grid
    if _main_grid is None:
        g = np.linspace(-FULL, FULL, 60)
        X, Y = np.meshgrid(g, g)
        _main_grid = (X, Y, F(X, Y))
    X, Y, Z = _main_grid
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.45, linewidth=0, rcount=60, ccount=60)
    ax.contour(X, Y, Z, levels=14, zdir="z", offset=FLOOR, cmap="viridis", linewidths=0.6, alpha=0.6)
    ax.set_xlim(-FULL, FULL); ax.set_ylim(-FULL, FULL); ax.set_zlim(FLOOR, 6)
    lift = 0.12

    done = pts[: (N_IT if phase == FINAL else k) + 1]
    # path on the surface and its shadow on the floor
    ax.plot(done[:, 0], done[:, 1], F(done[:, 0], done[:, 1]) + lift, "-", color=BLUE, lw=2, zorder=9)
    ax.plot(done[:, 0], done[:, 1], np.full(len(done), FLOOR), "-", color=BLUE, lw=1, alpha=0.6)
    for q in done[:-1]:
        ax.scatter(*q, F(*q) + lift, s=22, color="white", edgecolor=BLUE, depthshade=False, zorder=10)
        ax.scatter(q[0], q[1], FLOOR, s=8, color=BLUE, alpha=0.6, depthshade=False)

    cur = done[-1]
    if phase == NUDGE and k < N_IT:
        p0, p1 = pts[k], pts[k + 1]
        s = np.linspace(0, 1, 25)
        L = p0 + s[:, None] * (p1 - p0)
        ax.plot(L[:, 0], L[:, 1], F(L[:, 0], L[:, 1]) + lift, "--", color=RED, lw=2, zorder=11)
        ax.scatter(*p1, F(*p1) + lift, s=70, facecolor="none", edgecolor=RED, linewidth=2, depthshade=False, zorder=11)
        cur = p0 + (p1 - p0) * t

    ax.scatter(*GLOBAL_MIN, F(*GLOBAL_MIN) + lift, marker="*", s=200, color=GREEN, edgecolor="white", depthshade=False, zorder=12)
    ax.scatter(*cur, F(*cur) + lift, s=90, color=BLUE, edgecolor="white", linewidth=1.5, depthshade=False, zorder=13)
    ax.plot([cur[0]] * 2, [cur[1]] * 2, [FLOOR, F(*cur)], ":", color=BLUE, lw=1)

    # outline of the zoomed region, drawn on the floor
    if True:
        c, h = zoom_window(k)
        bx = [c[0] - h, c[0] + h, c[0] + h, c[0] - h, c[0] - h]
        by = [c[1] - h, c[1] - h, c[1] + h, c[1] + h, c[1] - h]
        ax.plot(bx, by, [FLOOR] * 5, color=ORANGE, lw=1.8)
    _style(ax)


# ----------------------------------------------------------------- zoomed view
def draw_zoom(ax, k, phase, t=1.0):
    ax.clear()
    kk = min(k, N_IT - 1)
    c, h = zoom_window(kk)
    xs = np.linspace(c[0] - h, c[0] + h, 40); ys = np.linspace(c[1] - h, c[1] + h, 40)
    X, Y = np.meshgrid(xs, ys); Z = F(X, Y)
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5, linewidth=0, rcount=40, ccount=40)
    zlo, zhi = Z.min(), Z.max(); pad = 0.08 * (zhi - zlo + 1e-9)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_zlim(zlo - pad, zhi + pad)
    lift = pad * 0.3

    def inside(p):
        return abs(p[0] - c[0]) <= h * 1.001 and abs(p[1] - c[1]) <= h * 1.001

    p0 = pts[kk]; p1 = pts[kk + 1]
    if phase == FINAL:
        p0 = pts[-1]
    if phase in (SLOPE, NUDGE):
        gx, gy = Fx(*p0), Fy(*p0); f0 = F(*p0); r = 0.45 * h
        tX, tY = np.meshgrid(np.linspace(p0[0] - r, p0[0] + r, 2), np.linspace(p0[1] - r, p0[1] + r, 2))
        ax.plot_surface(tX, tY, f0 + gx * (tX - p0[0]) + gy * (tY - p0[1]), color=ORANGE, alpha=0.45, linewidth=0)
    cur = p0
    if phase == NUDGE:
        s = np.linspace(0, 1, 30)
        L = p0 + s[:, None] * (p1 - p0)
        ax.plot(L[:, 0], L[:, 1], F(L[:, 0], L[:, 1]) + lift, "--", color=RED, lw=2, zorder=10)
        ax.scatter(*p1, F(*p1) + lift, s=90, facecolor="none", edgecolor=RED, linewidth=2, depthshade=False, zorder=11)
        cur = p0 + (p1 - p0) * t
    for q in pts[:kk]:
        if inside(q):
            ax.scatter(*q, F(*q) + lift, s=22, color="white", edgecolor=BLUE, depthshade=False, zorder=9)
    ax.scatter(*cur, F(*cur) + lift, s=110, color=BLUE, edgecolor="white", linewidth=1.5, depthshade=False, zorder=12)
    ax.plot([cur[0]] * 2, [cur[1]] * 2, [zlo - pad, F(*cur)], ":", color=BLUE, lw=1)
    _style(ax)


# ----------------------------------------------------------------- the numbers
def f2(v):
    return f"({v:+.3f})"

def text_lines(k, phase):
    """list of (text, colour, bold) lines for the number panel"""
    L = [(F_TEXT, DARK, True), (f"step size μ = {MU}", DARK, False), ("", None, False)]
    if phase == FINAL:
        x, y = pts[-1]; gxm, gym = GLOBAL_MIN
        L += [(f"STOPPED after {N_IT} iterations", GREEN, True),
              (f"point (x, y) = ({x:.3f}, {y:.3f}),  F = {F(x, y):.4f}", GREEN, False),
              (f"slopes  ∂F/∂x = {Fx(x, y):+.4f},  ∂F/∂y = {Fy(x, y):+.4f}", GREEN, False),
              ("slopes ≈ 0  →  steps ≈ 0  →  bottom of a valley", GREEN, False),
              ("", None, False),
              ("Note: this is a LOCAL minimum.", DARK, True),
              (f"Deepest valley (green star): ({gxm}, {gym}), F = {F(gxm, gym):.3f}", DARK, False),
              ("Gradient descent only sees the local slope,", DARK, False),
              ("so the start point decides which valley you reach.", DARK, False)]
        return L
    x, y = pts[k]; f0, gx, gy = F(x, y), Fx(x, y), Fy(x, y)
    L += [(f"ITERATION {k}", BLUE, True),
          (f"current point  (x, y) = ({x:.3f}, {y:.3f})", BLUE, False), ("", None, False),
          ("1) evaluate the function at this point", DARK, True),
          (f"  F = 0.25{f2(x)}⁴ − {f2(x)}² + 0.3{f2(x)}", DARK, False),
          (f"      + 0.25{f2(y)}⁴ − 0.8{f2(y)}² + 0.15{f2(y)}", DARK, False),
          (f"    = {f0:.4f}", DARK, True), ("", None, False)]
    if phase >= SLOPE:
        L += [("2) slopes (partial derivatives) at this point", ORANGE, True),
              ("  ∂F/∂x = x³ − 2x + 0.3", ORANGE, False),
              (f"        = {f2(x)}³ − 2{f2(x)} + 0.3 = {gx:+.4f}", ORANGE, True),
              ("  ∂F/∂y = y³ − 1.6y + 0.15", ORANGE, False),
              (f"        = {f2(y)}³ − 1.6{f2(y)} + 0.15 = {gy:+.4f}", ORANGE, True), ("", None, False)]
    if phase >= NUDGE:
        nx, ny = pts[k + 1]
        L += [("3) nudge AGAINST the slope", RED, True),
              (f"  x_new = x − μ·∂F/∂x = {x:.3f} − {MU}{f2(gx)} = {nx:.3f}", RED, False),
              (f"  y_new = y − μ·∂F/∂y = {y:.3f} − {MU}{f2(gy)} = {ny:.3f}", RED, False),
              (f"  F: {f0:.4f}  →  {F(nx, ny):.4f}   (went down)", RED, True)]
    return L
