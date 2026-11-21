"""
GIF for the README: main view and zoomed view on top, the numbers below.
    python make_3d_zoom.py   ->  assets/gd_3d_zoom.gif   (needs ffmpeg)
"""
import os
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from gd3d_core import (draw_main, draw_zoom, text_columns, zoom_title, N_IT,
                       EVAL, SLOPE, NUDGE, FINAL, F_TEXT, MU, H)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

schedule = []
for k in range(N_IT):
    hold = 9 if k < 3 else 6
    schedule += [(k, EVAL, 1.0)] * hold
    schedule += [(k, SLOPE, 1.0)] * (hold + 6)
    schedule += [(k, NUDGE, 0.0)] * (hold + 3)
    schedule += [(k, NUDGE, i / 9) for i in range(10)]      # the point slides
schedule += [(N_IT, FINAL, 1.0)] * 36

fig = plt.figure(figsize=(13, 9.2))
axm = fig.add_axes([0.0, 0.33, 0.5, 0.62], projection="3d")
axz = fig.add_axes([0.5, 0.33, 0.5, 0.62], projection="3d")
tx = fig.add_axes([0.03, 0.0, 0.95, 0.33])
fig.text(0.25, 0.965, "MAIN VIEW  (whole surface, stays fixed)", ha="center",
         fontsize=13, fontweight="bold", color="#374151")
ztitle = fig.text(0.75, 0.965, "", ha="center", fontsize=13, fontweight="bold", color="#ea580c")
fig.text(0.75, 0.935, "orange plane = the derivative (slope) at the point", ha="center",
         fontsize=10.5, color="#ea580c")
fig.add_artist(plt.Line2D([0.03, 0.97], [0.335, 0.335], color="#e5e7eb", lw=1))

last = {"k": None}

def draw(frame):
    k, ph, t = frame
    draw_main(axm, k, ph, t)
    key = (k, ph, t if ph == NUDGE else 1.0)     # zoomed view only changes when the step changes
    if key != last["k"]:
        draw_zoom(axz, k, ph, t)
        ztitle.set_text(zoom_title(k))
        last["k"] = key
    tx.clear(); tx.axis("off")
    tx.text(0.0, 0.97, f"{F_TEXT}      μ = {MU},  h = {H}", transform=tx.transAxes, va="top",
            family="DejaVu Sans Mono", fontsize=10.5, fontweight="bold", color="#374151")
    for col, x0 in zip(text_columns(k, ph), (0.0, 0.33, 0.70)):
        y = 0.83
        for text, c, bold in col:
            if text:
                tx.text(x0, y, text, color=c, fontweight="bold" if bold else "normal",
                        family="DejaVu Sans Mono", fontsize=10, transform=tx.transAxes, va="top")
            y -= 0.088 if text else 0.04

if __name__ == "__main__":
    print(len(schedule), "frames")
    mp4 = os.path.join(tempfile.gettempdir(), "gd_3d_zoom.mp4")
    gif = os.path.join(OUT, "gd_3d_zoom.gif")
    FuncAnimation(fig, draw, frames=schedule).save(
        mp4, writer=FFMpegWriter(fps=6, codec="libx264", extra_args=["-crf", "12", "-pix_fmt", "yuv444p"]), dpi=110)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    "fps=6,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
                    "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                    gif], check=True)
    print("wrote", gif, f"{os.path.getsize(gif)/1e6:.1f} MB")
