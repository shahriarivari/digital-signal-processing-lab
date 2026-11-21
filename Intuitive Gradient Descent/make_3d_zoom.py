"""
GIF version (for the README): main view + zoomed view + numbers, side by side.
    python make_3d_zoom.py   ->  assets/gd_3d_zoom.gif   (needs ffmpeg)
"""
import os
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from gd3d_core import draw_main, draw_zoom, text_lines, zoom_title, N_IT, EVAL, SLOPE, NUDGE, FINAL

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

schedule = []
for k in range(N_IT):
    hold = 6 if k < 3 else 4
    schedule += [(k, EVAL, 1.0)] * hold
    schedule += [(k, SLOPE, 1.0)] * (hold + 2)
    schedule += [(k, NUDGE, 0.0)] * (hold + 2)
    schedule += [(k, NUDGE, i / 7) for i in range(8)]       # the point slides
schedule += [(N_IT, FINAL, 1.0)] * 30

fig = plt.figure(figsize=(17.5, 6))
axm = fig.add_axes([0.0, 0.0, 0.33, 0.93], projection="3d")
axz = fig.add_axes([0.335, 0.0, 0.33, 0.93], projection="3d")
tx = fig.add_axes([0.685, 0.0, 0.315, 0.93])
fig.text(0.165, 0.955, "MAIN VIEW  (whole surface, stays fixed)", ha="center", fontsize=11, fontweight="bold", color="#374151")
ztitle = fig.text(0.50, 0.955, "", ha="center", fontsize=11, fontweight="bold", color="#ea580c")
fig.text(0.685, 0.955, "THE NUMBERS", ha="left", fontsize=11, fontweight="bold", color="#374151")

last = {"k": None}

def draw(frame):
    k, ph, t = frame
    draw_main(axm, k, ph, t)
    # the zoomed view only changes when the step changes
    key = (k, ph, t if ph == NUDGE else 1.0)
    if key != last["k"]:
        draw_zoom(axz, k, ph, t)
        ztitle.set_text(zoom_title(k))
        last["k"] = key
    tx.clear(); tx.axis("off")
    y = 0.95
    for text, col, bold in text_lines(k, ph):
        if text:
            tx.text(0.0, y, text, color=col, fontweight="bold" if bold else "normal",
                    family="DejaVu Sans Mono", fontsize=8.6, transform=tx.transAxes, va="top")
        y -= 0.046 if text else 0.022

if __name__ == "__main__":
    print(len(schedule), "frames")
    mp4 = os.path.join(tempfile.gettempdir(), "gd_3d_zoom.mp4")
    gif = os.path.join(OUT, "gd_3d_zoom.gif")
    FuncAnimation(fig, draw, frames=schedule).save(mp4, writer=FFMpegWriter(fps=8, bitrate=2000), dpi=90)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    "fps=8,scale=1200:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    gif], check=True)
    print("wrote", gif)
