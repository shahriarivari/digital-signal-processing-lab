"""
Interactive page: Prev / Next / Play buttons step through gradient descent.
    python make_interactive.py   ->  docs/index.html   (one self-contained file)
Left: main view (fixed surface, the path grows). Middle: zoomed view (changes each step).
Right: the numbers for the current step.
"""
import base64
import html
import io
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from gd3d_core import (draw_main, draw_zoom, text_lines, zoom_title, N_IT,
                       EVAL, SLOPE, NUDGE, FINAL, F_TEXT, MU)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs")
os.makedirs(OUT, exist_ok=True)

STEP_NAMES = {EVAL: "1/3 · evaluate F", SLOPE: "2/3 · measure the slopes",
              NUDGE: "3/3 · nudge against the slope", FINAL: "finished"}

states = [(k, ph) for k in range(N_IT) for ph in (EVAL, SLOPE, NUDGE)] + [(N_IT, FINAL)]

def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    buf.seek(0)
    out = io.BytesIO()
    Image.open(buf).convert("RGB").save(out, format="JPEG", quality=85, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()

figm = plt.figure(figsize=(6.2, 5.4)); axm = figm.add_axes([0, 0, 1, 1], projection="3d")
figz = plt.figure(figsize=(6.2, 5.4)); axz = figz.add_axes([0, 0, 1, 1], projection="3d")

data = []
for k, ph in states:
    draw_main(axm, k, ph)
    draw_zoom(axz, k, ph)
    lines = "".join(
        f'<div style="color:{c};font-weight:{"700" if b else "400"}">{html.escape(t)}</div>'
        if t else '<div class="gap"></div>'
        for t, c, b in text_lines(k, ph)[3:])            # header lines are shown separately
    label = f"Iteration {k} · step {STEP_NAMES[ph]}" if ph != FINAL else "Finished"
    data.append({"main": png(figm), "zoom": png(figz), "text": lines,
                 "label": label, "ztitle": zoom_title(k)})
    print("rendered", label)

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gradient descent, step by step</title>
<style>
  :root { --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --blue:#2563eb; --orange:#ea580c; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#fafafa; }
  header { padding:18px 24px 6px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:14px; margin:0; }
  .controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:12px 24px; position:sticky; top:0; background:#fafafa; border-bottom:1px solid var(--line); z-index:2; }
  button { font:inherit; font-size:14px; padding:8px 14px; border-radius:8px; border:1px solid #d1d5db; background:white; cursor:pointer; min-width:96px; }
  button:hover:not(:disabled) { border-color:var(--blue); color:var(--blue); }
  button:disabled { opacity:.4; cursor:default; }
  button.primary { background:var(--blue); color:white; border-color:var(--blue); }
  button.primary:hover { color:white; filter:brightness(1.1); }
  input[type=range] { flex:1; min-width:160px; accent-color:var(--blue); }
  .label { font-weight:600; font-size:14px; min-width:290px; }
  .grid { display:grid; grid-template-columns: 1fr 1fr minmax(380px, 0.95fr); gap:14px; padding:14px 24px 24px; }
  .card { background:white; border:1px solid var(--line); border-radius:12px; padding:10px; }
  .card h2 { font-size:13px; letter-spacing:.04em; margin:2px 4px 6px; color:#374151; }
  .card.zoom { border-color:#fdba74; } .card.zoom h2 { color:var(--orange); }
  .card img { width:100%; display:block; }
  .nums { font-family:ui-monospace,"DejaVu Sans Mono",Menlo,monospace; font-size:13px; line-height:1.55; white-space:pre; overflow-x:auto; }
  .nums .gap { height:8px; }
  .fn { font-family:ui-monospace,Menlo,monospace; font-size:13px; font-weight:700; margin:0 4px 2px; }
  .hint { color:var(--muted); font-size:12px; margin:6px 4px 0; }
  @media (max-width:1100px) { .grid { grid-template-columns:1fr; } }
</style></head>
<body>
<header>
  <h1>Gradient descent, step by step</h1>
  <p class="sub">Each iteration: evaluate F at the point → measure the slopes → take a small step against the slope. Repeat until the slopes are ≈ 0.</p>
</header>
<div class="controls">
  <button id="prev">◀ Previous</button>
  <button id="play" class="primary">▶ Play</button>
  <button id="next">Next ▶</button>
  <button id="reset">↺ Reset</button>
  <input id="slider" type="range" min="0" value="0">
  <span class="label" id="label"></span>
</div>
<div class="grid">
  <div class="card"><h2>MAIN VIEW · whole surface (stays fixed)</h2><img id="main" alt="main view"></div>
  <div class="card zoom"><h2 id="ztitle">ZOOMED VIEW</h2><img id="zoom" alt="zoomed view"></div>
  <div class="card">
    <h2>THE NUMBERS</h2>
    <div class="fn">__FTEXT__</div>
    <div class="fn" style="font-weight:400">step size μ = __MU__</div>
    <div style="height:10px"></div>
    <div class="nums" id="nums"></div>
    <p class="hint">Keyboard: ← / → to step, space to play or pause.</p>
  </div>
</div>
<script>
const S = __DATA__;
let i = 0, timer = null;
const $ = id => document.getElementById(id);
const slider = $("slider"); slider.max = S.length - 1;
// preload every image so stepping is instant
S.forEach(s => { new Image().src = s.main; new Image().src = s.zoom; });
function show(n) {
  i = Math.max(0, Math.min(S.length - 1, n));
  const s = S[i];
  $("main").src = s.main; $("zoom").src = s.zoom; $("nums").innerHTML = s.text;
  $("label").textContent = s.label; $("ztitle").textContent = s.ztitle;
  slider.value = i;
  $("prev").disabled = i === 0; $("next").disabled = i === S.length - 1;
  if (i === S.length - 1) stop();
}
function play() {
  if (i === S.length - 1) show(0);
  $("play").textContent = "❚❚ Pause";
  timer = setInterval(() => show(i + 1), 1500);
}
function stop() { clearInterval(timer); timer = null; $("play").textContent = "▶ Play"; }
$("prev").onclick = () => { stop(); show(i - 1); };
$("next").onclick = () => { stop(); show(i + 1); };
$("play").onclick = () => timer ? stop() : play();
$("reset").onclick = () => { stop(); show(0); };
slider.oninput = () => { stop(); show(+slider.value); };
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") { stop(); show(i + 1); }
  else if (e.key === "ArrowLeft") { stop(); show(i - 1); }
  else if (e.key === " ") { e.preventDefault(); timer ? stop() : play(); }
});
show(0);
</script>
</body></html>
"""

page = (PAGE.replace("__DATA__", json.dumps(data))
            .replace("__FTEXT__", html.escape(F_TEXT))
            .replace("__MU__", str(MU)))
path = os.path.join(OUT, "index.html")
with open(path, "w", encoding="utf-8") as f:
    f.write(page)
print("wrote", path, f"({os.path.getsize(path)/1e6:.1f} MB)")
