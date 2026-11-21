<div align="center">

# 📉 Least Squares, Visually

**How to find the line that best describes your data, from a plain `for` loop to the closed-form solution and the LMS adaptive filter.**

<img src="assets/hero_squares.gif" width="460" alt="Gradient descent shrinking the squared residuals">

*Each shaded square has area eᵢ². Least squares means **making the total area of the squares as small as possible**.*

</div>

---

## 1. A line has only two knobs: `w` and `b`

$$
y = w \cdot x + b
$$

<div align="center">
<img src="assets/line_anatomy.png" width="760" alt="What w and b do to a line">
</div>

* **`b` (intercept):** the height where the line crosses the y-axis, i.e. the value of y at x = 0. Changing `b` **slides** the line up or down.
* **`w` (slope):** rise ÷ run. Every step of +1 in x moves y by `w`. Changing `w` **tilts** the line.

So *"find the best line"* really means *"find the best two numbers, `w` and `b`"*.

---

## 2. What does "best" mean?

For each data point the line makes a prediction $\hat y_i = w x_i + b$. The miss is the **residual**
$e_i = y_i - \hat y_i$, the vertical gap between the point and the line.

We square every miss (so the signs can't cancel and big misses count more) and average them:

$$
J(w,b) = \frac{1}{N}\sum_{i=1}^{N}\big(y_i - (w x_i + b)\big)^2
$$

$J$ is **one number that tells us how bad a line is**. Least squares means picking the `w` and `b` that make $J$ as small as possible.

> ⚠️ It minimises **vertical** gaps, not the perpendicular distance to the line. Minimising perpendicular distance is *total least squares* (PCA).

---

## 3. Two ways to look at the same line ⭐ *(read this before the animations)*

Every line is just a pair of numbers `(w, b)`. So we can draw each line in **two places**:

<div align="center">
<img src="assets/how_to_read.png" width="760" alt="Three lines shown as three points">
</div>

| Left: **data space** | Right: **parameter space** |
|---|---|
| axes are `x` and `y` | axes are **slope `w`** and **intercept `b`** |
| a line is drawn as a **line** | the same line is drawn as a **single dot** |
| you see how well it fits the points | the background colour is the error $J$ of the line at that spot |

**How to read the right plot.** Think of it as a **map of a valley seen from above**:
* each spot on the map is one possible line,
* the colour is its error (**dark purple = small error**, **yellow = big error**),
* the ★ at the bottom of the valley is **the best line**.

Line **A** is too flat, so its dot sits in a high (lighter) area. Line **B** is far too steep, so it is high up in the yellow region. Line **C** fits well, so it sits at the bottom of the valley.
**When the dot moves on the right, the line changes on the left.** Moving right makes the line steeper, and moving up raises it.

---

## 4. The most intuitive optimiser: a `for` loop that tries everything

```python
best_J = float("inf")
for w in slopes:                       # every candidate slope
    for b in intercepts:               # every candidate intercept
        J = sum((yi - (w*xi + b))**2 for xi, yi in zip(x, y)) / N   # how bad is this line?
        if J < best_J:                 # better than the best so far? keep it
            best_J, best_w, best_b = J, w, b
```

<div align="center">
<img src="assets/for_loop_search.gif" width="680" alt="Brute force grid search">
</div>

**What you are watching:**
* 🔴 **Red** = the line being tried right now. On the left it is a line (with its gaps to the points); on the right it is the red dot. The dot walks through a grid: one column per slope `w`, every intercept `b` in that column.
* 🟢 **Green** = the best line found so far. It only moves when the loop finds a line with a smaller $J$.
* ⚪ **White dots** = all the lines already tried.
* As the red dot gets close to the dark valley floor, the red line hugs the data and $J$ drops.

It works, but the answer is only as precise as the grid, and the work grows like $(\text{grid size})^{\text{number of parameters}}$.
The map shows something useful: the error surface is **one smooth valley**. We can use that shape directly instead of trying everything.

---

## 5. Why there is only **one** best line (convexity)

Keep `b` fixed and look at how the error changes with `w`.
A single data point contributes $(y_i - w x_i - b)^2$. The part inside the brackets is a straight line in `w`, and squaring a straight line gives a **U-shaped parabola**.
The total error $J$ is just the sum of these U-shapes, and **adding U-shapes always gives another U-shape**. Exactly the same happens for `b`.

<div align="center">
<img src="assets/convexity.png" width="820" alt="Sum of parabolas is a parabola">
</div>

So in every direction, $J$ is a U. Together they form a bowl with **one bottom and no traps** (right panel: the kind of curve we *don't* get). This property is called **convex**.

**The second-derivative check.** A curve bends upward (a U) when its second derivative is positive:

$$
\frac{d^2 J}{d w^2} = 2 \cdot \text{average}(x_i^2), \qquad \frac{d^2 J}{d b^2} = 2
$$

Both are positive **for every** `w` and `b`. The bowl curves upward everywhere, so it can't have a hump or a second dip.

### Finding the bottom directly

At the bottom of the bowl the ground is flat, so the derivative with respect to **both** knobs is zero:

| Condition | What it says | Result |
|---|---|---|
| $\dfrac{\partial J}{\partial b} = 0$ | the misses above and below the line cancel out | $b_{\text{best}} = \bar y - w_{\text{best}}\ \bar x$ |
| $\dfrac{\partial J}{\partial w} = 0$ | the misses are not tilted against $x$ | $w_{\text{best}} = \dfrac{\mathrm{Cov}(x,y)}{\mathrm{Var}(x)}$ |

In words:
* the best line always passes through the **average point** $(\bar x, \bar y)$ of the data,
* its slope is **how much x and y move together, divided by how much x moves on its own**.

No loop and no guessing: one formula gives the exact answer. (With more parameters the same idea becomes one matrix equation, solved in Python by `np.linalg.lstsq(X, y)`.)

---

## 6. Iteratively: gradient descent

Instead of jumping straight to the bottom, we can also **walk downhill**. At every step we look at the slope of the ground and take a small step against it:

$$
w \leftarrow w - \eta \ \frac{\partial J}{\partial w}, \qquad b \leftarrow b - \eta \ \frac{\partial J}{\partial b}
$$

$\eta$ is the step size. If it is too small, the walk is slow. If it is too big, each step overshoots the valley and the error **explodes**.

<div align="center">
<img src="assets/gd_bowl.gif" width="680" alt="Gradient descent path on the cost bowl">
</div>

These are the same two views as before. We start at one line (□ *start*) and keep stepping **downhill** on the map, so the line on the left gets a little better with every step. The dashed green line is the target.

The path rushes down the steep side, then **crawls** along the long, flat valley floor. The valley is a thin ellipse because changing `w` affects the error much more than changing `b`.
Centring the data ($x \to x - \bar x$) makes the valley round and the walk much faster.

Why walk at all if a formula exists? Because the formula needs **all the data at once**. In signal processing the data often arrives **one sample at a time**, and that is exactly the next step.

---

## 7. From least squares to **LMS** (Least *Mean* Squares) — the adaptive filter

### 7.1 The setting: an adaptive filter

In DSP we often have a system whose behaviour we don't know (an acoustic echo path, a communication channel, 50 Hz mains hum leaking into an ECG). We want a filter that **learns it by itself, while the signal is running**.

```mermaid
flowchart LR
    X["input x[n]"] --> S["unknown system"]
    S --> D["desired d[n]"]
    X --> F["adaptive FIR filter<br/>weights w"]
    F --> Y["output y[n]"]
    D --> E(("d[n] − y[n]"))
    Y --> E
    E -- "error e[n] adjusts the weights" --> F
```

The filter is an FIR filter with $M$ taps. At time $n$ it looks at the last $M$ input samples:

$$
\mathbf{x}[n] = \big[\ x[n],\ x[n-1],\ \dots,\ x[n-M+1]\ \big]^T, \qquad y[n] = \mathbf{w}^T \mathbf{x}[n] = \sum_{k=0}^{M-1} w_k\ x[n-k]
$$

$$
e[n] = d[n] - y[n]
$$

**This is the same problem as fitting our line.** The line $\hat y = w x + b$ is an adaptive filter with 2 weights, $\mathbf{w} = [w,\ b]$, and input $\mathbf{x} = [x,\ 1]$.

### 7.2 Why "Least *Mean* Squares"?

* **Least squares** minimises the **sum** of squared errors over a fixed batch of data we already have.
* For a signal that keeps running there is no fixed batch. Instead we minimise the **expected (mean)** squared error:

$$
J(\mathbf{w}) = \mathbb{E}\big[e^2[n]\big]
$$

This is still the same convex bowl, now over $M$ weights. Its exact bottom is the **Wiener solution** $\mathbf{w}_{\text{opt}} = \mathbf{R}^{-1}\mathbf{p}$, where $\mathbf{R}$ is the autocorrelation matrix of the input and $\mathbf{p}$ is the cross-correlation between the input and $d[n]$.
It is the statistical twin of our closed-form formula: Var(x) becomes $\mathbf{R}$ and Cov(x,y) becomes $\mathbf{p}$.

The problem is that in real time we **don't know** $\mathbf{R}$ and $\mathbf{p}$, and inverting $\mathbf{R}$ every sample would be far too expensive.

### 7.3 The LMS trick (Widrow & Hoff, 1960)

1. Use gradient descent on the bowl: $\ \mathbf{w} \leftarrow \mathbf{w} - \eta\ \nabla J$.
2. The true gradient is $\nabla J = -2\ \mathbb{E}\big[e[n]\ \mathbf{x}[n]\big]$, which needs that unknown expectation.
3. **The trick:** drop the expectation and use the current sample only, $\ \nabla J \approx -2\ e[n]\ \mathbf{x}[n]$. It is a noisy guess, but correct on average.
4. Put it into step 1 and absorb the 2 into the step size $\mu$:

$$
\boxed{\ \mathbf{w}[n+1] = \mathbf{w}[n] + \mu\ e[n]\ \mathbf{x}[n]\ }
$$

**The whole algorithm, run once per new sample:**

```python
y = w @ x_vec            # 1. filter output         (M multiplications)
e = d - y                # 2. error
w = w + mu * e * x_vec   # 3. nudge every weight    (M multiplications)
```

How to read the update: **error × input**. If the error is large and a tap's input was large, that tap was probably responsible, so it gets corrected most. When the error is zero, nothing changes.

### 7.4 Watching it learn

#### 7.4.1 One update by hand

Let's do a single LMS step with real numbers from our line-fitting data ($\mu = 0.008$).
The filter starts with a bad guess, $w = -0.5$ and $b = 7.5$, i.e. the line $y = -0.5x + 7.5$.

| Step | Calculation | Result |
|---|---|---|
| a sample arrives | $x = 7.74$, $y = 7.54$ | |
| 1. predict | $-0.5 \cdot 7.74 + 7.5$ | $3.63$ |
| 2. error | $7.54 - 3.63$ | $e = +3.91$ (the point is **above** the line) |
| 3. update slope | $\Delta w = \mu\ e\ x = 0.008 \cdot 3.91 \cdot 7.74$ | $+0.242$ |
| 3. update intercept | $\Delta b = \mu\ e \cdot 1 = 0.008 \cdot 3.91$ | $+0.031$ |
| new line | | $y = -0.258x + 7.531$ |

Check: the new line predicts $5.53$ at $x = 7.74$, so the error on this sample dropped from $3.91$ to $2.01$.
Three things to notice:

1. **The sign of the error decides the direction.** The point was above the line ($e$ positive), so the line moved **up** at that spot.
2. **The input decides who gets corrected most.** The slope's input was $x = 7.74$ and the intercept's input is always $1$, so the slope got a correction 7.74 times bigger. Weights whose input was large had more influence on the error, so they get more of the correction.
3. **Where the stability limit comes from.** One update shrinks the error on that sample by the factor $(1 - \mu \cdot \lVert \mathbf{x} \rVert^2) = 1 - 0.008 \cdot (7.74^2 + 1) = 0.51$. If $\mu$ were so big that this factor went below $-1$, every update would **overshoot** and make the error bigger, and the filter would explode.

#### 7.4.2 The animation: many of those updates in a row

<div align="center">
<img src="assets/lms_sample_by_sample.gif" width="480" alt="LMS updating sample by sample">
</div>

**How to read each frame:**

| On screen | Meaning |
|---|---|
| 🔴 circled point | the **one** sample used in this update |
| thick red vertical segment | its error $e$: the gap between the point and the line *before* the update |
| red **dotted** line | the line before the update |
| red **solid** line | the line after the update |
| dashed green line | the exact least-squares answer (what LMS should approach) |
| numbers at the top | $e$, and the corrections $\Delta w = \mu e x$ and $\Delta b = \mu e$ |

**What to look for:**
* **Early on:** the errors are big, so the line swings a lot. Mostly the **slope** changes, as explained in point 2 above.
* **Middle:** the slope is roughly right, but the line is still too high. The intercept creeps down slowly because its corrections are tiny ($\Delta b$ is several times smaller than $\Delta w$, because $x$ is usually much bigger than 1). This is the same slow, flat valley we saw in gradient descent.
* **End:** the line sits on top of the green line, but it **never fully stops**. Every new sample tugs it a little toward that point. This permanent wiggle is the price of using one sample instead of all of them.

#### 7.4.3 A real DSP case: identifying an unknown system

Now a proper adaptive-filter experiment:
* the "unknown system" is an FIR filter $h = [0.8,\ -0.4,\ 0.2,\ 0.1]$ (4 taps),
* the input $x[n]$ is white noise with power $P_x = 1$,
* the desired signal is the system's output plus a little measurement noise, $d[n] = (h \ast x)[n] + v[n]$, with noise power $\sigma_v^2 = 0.0025$,
* an LMS filter with $M = 4$ taps, all starting at **0**, must discover $h$ from $x[n]$ and $d[n]$ alone.

**Graph 1: the weights over time**

<div align="center">
<img src="assets/lms_weights.png" width="780" alt="LMS weights converging to the true system">
</div>

* **x-axis:** time, i.e. how many samples the filter has seen. **y-axis:** the value of each weight.
* **Each coloured line is one tap** of the adaptive filter. The **dotted line of the same colour** is the true value it should find, which the filter itself never sees.
* **Yellow region (≈ first 300 samples), learning:** all weights start at 0 and move towards their targets. They are pulled by the error. As long as the filter is wrong, the error is big and the weights keep moving.
* **White region, steady state:** the weights have arrived (the final values on the right match $h$ to within 0.002). The error is now mostly the measurement noise, which can't be predicted, so the updates become tiny random nudges, which is the small wiggle you see.
* With $\mu = 0.01$ this took about 300 samples. A larger $\mu$ would arrive sooner but wiggle more.

**Graph 2: the learning curve, the most important graph in adaptive filtering**

<div align="center">
<img src="assets/lms_learning_curve.png" width="800" alt="LMS learning curves for three step sizes">
</div>

**The axes:**
* **x-axis:** time (samples).
* **y-axis:** the squared error $e^2[n]$ **in decibels**, $10 \log_{10}(e^2)$. We use dB because the error drops by a factor of hundreds: 0 dB means "as big as the signal" and −20 dB means "100× smaller".

**Why the grey mess, and why average 100 runs?**
The faint grey curve is **one single run**. $e^2[n]$ jumps up and down by ±20 dB from sample to sample, because the input is random, so you can't see the trend.
The learning curve is defined as the **mean** squared error $\mathbb{E}[e^2[n]]$. We estimate it by repeating the whole experiment 100 times (new random input each time) and averaging the 100 values at each $n$. That gives the smooth coloured curves. (This is also the "mean" in Least *Mean* Squares.)

**Reading the curves from left to right:**

1. **Start, at about −0.7 dB.** All weights are 0, so the filter outputs nothing and the error is the whole desired signal: $\mathbb{E}[d^2] = 0.8^2 + 0.4^2 + 0.2^2 + 0.1^2 + 0.0025 = 0.85$, which is −0.7 dB.
2. **Descent, a straight line in dB.** For white input the remaining "excess" error shrinks by a fixed factor $(1-\mu)^2$ every sample. That is an exponential decay, and an exponential becomes a **straight line on a dB scale**. Its slope is about $8.7\mu$ dB per sample, so **a bigger μ gives a steeper line** (blue → green → red). The thin dashed black lines are this theory, and they sit right on the simulation.
3. **Floor, at −26 dB.** Even with perfect weights ($\mathbf{w} = h$) the error is still the measurement noise $v[n]$, which no filter can predict from $x[n]$. So the curve can never go below $\sigma_v^2 = 0.0025$, i.e. −26 dB (solid black line). Reaching that floor means **the filter has learned everything that can be learned**.
4. **Slightly above the floor, called misadjustment.** In steady state the weights keep wiggling (Graph 1), and each wiggle adds a little error. Theory says the extra fraction is about $\mu \cdot M \cdot P_x / 2$. That is invisible for small $\mu$, but for $\mu = 0.08$ it is 16%, which is why the **red curve settles visibly above the black line**.

**The numbers, measured from the graph and compared with theory:**

| step size $\mu$ | time constant $1/(2\mu)$ | samples until within 3 dB of the floor (measured) | misadjustment (theory) | final error level (measured) |
|---|---|---|---|---|
| 0.002 🔵 | 250 samples | ≈ 1320 | 0.4 % | −24.8 dB (still not finished at n = 2000) |
| 0.01 🟢 | 50 samples | ≈ 290 | 2 % | −25.9 dB |
| 0.08 🔴 | 6 samples | ≈ 47 | 16 % | −25.1 dB (sits above the floor) |

**The take-away, which is the central design decision for every LMS filter:**
**μ trades speed for accuracy.** A large $\mu$ learns fast and reacts quickly when the system changes, but it is noisier in steady state. A small $\mu$ is precise but slow. And if $\mu$ goes above about $2/(M \cdot P_x)$ (here 0.5), the curve shoots **up** instead of down: the filter diverges.

### 7.5 The family at a glance

| | Least Squares | Wiener filter | **LMS** | RLS |
|---|---|---|---|---|
| Minimises | sum of e² over a batch | mean of e² | mean of e² (noisy estimate) | weighted sum of e² |
| Needs | all data at once | true statistics R, p | nothing, just the current sample | the past data, summarised recursively |
| Cost per sample | — (one big solve) | — (one big solve) | about 2M multiplications | about M² multiplications |
| Converges | exact, one shot | exact | slowly, keeps jittering | fast |
| Tracks a changing system | ❌ | ❌ | ✅ | ✅ |

**Where LMS is used:** echo cancellation in phones and video calls, channel equalisation in modems, active noise cancelling headphones, and removing 50 Hz hum from biomedical signals (an adaptive notch filter).

---

## 8. When the data is **not** linear

Least squares will always give you *a* line, even a useless one. **Check the residuals:** if they show a pattern, the model is wrong.

<div align="center">
<img src="assets/nonlinear_residuals.png" width="720" alt="Residual diagnosis for non-linear data">
</div>

**"Linear" means linear in the parameters, not in x.** Adding a column $x^2$ to $X$ gives a parabola fit that is *still* ordinary, convex least squares.

### Recommendations
1. **Plot the data and the residuals** before trusting $R^2$.
2. **Transform the features:** $x^2,\ \log x,\ \sin(\omega x)$. You still have the same convex least squares.
3. **Transform the target** for multiplicative laws: $y = ae^{kx} \Rightarrow \ln y = \ln a + kx$.
4. **Use splines or piecewise models** when the behaviour changes from one region to another.
5. **Use non-linear least squares** (`scipy.optimize.curve_fit`, Levenberg–Marquardt) when parameters sit inside the function, e.g. $Ae^{-t/\tau}$. It is no longer convex, so a good initial guess matters.
6. **Guard against overfitting:** check the error on held-out data, or use ridge regression $(X^TX+\lambda I)^{-1}X^Ty$.
7. **Deal with outliers:** use Huber loss or RANSAC.

---

## Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook least_squares_explained.ipynb   # full step-by-step notebook
python make_animations.py                        # regenerate every GIF in assets/
```

```
├── README.md
├── least_squares_explained.ipynb   # theory + code, cell by cell
├── make_animations.py              # builds all GIFs/figures
├── requirements.txt
└── assets/                         # generated animations
```

<div align="center"><sub>Made as part of the Master of Smart Systems, Hanze University of Applied Sciences, Groningen.</sub></div>
