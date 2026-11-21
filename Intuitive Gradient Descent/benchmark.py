"""
Closed form vs gradient descent on a quadratic cost J(w) = w^T R w - 2 p^T w + const.
Closed form: solve R w = p  (O(N^3)).  Gradient descent: w <- w - mu (R w - p)  (O(N^2) per step).
How many steps GD needs depends on the eigenvalue spread (condition number) of R.
"""
import time
import numpy as np

def run(N, cond, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    lam = np.geomspace(1.0, cond, N)               # eigenvalues of R
    R = (Q * lam) @ Q.T
    p = rng.standard_normal(N)

    t = time.perf_counter(); w_cf = np.linalg.solve(R, p); t_cf = time.perf_counter() - t

    mu = 2.0 / (lam.min() + lam.max())             # best fixed step (needs eigenvalue bounds!)
    w = np.zeros(N); k = 0
    t = time.perf_counter()
    while np.linalg.norm(w - w_cf) > tol * np.linalg.norm(w_cf) and k < 200_000:
        w -= mu * (R @ w - p); k += 1
    t_gd = time.perf_counter() - t
    return t_cf, t_gd, k

if __name__ == "__main__":
    print(f"{'N':>6} {'cond(R)':>9} {'solve [ms]':>11} {'GD [ms]':>9} {'GD steps':>9}")
    for N in (500, 2000):
        for cond in (2, 10, 1000):
            a, b, k = run(N, cond)
            print(f"{N:>6} {cond:>9} {a*1e3:>11.1f} {b*1e3:>9.1f} {k:>9}")
