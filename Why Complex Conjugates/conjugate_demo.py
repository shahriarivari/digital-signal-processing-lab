"""Numerical checks for the README: why FIR weights carry a complex conjugate."""
import numpy as np

rng = np.random.default_rng(0)

# 1) "Length" of a complex vector: w^T w vs w^H w
w = np.array([1, 1j])
print("w^T w =", w @ w)               # 0  -> a non-zero vector with zero "length"?!
print("w^H w =", np.vdot(w, w))       # 2  -> the real squared length

# 2) Matched taps: input phasors u_k, weights w = u
u = np.exp(1j * np.array([0.4, 1.5, 2.7, 3.9, 5.2]))
print("|sum w_k u_k|  =", abs(np.sum(u * u)))           # ~0.31, terms cancel
print("|sum w_k* u_k| =", abs(np.sum(np.conj(u) * u)))  # 5.00, terms add coherently

# 3) Wiener solution, both conventions, on a complex system-identification problem
M, N = 4, 20000
h_true = rng.normal(size=M) + 1j * rng.normal(size=M)
x = (rng.normal(size=N) + 1j * rng.normal(size=N)) / np.sqrt(2)
U = np.array([x[n - M + 1:n + 1][::-1] for n in range(M - 1, N)])   # rows = u(n)^T
d = U @ h_true + 0.05 * (rng.normal(size=len(U)) + 1j * rng.normal(size=len(U)))

R = U.T @ U.conj() / len(U)          # E[u u^H]
p = U.T @ d.conj() / len(U)          # E[u d*]
w_H = np.linalg.solve(R, p)                    # convention y = w^H u :  R w = p
w_T = np.linalg.solve(R.conj(), p.conj())      # convention y = w^T u :  R* w = p*

print("w^H-convention weights:", np.round(w_H, 3))
print("w^T-convention weights:", np.round(w_T, 3))
print("are they conjugates?   ", np.allclose(w_T, w_H.conj()))
print("same output?           ", np.allclose(U @ w_H.conj(), U @ w_T))
