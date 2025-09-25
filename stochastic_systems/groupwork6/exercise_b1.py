# Create a ready-to-run Python script for simulating the WSS process in part (b).
"""
Simulate the WSS process:
    X(n) = U(n) - 0.5 * U(n-1),  with  U(n) ~ i.i.d. N(3, 9)

What it does:
- Generates a realization of length N.
- Prints sample mean/variance and compares to theory.
- Computes sample autocorrelation at lags -20..20.
- Plots time series, histogram with theoretical PDF, and ACF.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- parameters you can change ---
N = 5000          # number of samples in the realization
MU_U = 3.0        # mean of U
VAR_U = 9.0       # variance of U
SEED = 42         # RNG seed for reproducibility (set to None for random)
# ---------------------------------

if SEED is not None:
    np.random.seed(SEED)

sigma_u = np.sqrt(VAR_U)

# 1) draw U(0..N) so we have U(n-1) available up to n=N
U = np.random.normal(loc=MU_U, scale=sigma_u, size=N+1)

# 2) build X(1..N)
X = U[1:] - 0.5 * U[:-1]

# --- theory ---
mu_X_theory = MU_U - 0.5 * MU_U
var_X_theory = VAR_U + (0.5**2) * VAR_U          # independence => no cross term
rho1_theory = (-0.5 * VAR_U) / var_X_theory      # normalized lag-1 ACF = -0.4

# --- sample stats ---
mu_hat = np.mean(X)
var_hat = np.var(X, ddof=0)

def acf(x, maxlag):
    """Normalized autocorrelation using biased estimator (divide by N)."""
    x = np.asarray(x)
    x = x - np.mean(x)
    N = len(x)
    denom = np.dot(x, x) / N
    lags = np.arange(-maxlag, maxlag+1)
    rho = np.zeros_like(lags, dtype=float)
    for i, k in enumerate(lags):
        if k >= 0:
            num = np.dot(x[:N-k], x[k:]) / N
        else:
            num = np.dot(x[-k:], x[:N+k]) / N
        rho[i] = num / denom
    return lags, rho

lags, rho = acf(X, maxlag=20)
rho1_hat = rho[lags==1][0]

print("=== SAMPLE vs THEORY ===")
print(f"mean:     sample {mu_hat:.4f}  | theory {mu_X_theory:.4f}")
print(f"variance: sample {var_hat:.4f} | theory {var_X_theory:.4f}")
print(f"rho(1):   sample {rho1_hat:.4f} | theory {rho1_theory:.4f}")

# --- plots ---
plt.figure()
plt.plot(X[:500])
plt.title("X(n) time series (first 500 samples)")
plt.xlabel("n"); plt.ylabel("X(n)")
plt.grid(True)

plt.figure()
plt.hist(X, bins=60, density=True, alpha=0.6, label="hist(X)")
xs = np.linspace(np.min(X), np.max(X), 500)
pdf = (1/np.sqrt(2*np.pi*var_X_theory)) * np.exp(-(xs-mu_X_theory)**2/(2*var_X_theory))
plt.plot(xs, pdf, label="theory PDF")
plt.title("Histogram with theoretical Normal PDF")
plt.xlabel("x"); plt.ylabel("density"); plt.legend(); plt.grid(True)

plt.figure()
plt.stem(lags, rho, basefmt=" ")
plt.title("Sample ACF (normalized)")
plt.xlabel("lag k"); plt.ylabel("rho_X(k)")
plt.grid(True)

plt.show()

'''
# write the script to disk
path = "/mnt/data/wss_process_sim.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(script)

print(f"Saved script to {path}")
'''