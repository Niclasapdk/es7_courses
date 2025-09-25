# Create a small script that computes & plots the running sample-mean estimates for part (c).
import numpy as np
import matplotlib.pyplot as plt

# Parameters (same process as before)
N = 15000
MU_U = 3.0
VAR_U = 9.0
SEED = 7

if SEED is not None:
    np.random.seed(SEED)

# Generate U(0..N) and X(1..N)
U = np.random.normal(loc=MU_U, scale=np.sqrt(VAR_U), size=N+1)
X = U[1:] - 0.5 * U[:-1]

# Running (cumulative) mean: \hat{mu}_N for N=1..M
cummean = np.cumsum(X) / np.arange(1, N+1)

# Theory: true mean and an approximate +/- 2σ confidence band for \hat{mu}_N
mu_X_theory = MU_U - 0.5 * MU_U
gamma0 = VAR_U + (0.5**2) * VAR_U  # 11.25
gamma1 = -0.5 * VAR_U              # -4.5
var_hat_mu_approx = (gamma0 + 2*gamma1) / np.arange(1, N+1)  # ≈ 2.25/N
std_hat_mu_approx = np.sqrt(var_hat_mu_approx)

plt.figure()
plt.plot(cummean, label=r"$\hat{\mu}_N$ (running mean)")
plt.axhline(mu_X_theory, label="true mean = 1.5")
plt.plot(mu_X_theory + 2*std_hat_mu_approx, label=r"approx. +2$\sigma$ band")
plt.plot(mu_X_theory - 2*std_hat_mu_approx, label=r"approx. -2$\sigma$ band")
plt.title("Convergence of the sample mean vs. number of samples N")
plt.xlabel("N")
plt.ylabel("mean estimate")
plt.legend()
plt.grid(True)
plt.show()
