import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_ar_process(h, sigma2_z, sigma2_w, n_samples=1000):
    """
    Generates Y(n) and X(n) processes.
    
    Parameters:
    h : float, AR coefficient
    sigma2_z : float, variance of process noise Z(n)
    sigma2_w : float, variance of observation noise W(n)
    n_samples : int, number of time steps
    
    Returns:
    Y : array, hidden AR process
    X : array, observed process
    """
    # Ensure WSS condition
    sigma2_y = sigma2_z / (1 - h**2)
    
    # Initialize Y with Y(0)
    Y = np.zeros(n_samples)
    Y[0] = np.random.normal(0, np.sqrt(sigma2_y))
    
    # Generate Z(n) and W(n)
    Z = np.random.normal(0, np.sqrt(sigma2_z), n_samples)
    W = np.random.normal(0, np.sqrt(sigma2_w), n_samples)
    
    # Generate Y(n) for n>=1
    for n in range(1, n_samples):
        Y[n] = h * Y[n-1] + Z[n]
    
    # Generate X(n)
    X = Y + W
    
    return Y, X

def lmmse_batch_estimate(h, sigma2_z, sigma2_w, X):
    """
    Compute the batch (non-recursive) LMMSE estimate of Y(n)
    from observations X(1), ..., X(n).
    """
    n = len(X)

    # Autocorrelation of Y
    def RY(k):
        return sigma2_z / (1 - h**2) * (h ** abs(k))

    # Covariance of X
    Cxx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Cxx[i, j] = RY(i - j)
            if i == j:
                Cxx[i, j] += sigma2_w  # add noise variance on diagonal

    # Cross-covariance between Y(n) and X(k)
    CYX = np.array([RY(n - k) for k in range(1, n + 1)])  # shape (n,)

    # Compute LMMSE estimate: Ŷ(n) = CYX Cxx^{-1} X
    y_hat = CYX @ np.linalg.inv(Cxx) @ X
    return y_hat

# Parameters for different cases
# Parameters for different cases
cases = [
    # 1️⃣ High positive correlation and low noise → smooth Y, X ~ Y, very accurate estimate (low MSE, high Corr)
    {'h': 0.9, 'sigma2_z': 1, 'sigma2_w': 0.1, 'title': 'High positive h, Low observation noise'},
    
    # 2️⃣ Moderate positive correlation → less smooth Y, still low noise, slightly higher MSE
    {'h': 0.5, 'sigma2_z': 1, 'sigma2_w': 0.1, 'title': 'Moderate positive h, Low observation noise'},
    
    # 3️⃣ High negative correlation → alternating Y values, but predictable pattern, low MSE, high Corr
    {'h': -0.9, 'sigma2_z': 1, 'sigma2_w': 0.1, 'title': 'High negative h, Low observation noise'},
    
    # 4️⃣ Weak negative correlation → mild oscillation, harder to predict, medium MSE
    {'h': -0.3, 'sigma2_z': 1, 'sigma2_w': 0.1, 'title': 'Low negative h, Low observation noise'},
    
    # 5️⃣ High process correlation but large observation noise → X is very noisy, estimator struggles (high MSE, low Corr)
    {'h': 0.9, 'sigma2_z': 1, 'sigma2_w': 10, 'title': 'High positive h, High observation noise'},
    
    # 6️⃣ Smooth process but large process noise → Y fluctuates strongly, estimation worse (medium-high MSE)
    {'h': 0.9, 'sigma2_z': 5, 'sigma2_w': 0.1, 'title': 'High positive h, High process noise'}
]

# Generate and plot for each case
# Generate and evaluate
n_samples = 100
results = []

plt.figure(figsize=(12, 14))
for i, case in enumerate(cases):
    Y, X = generate_ar_process(case['h'], case['sigma2_z'], case['sigma2_w'], n_samples=n_samples)

    # Compute LMMSE estimates for each time n
    Yhats = []
    for n in range(1, n_samples + 1):
        y_hat_n = lmmse_batch_estimate(case['h'], case['sigma2_z'], case['sigma2_w'], X[:n])
        Yhats.append(y_hat_n)
    Yhats = np.array(Yhats)

    # Compute MSE and correlation
    mse = np.mean((Y - Yhats)**2)
    corr = np.corrcoef(Y, Yhats)[0, 1]

    results.append({
        'Case': case['title'],
        'h': case['h'],
        'σ²_Z': case['sigma2_z'],
        'σ²_W': case['sigma2_w'],
        'MSE': mse,
        'Corr(Y, Ŷ)': corr
    })

    # Plot
    plt.subplot(len(cases), 1, i+1)
    plt.plot(X, label='X(n)', color='gray', alpha=0.6)
    plt.plot(Y, label='True Y(n)', color='black')
    plt.plot(Yhats, '--', label='LMMSE Ŷ(n)', color='red', linewidth=2)
    plt.title(case['title'])
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# ---- Print nice summary table ----
df_results = pd.DataFrame(results)
pd.set_option('display.precision', 4)
print("\nLMMSE Estimation Results for First 20 Samples:\n")
print(df_results.to_string(index=False))
