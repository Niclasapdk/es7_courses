import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture

# ------------------------------
# Config
# ------------------------------
DATA_PATH = "2D568class.mat"     # change if needed
NORMALIZE_255 = True             # set False if your data is already in [0,1]
RANDOM_STATE = 0
N_COMPONENTS = 3                 # we know there are 3 classes: 5, 6, 8
EPS_REG = 1e-6                   # small regularization for covariances

# ------------------------------
# Load data
# ------------------------------
mat = loadmat(DATA_PATH)

# Expected keys (as in the hint file)
train5 = mat["trn5_2dim"].astype(float)
train6 = mat["trn6_2dim"].astype(float)
train8 = mat["trn8_2dim"].astype(float)

if NORMALIZE_255:
    train5 /= 255.0
    train6 /= 255.0
    train8 /= 255.0

# Labeled (for the supervised Gaussians)
X5, X6, X8 = train5, train6, train8

# Unlabeled pool for the GMM
X_all = np.vstack([X5, X6, X8])
rng = np.random.RandomState(RANDOM_STATE)
rng.shuffle(X_all)

# ------------------------------
# Fit one Gaussian Mixture Model (unsupervised)
# ------------------------------
gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    covariance_type="full",
    random_state=RANDOM_STATE,
    init_params="kmeans",
    n_init=5,
    reg_covar=1e-6,
)
gmm.fit(X_all)

gmm_means = gmm.means_                  # (3, 2)
gmm_covs = gmm.covariances_             # (3, 2, 2)
gmm_weights = gmm.weights_              # (3,)

# ------------------------------
# Fit supervised Gaussian models (one per class)
# ------------------------------
def mean_cov(X):
    mu = X.mean(axis=0)
    # np.cov expects variables in rows -> rowvar=False
    Sigma = np.cov(X, rowvar=False)
    # regularize slightly for stability
    Sigma = Sigma + EPS_REG * np.eye(Sigma.shape[0])
    return mu, Sigma

mu5, S5 = mean_cov(X5)
mu6, S6 = mean_cov(X6)
mu8, S8 = mean_cov(X8)

cls_means = np.vstack([mu5, mu6, mu8])
cls_covs = np.stack([S5, S6, S8], axis=0)
cls_labels = np.array([5, 6, 8])  # for reference

# ------------------------------
# Align GMM components to classes
# We'll match each class mean to the nearest GMM mean (greedy assignment).
# ------------------------------
from scipy.optimize import linear_sum_assignment

# Cost matrix: distances between class means and GMM means
cost = np.linalg.norm(cls_means[:, None, :] - gmm_means[None, :, :], axis=-1)
row_ind, col_ind = linear_sum_assignment(cost)  # optimal assignment (Hungarian)

# Reorder GMM params to best match class order [5, 6, 8]
gmm_means_aligned = gmm_means[col_ind]
gmm_covs_aligned = gmm_covs[col_ind]
gmm_weights_aligned = gmm_weights[col_ind]

# ------------------------------
# Print comparison of means & covariances
# ------------------------------
def cov_summary(S):
    return dict(
        trace=float(np.trace(S)),
        det=float(np.linalg.det(S)),
        eigvals=list(np.linalg.eigvalsh(S)),
    )

print("\n=== MEAN COMPARISON (rows: class 5/6/8; cols: [supervised, gmm-aligned]) ===")
for i, digit in enumerate(cls_labels):
    print(f"\nClass {digit}:")
    print("  supervised mean:", np.round(cls_means[i], 4))
    print("  gmm mean       :", np.round(gmm_means_aligned[i], 4))

print("\n=== COVARIANCE SUMMARY (trace / det / eigenvalues) ===")
for i, digit in enumerate(cls_labels):
    sup = cov_summary(cls_covs[i])
    mix = cov_summary(gmm_covs_aligned[i])
    print(f"\nClass {digit}:")
    print("  supervised:", {k: (np.round(v, 4) if not isinstance(v, list) else np.round(v, 4).tolist()) for k, v in sup.items()})
    print("  gmm       :", {k: (np.round(v, 4) if not isinstance(v, list) else np.round(v, 4).tolist()) for k, v in mix.items()})

# ------------------------------
# Visualization helpers
# ------------------------------
def grid_for_contours(X, padding=0.05, num=300):
    x_min, y_min = X.min(axis=0) - padding
    x_max, y_max = X.max(axis=0) + padding
    xs = np.linspace(x_min, x_max, num)
    ys = np.linspace(y_min, y_max, num)
    XX, YY = np.meshgrid(xs, ys)
    XY = np.column_stack([XX.ravel(), YY.ravel()])
    return XX, YY, XY

def mixture_pdf(XY, weights, means, covs):
    # Weighted sum of component PDFs
    total = np.zeros(XY.shape[0], dtype=float)
    for w, mu, S in zip(weights, means, covs):
        total += w * mvn(mean=mu, cov=S, allow_singular=False).pdf(XY)
    return total

def gaussian_pdf(XY, mu, S):
    return mvn(mean=mu, cov=S, allow_singular=False).pdf(XY)

# ------------------------------
# Visualize: data + GMM hard labels + contours
# ------------------------------
# Hard component labels on the unlabeled pool
gmm_labels = gmm.predict(X_all)

XX, YY, XY = grid_for_contours(X_all, padding=0.05, num=300)

# full mixture density
mix_Z = mixture_pdf(XY, gmm_weights_aligned, gmm_means_aligned, gmm_covs_aligned).reshape(XX.shape)

# per-component (GMM) densities
comp_Z = [gaussian_pdf(XY, gmm_means_aligned[i], gmm_covs_aligned[i]).reshape(XX.shape) for i in range(3)]

# supervised class densities
sup_Z = [
    gaussian_pdf(XY, mu5, S5).reshape(XX.shape),
    gaussian_pdf(XY, mu6, S6).reshape(XX.shape),
    gaussian_pdf(XY, mu8, S8).reshape(XX.shape),
]

plt.figure(figsize=(14, 12))

# (1) Scatter unlabeled data colored by GMM component
ax1 = plt.subplot(2, 2, 1)
ax1.scatter(X_all[:, 0], X_all[:, 1], c=gmm_labels, s=8, alpha=0.7)
ax1.set_title("Unlabeled training data colored by GMM component")
ax1.set_xlabel("PC 1")
ax1.set_ylabel("PC 2")
ax1.grid(True, alpha=0.3)

# (2) Full mixture density contours
ax2 = plt.subplot(2, 2, 2)
ax2.contour(XX, YY, mix_Z, levels=10)
ax2.set_title("GMM: full mixture density")
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.grid(True, alpha=0.3)

# (3) GMM components vs supervised Gaussians (contours)
ax3 = plt.subplot(2, 2, 3)
for i in range(3):
    ax3.contour(XX, YY, comp_Z[i], levels=6, linestyles="solid")
for i in range(3):
    ax3.contour(XX, YY, sup_Z[i], levels=6, linestyles="dashed")
ax3.set_title("Solid: GMM components  |  Dashed: Supervised class Gaussians")
ax3.set_xlabel("PC 1")
ax3.set_ylabel("PC 2")
ax3.grid(True, alpha=0.3)

# (4) Class scatter + both model contours for a quick sanity check
ax4 = plt.subplot(2, 2, 4)
ax4.scatter(X5[:, 0], X5[:, 1], s=6, alpha=0.5, label="Class 5")
ax4.scatter(X6[:, 0], X6[:, 1], s=6, alpha=0.5, label="Class 6")
ax4.scatter(X8[:, 0], X8[:, 1], s=6, alpha=0.5, label="Class 8")
ax4.contour(XX, YY, mix_Z, levels=8)  # mixture
for i in range(3):
    ax4.contour(XX, YY, sup_Z[i], levels=4, linestyles="dashed")
ax4.legend(markerscale=2)
ax4.set_title("Labeled data + mixture (solid) + supervised (dashed)")
ax4.set_xlabel("PC 1")
ax4.set_ylabel("PC 2")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
