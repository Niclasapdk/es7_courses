# %%
import numpy as np
from scipy.stats import multivariate_normal as mvn  # Gaussian log-likelihoods
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# 1) Load & prep data
# -----------------------------
# MNIST subsets (digits 5,6,8). Each row = flattened 28x28 (784 feats).
# Normalize to [0,1] so everything is on the same scale.
train5 = np.loadtxt("mnist_all/train5.txt") / 255.0
train6 = np.loadtxt("mnist_all/train6.txt") / 255.0
train8 = np.loadtxt("mnist_all/train8.txt") / 255.0

test5  = np.loadtxt("mnist_all/test5.txt") / 255.0
test6  = np.loadtxt("mnist_all/test6.txt") / 255.0
test8  = np.loadtxt("mnist_all/test8.txt") / 255.0

# Labels as ints (sklearn likes integer class labels)
train5_target = np.full(len(train5), 5, dtype=int)
train6_target = np.full(len(train6), 6, dtype=int)
train8_target = np.full(len(train8), 8, dtype=int)

test5_target  = np.full(len(test5), 5, dtype=int)
test6_target  = np.full(len(test6), 6, dtype=int)
test8_target  = np.full(len(test8), 8, dtype=int)

# Concatenate into a single train/test set
train_data    = np.vstack([train5, train6, train8]).astype(np.float32)
train_targets = np.concatenate([train5_target, train6_target, train8_target])

test_data     = np.vstack([test5, test6, test8]).astype(np.float32)
test_targets  = np.concatenate([test5_target, test6_target, test8_target])

# Keep a fixed class order for later indexing
classes = np.array([5, 6, 8], dtype=int)

# Shuffle train set so there’s no weird ordering effects
rng = np.random.default_rng(0)  # fixed seed so runs are reproducible
perm = rng.permutation(len(train_data))
train_data, train_targets = train_data[perm], train_targets[perm]

print("Train:", train_data.shape, train_targets.shape, "unique:", np.unique(train_targets))
print("Test: ", test_data.shape,  test_targets.shape,  "unique:", np.unique(test_targets))

# -----------------------------
# 2) Helpers
# -----------------------------
def fit_gaussians_2d(Z, y, labels, eps=1e-6):
    """
    Fit a 2D Gaussian per class (mean + covariance) + class priors.
    bias=True -> ML covariance (divide by N). Add tiny eps*I to avoid singular covs.
    """
    params = {}
    priors = {}
    N = len(y)
    for c in labels:
        Zc = Z[y == c]
        mu = Zc.mean(axis=0)
        cov = np.cov(Zc, rowvar=False, bias=True) + eps * np.eye(2)
        params[c] = {"mu": mu, "cov": cov}
        priors[c] = len(Zc) / N  # empirical prior
    return params, priors

def predict_bayes_2d(Z, labels, params, priors):
    """
    Bayes classification in 2D: argmax_c log P(c) + log N(z | mu_c, Sigma_c).
    Using logs to keep things numerically stable.
    """
    log_posts = np.zeros((len(Z), len(labels)), dtype=np.float64)
    for j, c in enumerate(labels):
        mu, cov = params[c]["mu"], params[c]["cov"]
        log_lik = mvn(mean=mu, cov=cov).logpdf(Z)
        log_posts[:, j] = np.log(priors[c]) + log_lik
    return labels[np.argmax(log_posts, axis=1)]

def plot_2d_scatter(Z, y, labels, title):
    """
    Quick sanity-plot to see class separation in 2D.
    """
    plt.figure()
    markers = ["o", "s", "^"]  # one marker per class (aligned with labels order)
    for c, m in zip(labels, markers):
        Zc = Z[y == c]
        plt.scatter(Zc[:, 0], Zc[:, 1], s=8, alpha=0.5, label=str(c), marker=m)
    plt.legend(title="Class")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 3) LDA → 2D → Gaussian Bayes
# -----------------------------
# LDA is supervised: uses labels to find directions that separate classes.
# With 3 classes, max LDA comps = 2 -> perfect for 2D plotting + simple Gaussians.
lda = LDA(n_components=2, solver="svd")
lda.fit(train_data, train_targets)   # learn projection 784D -> 2D

Ztr_lda = lda.transform(train_data)  # train in 2D
Zte_lda = lda.transform(test_data)   # test in 2D

# Fit per-class Gaussians + priors in LDA space
params_lda, priors_lda = fit_gaussians_2d(Ztr_lda, train_targets, classes)

# Classify test set in LDA space
y_pred_lda = predict_bayes_2d(Zte_lda, classes, params_lda, priors_lda)

# Accuracy + diagnostics
acc_lda = (y_pred_lda == test_targets).mean()
print(f"LDA-2D Gaussian Bayes — Test accuracy: {acc_lda:.4f}")

print("\nConfusion matrix (LDA space):")
print(confusion_matrix(test_targets, y_pred_lda, labels=classes))

print("\nClassification report (LDA space):")
print(classification_report(test_targets, y_pred_lda, labels=classes, digits=4))

# Plot train embedding just to eyeball separation
plot_2d_scatter(Ztr_lda, train_targets, classes, "LDA (2D) on Train")

# -----------------------------
# 4) PCA comparison (optional)
# -----------------------------
# PCA is unsupervised: maximizes variance, not class separation.
# Do the same 2D → Gaussian → Bayes pipeline and compare to LDA.
pca = PCA(n_components=2, svd_solver="full", random_state=0)
pca.fit(train_data)

Ztr_pca = pca.transform(train_data)
Zte_pca = pca.transform(test_data)

params_pca, priors_pca = fit_gaussians_2d(Ztr_pca, train_targets, classes)
y_pred_pca = predict_bayes_2d(Zte_pca, classes, params_pca, priors_pca)

acc_pca = (y_pred_pca == test_targets).mean()
print(f"\nPCA-2D Gaussian Bayes — Test accuracy: {acc_pca:.4f}")

print("\nConfusion matrix (PCA space):")
print(confusion_matrix(test_targets, y_pred_pca, labels=classes))

print("\nClassification report (PCA space):")
print(classification_report(test_targets, y_pred_pca, labels=classes, digits=4))

# Visual check for PCA too
plot_2d_scatter(Ztr_pca, train_targets, classes, "PCA (2D) on Train")

# -----------------------------
# 5) Direct LDA baseline (optional)
# -----------------------------
# Baseline: let sklearn’s LDA classify directly in 784D (no 2D proj, no Gaussians).
# Usually better because we don’t throw away info.
lda_direct = LDA()
lda_direct.fit(train_data, train_targets)
acc_lda_direct = lda_direct.score(test_data, test_targets)
print(f"\nSklearn LDA (direct classifier) — Test accuracy: {acc_lda_direct:.4f}")

# Notes to self:
# - Make sure mnist_all/*.txt exists and rows really have 784 cols.
# - Keep the /255 normalization the same for train/test.
# - eps in cov keeps matrices invertible in case clusters are tight/degenerate.
# - 2D Gaussian model is intentionally simple; 784D LDA usually wins on accuracy.