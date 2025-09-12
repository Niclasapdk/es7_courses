#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import multivariate_normal

# -------------------------
# Load training and test data
# -------------------------

# Train data
train_x = np.loadtxt("dataset1_G_noisy_ASCII/trn_x.txt")
train_x_label = np.loadtxt("dataset1_G_noisy_ASCII/trn_x_class.txt")

train_y = np.loadtxt("dataset1_G_noisy_ASCII/trn_y.txt")
train_y_label = np.loadtxt("dataset1_G_noisy_ASCII/trn_y_class.txt")

# Test data
test_xy = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy.txt")
test_xy_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_class.txt")

test_xy_126 = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_126.txt")
test_xy_126_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_126_class.txt")

# -------------------------
# Step 1: Compute statistics
# -------------------------

# x statistics
train_x_mean = np.mean(train_x, axis=0)
train_x_cov = np.cov(train_x, rowvar=False)

# y statistics
train_y_mean = np.mean(train_y, axis=0)
train_y_cov = np.cov(train_y, rowvar=False)

# priors (from training set size)
prior_x = len(train_x) / (len(train_x) + len(train_y))
prior_y = len(train_y) / (len(train_x) + len(train_y))

# -------------------------
# Step 2: Likelihood function
# -------------------------

def likelihood(data, mean, cov):
    return multivariate_normal.pdf(data, mean=mean, cov=cov)

# -------------------------
# (a) Classify tst_xy
# -------------------------

likelihood_x = likelihood(test_xy, train_x_mean, train_x_cov)
likelihood_y = likelihood(test_xy, train_y_mean, train_y_cov)

posterior_x = likelihood_x * prior_x
posterior_y = likelihood_y * prior_y

classification = np.where(posterior_x > posterior_y, 1, 2)
accuracy_xy = np.mean(classification == test_xy_label)

print(f"(a) Accuracy on tst_xy: {accuracy_xy*100:.2f}%")

# -------------------------
# (b) Classify tst_xy_126 with uniform prior
# -------------------------

prior_x_uniform = 0.5
prior_y_uniform = 0.5

likelihood_x_uniform = likelihood(test_xy_126, train_x_mean, train_x_cov)
likelihood_y_uniform = likelihood(test_xy_126, train_y_mean, train_y_cov)

posterior_x_uniform = likelihood_x_uniform * prior_x_uniform
posterior_y_uniform = likelihood_y_uniform * prior_y_uniform

classification_uniform = np.where(posterior_x_uniform > posterior_y_uniform, 1, 2)
accuracy_xy_126_uniform = np.mean(classification_uniform == test_xy_126_label)

print(f"(b) Accuracy on tst_xy_126 with uniform prior: {accuracy_xy_126_uniform*100:.2f}%")

# -------------------------
# (c) Classify tst_xy_126 with non-uniform prior (0.9 vs 0.1)
# -------------------------

prior_x_non_uniform = 0.9
prior_y_non_uniform = 0.1

likelihood_x_non_uniform = likelihood(test_xy_126, train_x_mean, train_x_cov)
likelihood_y_non_uniform = likelihood(test_xy_126, train_y_mean, train_y_cov)

posterior_x_non_uniform = likelihood_x_non_uniform * prior_x_non_uniform
posterior_y_non_uniform = likelihood_y_non_uniform * prior_y_non_uniform

classification_non_uniform = np.where(posterior_x_non_uniform > posterior_y_non_uniform, 1, 2)
accuracy_xy_126_non_uniform = np.mean(classification_non_uniform == test_xy_126_label)

print(f"(c) Accuracy on tst_xy_126 with non-uniform prior (0.9/0.1): {accuracy_xy_126_non_uniform*100:.2f}%")

# Improvement compared to uniform
improvement = (accuracy_xy_126_non_uniform - accuracy_xy_126_uniform) * 100
print(f"Improvement over uniform prior: {improvement:.2f}%")
