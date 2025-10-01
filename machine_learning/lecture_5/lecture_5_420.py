# ###################################
# Group ID : <420>
# Members : <Niclas Alexander Pedersen, Snorre Johnsen, Mathias Majland Jørgensen, Johan Boelsbjerg Askjær, Rasmus Mellergaard Christensen, Markus Heinrich Toribio>
# Date : <01/10/2025>
# Lecture: <Lecture 5> <Linear Discrimination> (see moodle)
# Dependencies: os, numpy, matplotlib, scikit learn.
# Python version: 3.11.5
# Functionality: This script applies LDA to reduce dimensionality of the dataset, to classify the the dimension-reduced data.
# Further, it compares the classification performance with PCA.
# ###################################

# %%
import numpy as np                              # Numerical computations
from scipy.io import loadmat                    # Load MATLAB .mat files
from scipy.stats import multivariate_normal as norm  # Gaussian distribution
import matplotlib.pyplot as plt                 # Plotting
from sklearn.decomposition import PCA           # Principal Component Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # Linear Discriminant Analysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Confusion matrix tools

# %%
def create_complete_datasets(data_dict):
    '''
    Function for creating complete training and test sets containing
    all classes.
    '''
    trainset = []        # List for training samples
    traintargets = []    # List for training labels
    testset = []         # List for test samples
    testtargets = []     # List for test labels
    
    #For each class (digits 0–9)
    for i in range(10):
        trainset.append(data_dict["train%d"%i])                    # Append training samples for class i
        traintargets.append(np.full(len(data_dict["train%d"%i]),i))# Append labels for training samples
        testset.append(data_dict["test%d"%i])                      # Append test samples for class i
        testtargets.append(np.full(len(data_dict["test%d"%i]),i))  # Append labels for test samples
    
    #Concatenate lists into full datasets
    trainset = np.concatenate(trainset)        # Full training set
    traintargets = np.concatenate(traintargets) # Full training labels
    testset = np.concatenate(testset)          # Full test set
    testtargets = np.concatenate(testtargets)  # Full test labels
    return trainset, traintargets, testset, testtargets  # Return datasets

file = "mnist_all.mat"          # Path to dataset file
data = loadmat(file)            # Load dataset from file

#Complete training and test sets
train_set, train_targets, test_set, test_targets = create_complete_datasets(data)  # Build datasets

# %%
n_components = 9    # Number of components for dimensionality reduction

#PCA
pca = PCA(n_components=n_components)     # Create PCA object
train_pca = pca.fit_transform(train_set) # Fit PCA on training data and transform it
test_pca = pca.transform(test_set)       # Transform test data using trained PCA

#LDA
lda = LDA(n_components=n_components)                 # Create LDA object
train_lda = lda.fit_transform(train_set, train_targets) # Fit LDA on training data with labels and transform it
test_lda = lda.transform(test_set)                   # Transform test data using trained LDA

# %%
# Analyze proportion of Variance. If num_components=2 try to visualize dim. reduced data.
plt.figure()                                         # Create figure
plt.plot(np.cumsum(pca.explained_variance_ratio_))   # Plot cumulative explained variance of PCA
plt.xlabel("Number of components")                   # X-axis label
plt.ylabel("Cumulative explained variance")          # Y-axis label
plt.title("PCA Explained Variance")                  # Plot title

if n_components == 2:                                # If reduced to 2D, plot scatterplots
    plt.figure()
    for i in range(10):
        plt.scatter(train_pca[train_targets==i,0], train_pca[train_targets==i,1], s=5, label=str(i))  # PCA scatter by class
    plt.legend()
    plt.title("PCA (2 components)")

    plt.figure()
    for i in range(10):
        plt.scatter(train_lda[train_targets==i,0], train_lda[train_targets==i,1], s=5, label=str(i))  # LDA scatter by class
    plt.legend()
    plt.title("LDA (2 components)")

# %%
# Estimate Gaussians from PCA/LDA
classes = np.unique(train_targets)   # Unique digit classes (0–9)

means_pca = []  # Store class means in PCA space
covs_pca = []   # Store class covariances in PCA space
for c in classes:
    data_c = train_pca[train_targets==c]       # Get PCA-reduced samples for class c
    means_pca.append(np.mean(data_c, axis=0))  # Compute mean vector
    covs_pca.append(np.cov(data_c, rowvar=False)) # Compute covariance matrix

means_lda = []  # Store class means in LDA space
covs_lda = []   # Store class covariances in LDA space
for c in classes:
    data_c = train_lda[train_targets==c]       # Get LDA-reduced samples for class c
    means_lda.append(np.mean(data_c, axis=0))  # Compute mean vector
    covs_lda.append(np.cov(data_c, rowvar=False)) # Compute covariance matrix

# %%
#Compute predictions
def predict(test_data, means, covs):
    preds = []   # List of predictions
    for x in test_data:   # For each test sample
        scores = [norm(mean=means[c], cov=covs[c]).pdf(x) for c in range(len(means))]  # Likelihood under each Gaussian
        preds.append(np.argmax(scores))  # Choose class with max likelihood
    return np.array(preds)  # Return predicted labels

preds_pca = predict(test_pca, means_pca, covs_pca)  # Predict using PCA-reduced Gaussians
preds_lda = predict(test_lda, means_lda, covs_lda)  # Predict using LDA-reduced Gaussians

#Compute accuracy
acc_pca = np.mean(preds_pca == test_targets)  # PCA classification accuracy
acc_lda = np.mean(preds_lda == test_targets)  # LDA classification accuracy

print("PCA accuracy:", acc_pca)   # Print PCA accuracy
print("LDA accuracy:", acc_lda)   # Print LDA accuracy

# %%
#Compute the confusion matrices for PCA and LDA
cm_pca = confusion_matrix(test_targets, preds_pca)  # Confusion matrix for PCA
cm_lda = confusion_matrix(test_targets, preds_lda)  # Confusion matrix for LDA

#Plot Confusion matrices
disp_pca = ConfusionMatrixDisplay(confusion_matrix=cm_pca)  # Create display for PCA
disp_pca.plot()                                             # Plot PCA confusion matrix
plt.title("Confusion Matrix PCA")                           # Title

disp_lda = ConfusionMatrixDisplay(confusion_matrix=cm_lda) # Create display for LDA
disp_lda.plot()                                            # Plot LDA confusion matrix
plt.title("Confusion Matrix LDA")                          # Title
plt.show()                                                 # Show all plots
