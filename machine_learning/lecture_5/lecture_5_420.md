# Assignment 5 - Linear Discrimination

### Exercise
Perform classification for the entire MNIST dataset (the notebook uses the mnist_all.mat file attached to exercise 4) based on the algorithms introduced: Use LDA for dimensionality reduction to 2 or 9 dimensions, classify the dimension-reduced data and compare this classification performance with that of using PCA. 

### For 2 components
- Scatter Plots
![image](lda_2_components_scatter.png)
![image](pca_2_components_scatter.png)

- Confusion Matrix for 2 components
![image](confusion_matrix_lda_2_components.png)
![image](confusion_matrix_pca_2_components.png)

- Accuracy:
  - PCA accuracy: 0.465
  - LDA accuracy: 0.5647

### For 9 components

- Confusion Matrix for 2 components
![image](confusion_matrix_lda_9_components.png)
![image](confusion_matrix_pca_9_components.png)

- Accuracy:
  - PCA accuracy: 0.8779
  - LDA accuracy: 0.895