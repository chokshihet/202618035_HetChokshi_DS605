# Face Recognition Using PCA and Eigenfaces

**DS605: Fundamentals of Machine Learning**  
**Lab Assignment 7 – Image Representation, PCA, Eigenfaces and Face Recognition**

## 1. Project Overview

This project implements a face recognition system using **Principal Component Analysis (PCA)**, Eigenfaces, and traditional machine learning algorithms.

The objective is to represent facial images numerically, reduce their dimensionality using PCA, reconstruct faces from reduced representations, and perform face recognition using similarity-based matching and machine learning classifiers.

PCA is implemented **from scratch using NumPy linear algebra operations**, without using Scikit-learn's built-in PCA implementation.

The project compares face recognition performance using original pixel features and PCA-reduced features.

## 2. Dataset

**Dataset:** AT&T Database of Faces (ORL)

**Source:** https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

The dataset contains grayscale facial images of 40 different individuals.

| Property | Description |
|---|---|
| Total Images | 400 |
| Total Subjects | 40 |
| Images per Subject | 10 |
| Image Dimensions | 112 × 92 |
| Original Features | 10,304 |
| Image Type | Grayscale |
| Training Images | 320 |
| Testing Images | 80 |
| Train-Test Split | 80:20 |
| Random State | 42 |

Each image is represented as a matrix of pixel intensity values and flattened into a one-dimensional feature vector.

A stratified train-test split ensures that every subject is represented in both sets.

## 3. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- Scikit-learn
- Jupyter Notebook

NumPy is used to implement PCA through covariance eigendecomposition and Singular Value Decomposition (SVD).

Scikit-learn is used for dataset splitting, traditional machine learning classifiers, hyperparameter tuning, and evaluation.

## 4. Project Methodology

### Part A – Image Representation

The first stage converts facial images into numerical representations suitable for machine learning.

**Steps performed:**

1. Loaded grayscale face images from the dataset ZIP file.
2. Inspected image dimensions and pixel intensity values.
3. Converted images into NumPy arrays.
4. Flattened each 112 × 92 image into a vector containing 10,304 features.
5. Created subject labels.
6. Performed a stratified 80:20 train-test split.

**Output:**

- Training data: `(320, 10304)`
- Testing data: `(80, 10304)`
- Total subjects: 40

### Part B – PCA, Eigenfaces and Compression

Principal Component Analysis was implemented from scratch to reduce the dimensionality of the facial images.

**Steps performed:**

1. Calculated the mean face from the training dataset.
2. Centered training and testing images using the training mean.
3. Implemented PCA using covariance matrix eigendecomposition.
4. Implemented PCA using Singular Value Decomposition.
5. Compared eigenvalues obtained from both approaches.
6. Calculated cumulative explained variance.
7. Determined the minimum number of components required to preserve at least 90% variance.
8. Visualized the principal components as Eigenfaces.
9. Reconstructed facial images using different numbers of components.
10. Evaluated reconstruction quality using Mean Squared Error.

**PCA Results:**

| Metric | Result |
|---|---|
| Original Features | 10,304 |
| Selected PCA Components | 98 |
| Explained Variance | 90.06% |
| Feature Dimensionality Reduction | 99.05% |
| Covariance and SVD Eigenvalues | Matched |

**Reconstruction Results:**

| PCA Components | Explained Variance | Reconstruction MSE |
|---|---|---|
| 10 | 60.40% | 664.88 |
| 30 | 76.16% | 474.28 |
| 60 | 84.70% | 380.93 |
| 98 | 90.06% | 334.99 |

**Observation:**

Reconstruction error decreases as more principal components are retained. Higher component counts preserve more facial information, resulting in better reconstruction quality.

### Part C – Face Recognition in PCA Space

Face recognition was implemented using nearest-neighbor matching based on Euclidean distance.

Each test face was compared with the stored training faces, and the identity of the nearest training face was assigned as its predicted subject.

Two approaches were compared:

- Recognition using original pixel features.
- Recognition using PCA-reduced features.

**Recognition Results:**

| Method | Features | Accuracy |
|---|---|---|
| Raw Pixel Matching | 10,304 | 96.25% |
| PCA-Based Matching | 98 | 96.25% |

Both approaches correctly classified 77 out of 80 test images.

**Effect of PCA Components:**

| Components | Recognition Accuracy |
|---|---|
| 10 | 95.00% |
| 20 | 96.25% |
| 30 | 96.25% |
| 50 | 98.75% |
| 60 | 97.50% |
| 80 | 97.50% |
| 98 | 96.25% |
| 120 | 96.25% |
| 150 | 96.25% |

**Observations:**

- PCA substantially reduced the number of features without reducing recognition accuracy at the 98-component setting.
- The highest observed nearest-neighbor accuracy was 98.75%, achieved with 50 PCA components.
- Increasing the number of principal components does not always improve recognition accuracy.
- The component-count comparison was exploratory and evaluated on the test set.

### Part D – Traditional Machine Learning

Three traditional machine learning classifiers were trained using PCA-reduced facial features.

**Algorithms implemented:**

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Support Vector Machine (SVM)

Hyperparameter tuning was performed using GridSearchCV with four-fold stratified cross-validation.

**Baseline Model Results:**

| Model | Accuracy |
|---|---|
| KNN | 96.25% |
| Logistic Regression | 93.75% |
| SVM | 98.75% |

**Hyperparameter Tuning Results:**

| Model | Best Parameters | CV Accuracy |
|---|---|---|
| KNN | n_neighbors=1, weights=uniform | 95.94% |
| Logistic Regression | C=1 | 92.50% |
| SVM | C=10, gamma=0.001 | 91.88% |

**Baseline vs Tuned Model Performance:**

| Model | Baseline Accuracy | Tuned Accuracy |
|---|---|---|
| KNN | 96.25% | 96.25% |
| Logistic Regression | 93.75% | 93.75% |
| SVM | 98.75% | 95.00% |

**Observations:**

- Baseline SVM achieved the highest test accuracy of 98.75%.
- KNN achieved 96.25% accuracy.
- Logistic Regression achieved 93.75% accuracy.
- Hyperparameter tuning did not improve test accuracy in these experiments.
- PCA enabled effective face recognition using a substantially reduced feature representation.

**Evaluation note:** PCA was fitted on the full training set before cross-validation, so the reported cross-validation scores may contain preprocessing leakage. Additionally, multiple configurations were evaluated on the test set. These results are reported as experimental comparisons rather than unbiased generalization estimates.

## 5. Project Workflow

```text
ORL Face Dataset
       |
       v
Load Grayscale Images
       |
       v
Convert Images to NumPy Arrays
       |
       v
Flatten Images into Feature Vectors
       |
       v
Stratified Train-Test Split
       |
       v
Calculate Mean Face
       |
       v
Mean Centering
       |
       v
PCA Using NumPy
       |
       +---- Covariance Eigendecomposition
       |
       +---- Singular Value Decomposition
       |
       v
Explained Variance Analysis
       |
       v
Eigenfaces Visualization
       |
       v
Reduced PCA Representation
       |
       +---- Face Reconstruction
       |
       +---- Nearest Neighbor Recognition
       |
       +---- KNN / Logistic Regression / SVM
       |
       v
Performance Evaluation
```

## 6. Project Structure

```text
DS605_Lab7_Face_Recognition/
|
|-- Lab7_Face_Recognition.ipynb
|-- att_faces.zip
|-- README.md
|
|-- outputs/
    |-- model_comparison.png
    |-- eigenfaces.png
    |-- reconstruction.png
    |-- recognition_matches.png
    |-- model_comparison.csv
    |-- pca_accuracy_comparison.csv
    |-- reconstruction_results.csv
```

The dataset ZIP filename may vary. The notebook automatically detects a ZIP file in its working directory.

The listed image files should be generated and saved before final submission.


## 7. Key Findings

1. PCA successfully reduced the original 10,304-dimensional feature space to 98 components while preserving approximately 90% of the training variance.

2. Raw-pixel and PCA-based nearest-neighbor recognition both achieved 96.25% accuracy.

3. Using 50 PCA components resulted in the highest observed nearest-neighbor recognition accuracy of 98.75%.

4. Face reconstruction quality improved as the number of retained principal components increased.

5. The baseline SVM classifier achieved the highest observed classification accuracy of 98.75%.

6. Hyperparameter tuning did not necessarily improve test performance.

7. Dimensionality reduction can preserve useful facial identity information while substantially reducing the number of features used for recognition.

## 9. Conclusion

This project demonstrates the application of Principal Component Analysis and Eigenfaces for face recognition.

By implementing PCA from scratch using NumPy, the project explores the mathematical foundations of dimensionality reduction, including covariance eigendecomposition, Singular Value Decomposition, explained variance, and reconstruction.

The experimental results show that PCA can reduce facial image dimensionality significantly while maintaining strong recognition accuracy.

The baseline SVM achieved 98.75% test accuracy, while PCA-based nearest-neighbor recognition achieved 96.25% accuracy using only 98 components.

Overall, the project demonstrates how dimensionality reduction and traditional machine learning techniques can be combined to build an effective face recognition system.



**Course:** DS605 – Fundamentals of Machine Learning  
**Assignment:** Lab Assignment 7  
**Topic:** Image Representation, PCA, Eigenfaces and Face Recognition

