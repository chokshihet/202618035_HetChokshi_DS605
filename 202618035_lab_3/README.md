# Lab 3: Scikit-learn Preprocessing and Model Evaluation

**Course:** Fundamentals of Machine Learning (DS605)  
**Name:** Het Chokshi  
**Student ID:** 202618035  
**Dataset:** [Hotel Booking Demand (Kaggle)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

## Project Overview
This repository contains a comprehensive Machine Learning pipeline in Python using Scikit-learn. The goal is to predict hotel booking cancellations (`is_canceled`) by comparing two preprocessing strategies and two classification algorithms (Logistic Regression and Decision Tree).

---

## Preprocessing Choices

1. **Target Leakage & High Missingness Removal:**
   - **`reservation_status` & `reservation_status_date`:** Removed because they directly reveal the final outcome of the booking, which leads to target leakage.
   - **`company`:** Removed due to an excessively high proportion of missing values (>90%).

2. **Outlier & Anomaly Filtering:**
   - Removed corrupt/extreme price entries (`adr < 0` or `adr >= 5000`).
   - Removed invalid booking rows where `adults == 0` or unrealistically high (`adults > 10`).

3. **Preprocessing Pipelines:**
   - **Shared Categorical Pipeline:** Imputed missing values using `SimpleImputer(strategy='most_frequent')` and encoded categorical variables using `OneHotEncoder(handle_unknown='ignore')`.
   - **Pipeline A (StandardScaler):** Numerical features imputed using `KNNImputer(n_neighbors=5)` and scaled using `StandardScaler`.
   - **Pipeline B (MinMaxScaler):** Numerical features imputed using `KNNImputer(n_neighbors=5)` and scaled using `MinMaxScaler`.

---

## Final Performance Comparison

| Model & Pipeline | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | Overfit Gap (Train - Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Pipeline A - Standard)** | 0.8187 | 0.8182 | 0.8126 | 0.6625 | 0.7299 | 0.0004 |
| **Logistic Regression (Pipeline B - MinMax)** | 0.8151 | 0.8152 | 0.8117 | 0.6529 | 0.7237 | -0.0000 |
| **Decision Tree (Pipeline A - Standard)** | 0.9962 | **0.8616** | 0.8117 | **0.8159** | **0.8138** | 0.1346 |
| **Decision Tree (Pipeline B - MinMax)** | 0.9962 | 0.8611 | 0.8112 | 0.8150 | 0.8131 | 0.1351 |

---

## Key Observations & Findings

1. **Best Overall Combination:** 
   **Decision Tree with Pipeline A (StandardScaler)** achieved the highest test performance across the board, with a **Test Accuracy of 86.16%** and an **F1-Score of 0.8138**.

2. **Impact of Feature Scaling on Logistic Regression:** 
   StandardScaler performed slightly better than MinMaxScaler (81.82% vs 81.52% test accuracy). Because MinMaxScaler squashes all features strictly into a $[0, 1]$ range, skewed features with high values (like `adr` or `lead_time`) compress normal values into a tiny range. StandardScaler centers data around its mean, handling variance far better.

3. **Impact of Feature Scaling on Decision Trees:** 
   Scaling had **no impact** on the Decision Tree model (both pipelines yielded ~86.1% test accuracy). Tree algorithms evaluate threshold splits on individual features independently, making them naturally scale-invariant.

4. **Overfitting Analysis:** 
   - **Decision Tree:** Displayed severe overfitting with a **~13.5% gap** between training accuracy (99.62%) and test accuracy (86.16%). This occurs because an unconstrained tree memorizes noisy patterns in the training data.
   - **Logistic Regression:** Showed virtually zero overfitting gap (0.0004), demonstrating strong generalization despite being bounded by linear decision boundaries.

5. **Class Detection (Recall):** 
   Logistic Regression struggled significantly with Recall (0.6625), missing a notable portion of actual cancellations. The Decision Tree handled positive cancellation detection much better (Recall of 0.8159).

---

## Repository Structure
- `202618035_Lab_Assignment-3.ipynb`: Complete runnable notebook covering data cleaning, pipelines, model fitting, and confusion matrix visualizations.
- `cleaned_hotel_bookings.csv`: Cleaned base dataset prior to train-test splitting.
- `README.md`: Lab assignment overview, methodology, and final findings.
