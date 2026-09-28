# DS605 Lab 6 – Linear and Logistic Regression

## 1. Project Overview

This assignment implements and compares **Linear Regression and Logistic Regression using Scikit-learn and from-scratch NumPy implementations**.

The project uses the Garments Worker Productivity dataset to study productivity prediction and target achievement classification.

The assignment is divided into three parts:

- **Part A:** Implement Linear Regression and Logistic Regression using Scikit-learn.
- **Part B:** Implement both algorithms from scratch using NumPy and Pandas.
- **Part C:** Compare the implementations and optimize the manual Logistic Regression model.

The objective is to understand how regression and classification algorithms work mathematically, evaluate their predictive performance, and compare their execution times.

---

## 2. Dataset Description

**Dataset:** Garments Worker Productivity

The dataset contains 1,197 observations describing productivity-related factors in a garment manufacturing environment.

The dataset includes features such as:

| Feature | Description |
|---|---|
| quarter | Quarter of the month |
| department | Manufacturing department |
| day | Day of the week |
| team | Team number |
| targeted_productivity | Target productivity assigned to workers |
| smv | Standard Minute Value |
| wip | Work in progress |
| over_time | Overtime assigned |
| incentive | Financial incentive |
| idle_time | Time during which production was idle |
| idle_men | Number of idle workers |
| no_of_style_change | Number of style changes |
| no_of_workers | Number of workers |
| actual_productivity | Actual productivity achieved |

### Target Variables

**Linear Regression**

The target variable is `actual_productivity`, which represents the productivity achieved by workers.

**Logistic Regression**

A binary target variable called `MeetsTarget` is created:

- `1`: Actual productivity is greater than or equal to targeted productivity.
- `0`: Actual productivity is below targeted productivity.

For Logistic Regression, `actual_productivity` is excluded from the input features to prevent target leakage.

---

## 3. Technologies Used

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook
- Matplotlib (if used for visualization)

The manual implementations use NumPy and Pandas without relying on Scikit-learn for model training, prediction, or metric calculations.

---

## 4. Part A – Scikit-learn Implementation

### Data Preprocessing

The following preprocessing steps were performed:

1. Split the dataset into 80% training data and 20% testing data.
2. Handle missing values using median imputation.
3. Apply one-hot encoding to categorical variables.
4. Standardize numerical features.
5. Use the same training and testing samples for the corresponding manual implementations.

Preprocessing statistics are calculated using training data to prevent data leakage.

### Linear Regression

Scikit-learn's `LinearRegression` was used to predict actual worker productivity.

**Results:**

| Metric | Value |
|---|---:|
| MAE | 0.1088 |
| RMSE | 0.1481 |
| R² Score | 0.1736 |

### Logistic Regression

Scikit-learn's `LogisticRegression` was used to classify whether workers achieved their productivity targets.

**Results:**

| Metric | Value |
|---|---:|
| Accuracy | 0.7500 |
| Precision | 0.7696 |
| Recall | 0.9435 |
| F1 Score | 0.8477 |

---

## 5. Part B – From-Scratch Implementation

Both algorithms were implemented using NumPy and Pandas.

### Linear Regression

Linear Regression was implemented using the pseudo-inverse approach.

The prediction equation is:

\[
\hat{y}=X\beta
\]

The coefficients are calculated using:

\[
\beta=X^{+}y
\]

where \(X^{+}\) represents the Moore–Penrose pseudo-inverse.

The following evaluation metrics were calculated manually:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

**Results:**

| Metric | Value |
|---|---:|
| MAE | 0.1088 |
| RMSE | 0.1481 |
| R² Score | 0.1736 |

The manual implementation produced nearly identical predictive results to Scikit-learn.

### Logistic Regression

Logistic Regression was implemented using gradient descent.

The sigmoid function converts linear outputs into probabilities:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The model parameters are updated using:

\[
w_{\text{new}}=w_{\text{old}}-\alpha\nabla J(w)
\]

where:

- \(w\) represents model weights.
- \(\alpha\) represents the learning rate.
- \(\nabla J(w)\) represents the gradient.

The initial implementation used:

- Learning rate: `0.01`
- Iterations: `5000`
- Classification threshold: `0.5`

The following metrics were calculated manually:

- Accuracy
- Precision
- Recall
- F1 Score

**Results:**

| Metric | Value |
|---|---:|
| Accuracy | 0.7542 |
| Precision | 0.7565 |
| Recall | 0.9831 |
| F1 Score | 0.8550 |

---

## 6. Part C – Comparison and Optimization

### Linear Regression Comparison

| Metric | Scikit-learn | From Scratch |
|---|---:|---:|
| MAE | 0.1088 | 0.1088 |
| RMSE | 0.1481 | 0.1481 |
| R² Score | 0.1736 | 0.1736 |

Both implementations produced nearly identical evaluation metrics.

This demonstrates that the manual pseudo-inverse implementation successfully reproduced the least-squares regression approach.

### Logistic Regression Comparison

| Metric | Scikit-learn | Manual Original | Manual Optimized |
|---|---:|---:|---:|
| Accuracy | 0.7500 | 0.7542 | 0.7625 |
| Precision | 0.7696 | 0.7565 | 0.7778 |
| Recall | 0.9435 | 0.9831 | 0.9492 |
| F1 Score | 0.8477 | 0.8550 | 0.8550 |

### Optimization Method

The manual Logistic Regression implementation was optimized using learning-rate tuning and convergence checking.

The following learning rates were evaluated:

| Learning Rate | Iterations | Training Loss | Accuracy | F1 Score |
|---|---:|---:|---:|---:|
| 0.001 | 10000 | 0.524333 | 0.7458 | 0.8530 |
| 0.005 | 10000 | 0.502449 | 0.7542 | 0.8550 |
| 0.010 | 10000 | 0.495133 | 0.7542 | 0.8521 |
| 0.020 | 10000 | 0.491912 | 0.7625 | 0.8550 |

The learning rate of **0.02** was selected because it produced the lowest training loss among the tested configurations.

### Optimized Model Results

| Metric | Value |
|---|---:|
| Learning Rate | 0.02 |
| Iterations | 10000 |
| Training Loss | 0.491912 |
| Accuracy | 0.7625 |
| Precision | 0.7778 |
| Recall | 0.9492 |
| F1 Score | 0.8550 |

The optimized model achieved higher accuracy and precision than the original manual implementation.

However, recall decreased, while the F1 score remained approximately unchanged.

All learning-rate configurations reached the maximum iteration limit, indicating that the convergence stopping criterion did not trigger.

---

## 7. Execution Time Comparison

The following execution times were recorded during the experiment.

### Logistic Regression

| Metric | Scikit-learn | Manual Original | Manual Optimized |
|---|---:|---:|---:|
| Training Time (seconds) | 0.139415 | 0.383005 | 0.748739 |
| Prediction Time (seconds) | 0.010438 | 0.000361 | 0.000023 |

The optimized manual implementation required more training time because it performed 10,000 gradient-descent iterations.

The manual implementations recorded lower prediction times in this run.

Execution times may vary depending on system specifications, background processes, and runtime conditions. Single-run timing measurements should not be interpreted as universal performance differences.

---

## 8. Key Findings

### Finding 1: Linear Regression Implementations Agree

The manual Linear Regression implementation produced almost identical MAE, RMSE, and R² values compared with Scikit-learn.

This validates the mathematical implementation of the least-squares approach.

### Finding 2: Logistic Regression Performance Depends on Optimization

The manual and Scikit-learn implementations produced slightly different classification results.

These differences can arise from optimization algorithms, regularization, convergence behavior, and solver configurations.

### Finding 3: Learning Rate Affects Model Performance

Increasing the learning rate from 0.001 to 0.02 reduced the final training loss within the tested configurations.

The selected learning rate of 0.02 achieved an accuracy of 76.25%.

### Finding 4: Accuracy and Recall Have Different Interpretations

The original manual Logistic Regression achieved a recall of approximately 98.31%.

The optimized model achieved higher accuracy and precision but lower recall.

This illustrates that improving one evaluation metric does not necessarily improve every other metric.

### Finding 5: Optimization Can Increase Computational Cost

Although the tuned model improved accuracy and precision, its training time increased because more gradient-descent iterations were performed.

Optimization therefore involves trade-offs between predictive performance and computational cost.

---


## 9. Conclusion

This assignment provided practical experience in implementing Linear Regression and Logistic Regression using both Scikit-learn and NumPy.

The from-scratch Linear Regression implementation successfully reproduced the predictive results obtained using Scikit-learn.

The manual Logistic Regression implementation demonstrated the working principles of the sigmoid function, gradient descent, probability prediction, and classification thresholding.

Learning-rate tuning improved the manual Logistic Regression model's accuracy from approximately 75.42% to 76.25% and precision from approximately 75.65% to 77.78%, while recall decreased.

The experiment also demonstrated that additional optimization iterations can increase training time.

Overall, the assignment strengthened understanding of regression mathematics, preprocessing, numerical optimization, evaluation metrics, and the trade-offs between predictive performance and computational efficiency.