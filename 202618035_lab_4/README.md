#  NYC Airbnb Price Prediction

**DS605: Fundamentals of Machine Learning — Lab Assignment 4**

An end-to-end machine learning project for predicting the nightly price of Airbnb listings in New York City.

> **Live Application:** Deployment link will be added after Streamlit Community Cloud deployment.

---

##  Project Overview

This project develops a complete machine learning workflow using the New York City Airbnb Open Data (`AB_NYC_2019.csv`).

The project includes:

- Data cleaning and preprocessing
- Missing value handling
- Outlier detection and removal
- Exploratory Data Analysis (EDA)
- Feature selection
- Categorical encoding
- Numerical feature scaling
- Training multiple regression models
- Model evaluation and comparison
- Hyperparameter tuning
- Overfitting analysis
- Final model selection
- Model pipeline serialization
- Streamlit web application

---

##  Dataset

**Dataset:** New York City Airbnb Open Data (`AB_NYC_2019.csv`)

The original dataset contains:

- **48,895 listings**
- **16 columns**

The target variable is:

- `price` — nightly Airbnb listing price in US dollars.

After data cleaning and price outlier removal, **45,912 listings** were retained.

---

##  Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary identifier and text columns
- Handled missing values
- Replaced missing `reviews_per_month` values with 0
- Checked and removed duplicate records
- Removed zero-price listings
- Detected price outliers using the IQR method
- One-Hot Encoded categorical variables
- Standardized numerical variables
- Split the dataset into 80% training and 20% testing data

The preprocessing steps were incorporated into a Scikit-learn pipeline to ensure consistent transformations during training and prediction.

---

##  Exploratory Data Analysis

EDA was performed to study:

- Airbnb listings by neighbourhood group
- Average price by neighbourhood group
- Average price by room type
- Top neighbourhoods by average price
- Correlation between numerical variables
- Geographic distribution of Airbnb listings
- Distribution of the target variable

Important observations include:

- Airbnb prices vary considerably across locations.
- Room type has a strong influence on nightly price.
- Entire homes/apartments generally cost more than private or shared rooms.
- Both geographical and listing-related characteristics contribute to price prediction.

---

##  Machine Learning Models

Four regression models were trained and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

### Initial Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 31.54 | 43.89 | 0.575 |
| Gradient Boosting | 32.49 | 44.56 | 0.562 |
| Linear Regression | 34.08 | 46.43 | 0.524 |
| Decision Tree | 43.19 | 61.93 | 0.153 |

Random Forest achieved the strongest initial performance.

---

##  Hyperparameter Tuning

Random Forest was further optimized using `RandomizedSearchCV` with cross-validation.

The tuned Random Forest achieved:

| Metric | Result |
|---|---:|
| MAE | **31.17** |
| RMSE | **43.18** |
| R² Score | **0.5883** |

The tuned model improved upon the original Random Forest and was selected as the final model.

---

##  Final Model

The final model is a **Tuned Random Forest Regressor**.

The complete preprocessing and prediction pipeline is stored in:

```text
model.pkl
```

This allows the same preprocessing operations used during training to be automatically applied to new Airbnb listing inputs.

---

##  Streamlit Web Application

A Streamlit application was developed to provide a simple interface for using the trained model.

Users can provide:

- Neighbourhood group
- Neighbourhood
- Room type
- Latitude
- Longitude
- Minimum nights
- Number of reviews
- Reviews per month
- Host listing count
- Availability

The application processes these values through the saved machine learning pipeline and displays the estimated nightly Airbnb price.

### Application Screenshot

![NYC Airbnb Price Predictor](screenshots/airbnb_prediction_app.png)

---

##  Project Structure

```text
202618035_lab_4/
│
├── data/
│   └── AB_NYC_2019.csv
│
├── screenshots/
│   └── airbnb_prediction_app.png
│
├── app.py
├── code.ipynb
├── model.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── New_York_City_.png
```

---

## ▶ How to Run the Application

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2. Open the project directory

```bash
cd 202618035_lab_4
```

### 3. Install the required libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will normally be available locally at:

```text
http://localhost:8501
```

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

##  Limitations

- The dataset represents Airbnb listings from 2019 and does not reflect current market prices.
- Factors such as amenities, property photographs, seasonal demand, special events, and property quality are not included.
- The final model explains approximately 58.8% of the variation in test-set prices.
- Predictions should therefore be interpreted as estimates rather than exact market prices.

---

##  Conclusion

This project demonstrates a complete end-to-end machine learning workflow for Airbnb price prediction.

The Tuned Random Forest model achieved the best overall performance with an R² score of approximately **0.5883** on unseen test data. The final preprocessing and prediction pipeline was successfully integrated into a Streamlit application, allowing users to obtain price estimates for new Airbnb listings.

---

##  Author

**Het Chokshi**

DS605 — Fundamentals of Machine Learning