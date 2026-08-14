# Data Analysis and Vectorized Computing Assignment

## Student Information
* **Name:** Het Chokshi
* **Student ID:** 202618035
* **Course:** DS605- Fundamentals of Machine Learning
* **Date:** 14 August 2026

---

## Project Overview
This project is divided into two core parts:
1. **Vectorized Numerical Computing & Simulation with NumPy:** Implementing mathematical array operations, linear algebra, normal distributions, and simulating **Pólya’s Urn** to examine path-dependent limiting distributions.
2. **Data Wrangling, Feature Engineering & Visualization with Pandas:** Performing data cleaning, complex querying, aggregation, feature engineering, and statistical visual analysis on the **Titanic dataset**.

---

## Dataset Details
* **Dataset Name:** Titanic - Machine Learning from Disaster (`train.csv`)
* **Source:** Kaggle
* **Target Variable:** `Survived` (0 = No, 1 = Yes)
* **Key Features:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
* **Engineered Features:** 
  * `FamilySize`: Total size of family onboard (`SibSp + Parch + 1`)
  * `IsAlone`: Binary flag indicating solo travel (`1` if `FamilySize == 1`, else `0`)

---

## Requirements & Dependencies
Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas matplotlib seaborn