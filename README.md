# SVM Regression House Price Prediction App

A Machine Learning web application built using **Streamlit** and **Support Vector Regression (SVR)** to predict house prices based on housing features.

## Live Demo

https://svm-regression-model.streamlit.app/

---

# Project Overview

This project demonstrates a complete **Machine Learning Regression workflow** including:

- Data Collection
- Data Cleaning
- Feature Scaling
- Model Training
- Model Evaluation
- Model Deployment using Streamlit

The application predicts estimated house prices using housing-related input features.

Support Vector Machines are widely used for both classification and regression tasks because of their ability to handle linear and non-linear data effectively. :contentReference[oaicite:0]{index=0}

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

---

# Machine Learning Algorithm

## Support Vector Regression (SVR)

Support Vector Regression is a supervised machine learning algorithm used for predicting continuous numerical values.

In this project:
- Input → House Features
- Output → Predicted House Price

The model learns relationships between housing features and prices to make accurate predictions.

---

# Dataset Information

The dataset contains:

- 1000 rows
- 8 columns

## Features

| Feature | Description |
|---|---|
| Area | House area in square feet |
| Bedrooms | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Floors | Number of floors |
| Parking | Parking spaces |
| HouseAge | Age of house |
| DistanceFromCity | Distance from city center |
| Price | Target variable |

---

# Project Workflow

## 1. Data Preprocessing

- Removed duplicate rows
- Checked missing values
- Applied feature scaling using `StandardScaler`

Feature scaling is especially important for SVM/SVR models. :contentReference[oaicite:1]{index=1}

---

## 2. Train-Test Split

Dataset split:
- 80% Training
- 20% Testing

---

## 3. Model Training

Used:

```python
from sklearn.svm import SVR

model = SVR(kernel='rbf')
