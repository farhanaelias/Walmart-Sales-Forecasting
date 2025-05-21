
# 🛒 Walmart Sales Forecasting using Machine Learning & XAI

A comparative analysis of multiple machine learning models to predict Walmart's weekly sales using historical data. This project evaluates 12 different ML algorithms, identifies the best-performing model (XGBoost).

## 📌 Project Overview

This study focuses on:

- Comparing 12 ML models to predict Walmart's sales.
 
- Evaluating models based on metrics like Accuracy, Precision, Recall, F1-score, MSE, and MAE.
 
- Highlighting the best model (XGBoost, with 98% accuracy) for real-world deployment.

## 📊 Models Compared

- Linear Regression
  
- Ridge Regression
  
- LinearSVR
  
- Support Vector Regression (SVR)
  
- Decision Tree
  
- Random Forest
  
- K-Nearest Neighbors (KNN)
  
- Gradient Boosting
  
- AdaBoost
  
- XGBoost ⭐ (Best)
  
- LightGBM
  
- Multi-Layer Perceptron (MLP)
  

## 🧠 Best Model: XGBoost

- **Accuracy**: 98%
  
- **Precision/Recall**: High on both safe and risky sales classes
  
- **MSE/MAE**: Low error
  
- **Conclusion**: Outperforms other models across all evaluation metrics
  

## 🧰 Technologies Used

- **Python**
  
- **Pandas, NumPy, Scikit-learn, XGBoost, LightGBM**
  
- **Matplotlib, Seaborn (for visualization)**
  

## 📁 Dataset

- Source: [Kaggle Walmart Dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
 
- Features: Weekly sales, store ID, holidays, temperature, fuel prices, CPI, unemployment

## 🧪 Evaluation Metrics

| Metric        | Description                                      |
|---------------|--------------------------------------------------|
| Accuracy      | Correct predictions over total predictions       |
| Precision     | TP / (TP + FP)                                   |
| Recall        | TP / (TP + FN)                                   |
| F1 Score      | Harmonic mean of precision and recall            |
| MSE/MAE       | Measures prediction error                        |



## 🔄 Workflow

1. Data Cleaning & Preprocessing
 
2. Exploratory Data Analysis (EDA)
 
3. Feature Scaling & Encoding
 
4. Model Training & Evaluation
 
5. Best Model Selection


## 📈 Results Summary

- XGBoost, LightGBM, and Random Forest are top performers.
 
- XGBoost leads with best metrics and lowest error.
 
- LinearSVR performs poorly with only 49% accuracy.

## 🚀 Future Work

- Add more feature engineering (seasonality, promotions)
 
- Integrate deep learning models (e.g., LSTM, CNN)


