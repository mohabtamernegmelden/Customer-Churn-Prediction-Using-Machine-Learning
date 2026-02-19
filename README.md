# Customer Churn Prediction Using Machine Learning
1️⃣ Introduction

Customer churn refers to customers who stop using a company’s service.
Telecom companies lose significant revenue due to customer churn.

This project aims to build a Machine Learning model to predict whether a customer will churn based on demographic and service-related features.

2️⃣ Objective

Predict customer churn (Yes / No)

Identify high-risk customers

Help business reduce customer loss

Deploy model using Streamlit

3️⃣ Dataset Description

The dataset contains telecom customer information including:

Feature	Description
gender	Male/Female
SeniorCitizen	Whether customer is senior 
Partner	Has partner
Dependents	Has dependents
tenure	Months with company
PhoneService	Has phone service
MultipleLines	Multiple phone lines
InternetService	DSL / Fiber / No
OnlineSecurity	Has security
OnlineBackup	Has backup
DeviceProtection	Has protection
TechSupport	Has support
StreamingTV	Streaming TV service
StreamingMovies	Streaming movies
Contract	Contract type
PaperlessBilling	Uses paperless billing
PaymentMethod	Payment type
MonthlyCharges	Monthly bill
TotalCharges	Total paid
Churn	Target variable
4️⃣ Data Preprocessing

Handled missing values

Converted categorical variables using Label Encoding

Converted numeric columns to float

Split data into training and testing sets

5️⃣ Models Used

We trained multiple models:

Logistic Regression

Decision Tree

Random Forest

XGBoost

Final selected model: (Write your best model here)

6️⃣ Model Evaluation

Evaluation Metrics Used:

Accuracy

Precision

Recall

F1 Score

ROC-AUC Score

Example Results:

Model	Accuracy
Logistic Regression	79%
Decision Tree	78%
Random Forest	83.7%
XGBoost	83.1%


7️⃣ Deployment

The model was deployed using Streamlit.

Features:

User-friendly interface

Real-time prediction

Probability score display

Risk level indication

8️⃣ Results

The model successfully predicts churn probability.

Customers with Month-to-Month contracts and Fiber optic services showed higher churn.

Long tenure customers showed lower churn probability.

9️⃣ Conclusion

The project demonstrates how Machine Learning can help telecom companies reduce churn by identifying high-risk customers.

Future improvements:

Use advanced feature engineering

Use deep learning models

Connect to real database

Deploy on cloud with API

🔟 Tools & Technologies

Python

Pandas

NumPy

Scikit-learn

XGBoost

Streamlit

Matplotlib / Seaborn