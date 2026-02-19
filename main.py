import numpy as np
import pandas as pd
import pickle
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / 'models' / 'Customer_Churn_Model.pkl'

with open(path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['features_names']
encoder_col = model_data['encoder_col']
def prediction(input_data):

    input_df = pd.DataFrame([input_data])

    for column in encoder_col.keys():
        if column in input_df.columns:
            try:
                input_df[column] = encoder_col[column].transform(input_df[column])
            except ValueError:
                st.error(f"Unknown category detected in {column}")
                return

    input_df = input_df.reindex(columns=feature_names)

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numeric_cols] = input_df[numeric_cols].astype(float)

    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error("🚨 Customer WILL Churn")
    else:
        st.success("✅ Customer will NOT Churn")

    st.write(f"### 📊 Churn Probability: {pred_proba:.2%}")

    if pred_proba >= 0.7:
        st.warning("⚠️ High Risk Customer")
    elif pred_proba >= 0.4:
        st.info("Moderate Risk Customer")
    else:
        st.success("Low Risk Customer")

def main():
    st.title("📊 Customer Churn Prediction")

    gender = st.selectbox('Gender', ('Female', 'Male'))

    SeniorCitizen = st.selectbox('Senior Citizen', ('No', 'Yes'))
    SeniorCitizen = 1 if SeniorCitizen == 'Yes' else 0

    Partner = st.selectbox('Partner', ('No', 'Yes'))
    Dependents = st.selectbox('Dependents', ('No', 'Yes'))

    tenure = st.number_input('Tenure (Months)', min_value=0)

    PhoneService = st.selectbox('Phone Service', ('No', 'Yes'))

    MultipleLines = st.selectbox(
        'Multiple Lines',
        ('No', 'Yes', 'No phone service')
    )

    InternetService = st.selectbox(
        'Internet Service',
        ('DSL', 'Fiber optic', 'No')
    )

    OnlineSecurity = st.selectbox(
        'Online Security',
        ('No', 'Yes', 'No internet service')
    )

    OnlineBackup = st.selectbox(
        'Online Backup',
        ('No', 'Yes', 'No internet service')
    )

    DeviceProtection = st.selectbox(
        'Device Protection',
        ('No', 'Yes', 'No internet service')
    )

    TechSupport = st.selectbox(
        'Tech Support',
        ('No', 'Yes', 'No internet service')
    )

    StreamingTV = st.selectbox(
        'Streaming TV',
        ('No', 'Yes', 'No internet service')
    )

    StreamingMovies = st.selectbox(
        'Streaming Movies',
        ('No', 'Yes', 'No internet service')
    )

    Contract = st.selectbox(
        'Contract Type',
        ('Month-to-month', 'One year', 'Two year')
    )

    PaperlessBilling = st.selectbox(
        'Paperless Billing',
        ('No', 'Yes')
    )

    PaymentMethod = st.selectbox(
        'Payment Method',
        (
            'Electronic check',
            'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        )
    )

    MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0)
    TotalCharges = st.number_input('Total Charges', min_value=0.0)

    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    if st.button('Predict'):
        prediction(input_data)

if __name__=='__main__':
    main()



    