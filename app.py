# Import required libraries
import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Loading the pre-trained XGBoost model and encoders from the Saved Files
xgb_model = joblib.load('xgb_model.joblib')

#loading the pickled encoders
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Title of the Project
st.title("🙎‍♂️Customer Churn Prediction")

# About the Application.
st.markdown("""
    This app predicts customer churn based on various customer attributes. 
    Please enter the required information below to get the Customer churn prediction.
""")

# User input fields for the dataset columns
gender = st.selectbox("Gender", ['Male', 'Female'])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['Yes', 'No'])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=200, value=50)
total_charges = st.number_input("Total Charges", min_value=0, value=500)

#input data as a DataFrame, including all necessary columns
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Preprocess the inputs
def preprocess_input(data):
    # Encoding categorical columns using saved label encoders
    for column, encoder in encoders.items():
        if column in data.columns:  #the column are exists in the input data
            data[column] = encoder.transform(data[column])
    return data

#preprocessing the columns
input_data = preprocess_input(input_data)

# Display a button to make the prediction
if st.button("Predict Churn"):
    # Makeing the predictions using the pre-trained model
    prediction = xgb_model.predict(input_data)
    
    #The prediction result
    if prediction[0] == 1:
        st.write("**Prediction:** Customer will churn.")
    else:
        st.write("**Prediction:** Customer will not churn.")
