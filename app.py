import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model and scaler
# Ensure you have saved the model and scaler as 'model.pkl' and 'scaler.pkl' after training
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define the Streamlit app
st.title("Heart Disease Prediction App")
st.write("""
This app predicts the likelihood of heart disease based on user-provided health metrics.
""")

# Sidebar inputs
st.sidebar.header("User Input Features")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", options=[
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal Pain",
    "Asymptomatic"
])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
restecg = st.sidebar.selectbox("Resting ECG Results", options=[
    "Normal",
    "ST-T wave abnormality",
    "Left ventricular hypertrophy"
])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=["No", "Yes"])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[
    "Upsloping",
    "Flat",
    "Downsloping"
])
ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalium Stress Test Result", options=[
    "Normal",
    "Fixed Defect",
    "Reversible Defect",
    "Not Described"
])

# Map categorical inputs to numerical values
sex = 1 if sex == "Female" else 0
cp = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
fbs = 1 if fbs == "Yes" else 0
restecg = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}[restecg]
exang = 1 if exang == "Yes" else 0
slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
thal = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2, "Not Described": 3}[thal]

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Scale numerical features
scaled_data = scaler.transform(input_data)

# Predict the likelihood of heart disease
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)

# Display prediction results
st.subheader("Prediction Results")
if prediction[0] == 1:
    st.write("### The model predicts that the person **has heart disease**.")
else:
    st.write("### The model predicts that the person **does not have heart disease**.")

st.write(f"### Confidence: {probability[0][prediction][0] * 100:.2f}%")

