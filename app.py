import streamlit as st
import pandas as pd
import joblib

st.title("Multi-Disease Early Prediction System")

# Sidebar: select disease
disease = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Chronic Kidney Disease", "Parkinson's Disease", "Breast Cancer"]
)

# ------------------- Diabetes -------------------
if disease == "Diabetes":
    st.subheader("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 100, 30)

    features = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                            columns=["Pregnancies","Glucose","BloodPressure","SkinThickness",
                                     "Insulin","BMI","DiabetesPedigreeFunction","Age"])
    
    model = joblib.load("models/diabetes_model.pkl")
    scaler = joblib.load("scalers/diabetes_scaler.pkl")
    features_scaled = scaler.transform(features)

# ------------------- Heart Disease -------------------
elif disease == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex (1=Male,0=Female)", [1,0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl (1=Yes,0=No)", [1,0])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes,0=No)", [1,0])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment (0-2)", [0,1,2])
    ca = st.selectbox("Major Vessels Colored (0-3)", [0,1,2,3])
    thal = st.selectbox("Thalassemia (1=Normal,2=Fixed,3=Reversible)", [1,2,3])

    features = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                            columns=["age","sex","cp","trestbps","chol","fbs","restecg",
                                     "thalach","exang","oldpeak","slope","ca","thal"])
    
    model = joblib.load("models/heart_model.pkl")
    scaler = joblib.load("scalers/heart_scaler.pkl")
    features_scaled = scaler.transform(features)

# ------------------- Chronic Kidney Disease -------------------
elif disease == "Chronic Kidney Disease":
    st.subheader("CKD Prediction")
    bp = st.number_input("Blood Pressure", 0, 200, 80)
    sg = st.number_input("Specific Gravity (1.005-1.025)", 1.005, 1.025, 1.010)
    al = st.number_input("Albumin (0-5)", 0, 5, 0)
    su = st.number_input("Sugar (0-5)", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random", 0, 500, 100)
    bu = st.number_input("Blood Urea", 0, 200, 20)
    sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.0)
    sod = st.number_input("Sodium", 0, 200, 135)
    pot = st.number_input("Potassium", 0, 10, 4)
    hemo = st.number_input("Hemoglobin", 0.0, 20.0, 13.5)

    features = pd.DataFrame([[bp, sg, al, su, bgr, bu, sc, sod, pot, hemo]],
                            columns=["bp","sg","al","su","bgr","bu","sc","sod","pot","hemo"])
    
    model = joblib.load("models/ckd_model.pkl")
    scaler = joblib.load("scalers/ckd_scaler.pkl")
    features_scaled = scaler.transform(features)

# ------------------- Parkinson's Disease -------------------
elif disease == "Parkinson's Disease":
    st.subheader("Parkinson's Prediction")
    # Example: 5 input features (you can add more from dataset)
    mdvp_fo = st.number_input("MDVP:Fo(Hz)", 100.0, 300.0, 120.0)
    mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 300.0, 120.0)
    mdvp_flo = st.number_input("MDVP:Flo(Hz)", 50.0, 200.0, 70.0)
    jitter_percent = st.number_input("Jitter (%)", 0.0, 1.0, 0.1)
    shimmer = st.number_input("Shimmer", 0.0, 1.0, 0.1)

    features = pd.DataFrame([[mdvp_fo, mdvp_fhi, mdvp_flo, jitter_percent, shimmer]],
                            columns=["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","Jitter(%)","Shimmer"])
    
    model = joblib.load("models/parkinson_model.pkl")
    scaler = joblib.load("scalers/parkinson_scaler.pkl")
    features_scaled = scaler.transform(features)

# ------------------- Breast Cancer -------------------
elif disease == "Breast Cancer":
    st.subheader("Breast Cancer Prediction")
    radius_mean = st.number_input("Radius Mean", 0.0, 30.0, 14.0)
    texture_mean = st.number_input("Texture Mean", 0.0, 40.0, 20.0)
    perimeter_mean = st.number_input("Perimeter Mean", 0.0, 200.0, 90.0)
    area_mean = st.number_input("Area Mean", 0.0, 2500.0, 600.0)
    smoothness_mean = st.number_input("Smoothness Mean", 0.0, 0.2, 0.1)

    features = pd.DataFrame([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]],
                            columns=["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"])
    
    model = joblib.load("models/breast_model.pkl")
    scaler = joblib.load("scalers/breast_scaler.pkl")
    features_scaled = scaler.transform(features)

# ------------------- Predict Button -------------------
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of {disease}. Probability: {proba:.2f}")
    else:
        st.success(f"✅ No {disease} detected. Probability: {proba:.2f}")
