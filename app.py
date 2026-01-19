import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction App")
st.caption("⚠ Educational use only — not a medical diagnosis")

# ---------------- LOAD MODEL & SCALER ----------------
@st.cache_resource
def load_model():
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ---------------- LOAD DATA (FOR DISPLAY ONLY) ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df.drop_duplicates()

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- MODEL PERFORMANCE (STATIC EVALUATION) ----------------
X = df.drop("target", axis=1)
y = df["target"]

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", round(accuracy_score(y, y_pred), 3))
    st.text("Classification Report")
    st.text(classification_report(y, y_pred))

with col2:
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    st.pyplot(fig)

st.write("ROC-AUC Score:", round(roc_auc_score(y, y_prob), 3))

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("Feature Importance")

importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

fig2, ax2 = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", ax=ax2)
st.pyplot(fig2)

# ---------------- USER INPUT ----------------
st.subheader("🧑‍⚕️ Predict Heart Disease (User Input)")

age = st.slider("Age", 18, 100, 30)

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex == "Female" else 1

cp = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
cp = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}[cp]

trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 0 if fbs == "No" else 1

restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"])
restecg = {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2}[restecg]

thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 0 if exang == "No" else 1

oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)

slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]

ca = st.slider("Number of Major Vessels (0–3)", 0, 3, 0)

thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
thal = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]

# ---------------- VALIDATION ----------------
if trestbps < 90:
    st.warning("⚠ Resting blood pressure seems unusually low")

if chol < 120:
    st.warning("⚠ Cholesterol value seems unusually low")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠ High Risk of Heart Disease\n\n"
            f"Probability: {probability * 100:.2f}%"
        )
    else:
        st.success(
            f"✅ Low Risk of Heart Disease\n\n"
            f"Confidence: {(1 - probability) * 100:.2f}%"
        )
