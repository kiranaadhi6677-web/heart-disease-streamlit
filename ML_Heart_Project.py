import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    df = df.drop_duplicates()
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- DATA SPLIT ----------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- MODEL TRAINING ----------------
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# ---------------- MODEL EVALUATION ----------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

with col2:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    st.pyplot(fig)

st.write("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("Feature Importance")

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
fig2, ax2 = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", ax=ax2)
st.pyplot(fig2)

# ---------------- USER INPUT ----------------
st.subheader("Predict Heart Disease (User Input)")

user_input = []
for col in X.columns:
    value = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease (Probability: {probability:.2f})")

# ---------------- SAVE MODEL ----------------
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
