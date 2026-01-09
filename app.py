import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

st.title("Heart Disease Prediction – ML Models")

# Models and their filenames
models = {
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "nb_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

model_choice = st.selectbox("Choose a Model", list(models.keys()))

# Load scaler
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    scaling_available = True
except:
    scaling_available = False

file = st.file_uploader("Upload Test CSV (heart.csv format)", type=["csv"])

if file:

    df = pd.read_csv(file)
    st.write("Uploaded Data Sample")
    st.write(df.head())

    # Remove target column if present
    if "target" in df.columns:
        y_true = df["target"]
        df = df.drop("target", axis=1)
    else:
        y_true = None

    # Ensure feature order matches training
    expected_features = [
        'age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal'
    ]

    df = df[expected_features]

    # Scale if needed (LR and KNN need scaling)
    if model_choice in ["Logistic Regression", "KNN"]:
        if scaling_available:
            df_scaled = scaler.transform(df)
        else:
            st.error("Scaler file missing! Train models again.")
            st.stop()
        X_input = df_scaled
    else:
        X_input = df.values

    # Load selected model
    model = pickle.load(open(models[model_choice], "rb"))

    y_pred = model.predict(X_input)

    st.subheader("Predictions")
    st.write(y_pred)

    if y_true is not None:
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))
    else:
        st.warning("No 'target' column found — showing predictions only.")
