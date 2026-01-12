import streamlit as st
import pandas as pd
import joblib

st.title("Fetal Health Classification App")
st.write("Upload a CSV file containing **test data** (without the target column).")

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# -------------------------
# Model Selection
# -------------------------
model_names = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

model_choice = st.selectbox("Choose a Model", model_names)

# -------------------------
# Prediction
# -------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Load model & scaler from model/ folder
    model_path = f"model/{model_choice.replace(' ', '_')}.pkl"
    scaler_path = "model/scaler.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(df)

    predictions = model.predict(X_scaled)

    st.subheader("Model Predictions")
    st.write(predictions)

    # Optional button
    if st.button("Explain Model"):
        st.write(f"Model used: **{model_choice}**")

