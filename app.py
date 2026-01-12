import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

st.title("Fetal Health Classification App")

st.write("Upload a CSV file containing **test data** (without target column).")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

model_names = [
    "Logistic Regression", "Decision Tree", "KNN",
    "Naive Bayes", "Random Forest", "XGBoost"
]

model_choice = st.selectbox("Choose Model", model_names)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Load model & scaler
    model = joblib.load(f"{model_choice.replace(' ', '_')}.pkl")
    scaler = joblib.load("scaler.pkl")

    X_scaled = scaler.transform(df)

    preds = model.predict(X_scaled)

    st.subheader("Predictions")
    st.write(preds)

    if st.button("Show Model Description"):
        st.write(f"{model_choice} model used for fetal health prediction.")
