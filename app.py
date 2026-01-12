import streamlit as st
import pandas as pd
import joblib

st.title("Fetal Health Classification App")
st.write("Upload a CSV file containing **test data (WITHOUT fetal_health column)**.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

model_names = [
    "Logistic Regression", "Decision Tree", "KNN",
    "Naive Bayes", "Random Forest", "XGBoost"
]

model_choice = st.selectbox("Choose a Model", model_names)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ---- FIX: Remove target column if user accidentally uploads it ----
    if "fetal_health" in df.columns:
        df = df.drop("fetal_health", axis=1)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Load models
    model_path = f"model/{model_choice.replace(' ', '_')}.pkl"
    scaler_path = "model/scaler.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    predictions = model.predict(X_scaled)

    st.subheader("Predictions (1=Normal, 2=Suspect, 3=Pathological)")
    st.write(predictions)
