import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

st.title("Heart Disease Prediction – ML Models")

models = {
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "nb_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

model_choice = st.selectbox("Choose a Model", list(models.keys()))

file = st.file_uploader("Upload Test CSV (heart.csv format)", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("Uploaded Data Sample:")
    st.write(df.head())

    if "HeartDisease" in df.columns:
        y_true = df["HeartDisease"]
        df = df.drop("HeartDisease", axis=1)
    else:
        y_true = None

    # Load full pipeline (preprocessing + model)
    model = pickle.load(open(models[model_choice], "rb"))

    # Predict
    y_pred = model.predict(df)

    st.subheader("Predictions")
    st.write(y_pred)

    if y_true is not None:
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))
    else:
        st.warning("No target column found — predictions only")
