import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Fetal Health Classification App")
st.write("Upload test CSV with **fetal_health** column as true labels.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

model_names = [
    "Logistic Regression", "Decision Tree", "KNN",
    "Naive Bayes", "Random Forest", "XGBoost"
]

model_choice = st.selectbox("Choose a Model", model_names)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # --------------------------------------
    # Detect true labels
    # --------------------------------------
    if "fetal_health" in df.columns:
        true_labels = df["fetal_health"]
    elif "true_label" in df.columns:
        true_labels = df["true_label"]
    else:
        st.error("CSV must contain fetal_health column as true label.")
        st.stop()

    # --------------------------------------
    # Prepare features (drop label before prediction)
    # --------------------------------------
    X = df.drop(["fetal_health"], axis=1, errors="ignore")
    X = df.drop(["true_label"], axis=1, errors="ignore")

    # Load model + scaler
    model = joblib.load(f"model/{model_choice.replace(' ', '_')}.pkl")
    scaler = joblib.load("model/scaler.pkl")

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    st.subheader("Model Predictions (1=Normal, 2=Suspect, 3=Pathological)")
    st.write(predictions)

    # --------------------------------------
    # Evaluation Metrics
    # --------------------------------------
    st.subheader("Evaluation Metrics")

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    mcc = matthews_corrcoef(true_labels, predictions)

    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**MCC:** {mcc:.4f}")

    # --------------------------------------
    # Confusion Matrix
    # --------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(true_labels, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[1, 2, 3], yticklabels=[1, 2, 3], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # --------------------------------------
    # Classification Report
    # --------------------------------------
    st.subheader("Classification Report")
    report = classification_report(true_labels, predictions)
    st.text(report)
