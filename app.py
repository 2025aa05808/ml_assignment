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
st.write("Upload test data (CSV) WITHOUT the fetal_health column.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

model_names = [
    "Logistic Regression", "Decision Tree", "KNN",
    "Naive Bayes", "Random Forest", "XGBoost"
]

model_choice = st.selectbox("Choose a Model", model_names)

if uploaded_file is not None:

    # Load data
    df = pd.read_csv(uploaded_file)

    # Remove target column if present
    if "fetal_health" in df.columns:
        df = df.drop("fetal_health", axis=1)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Load model + scaler
    model = joblib.load(f"model/{model_choice.replace(' ', '_')}.pkl")
    scaler = joblib.load("model/scaler.pkl")

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    predictions = model.predict(X_scaled)
    st.subheader("Predictions (1=Normal, 2=Suspect, 3=Pathological)")
    st.write(predictions)

    # ---------------------------------------------
    # If true labels are present in upload → evaluate
    # ---------------------------------------------
    st.subheader("Evaluation Metrics")

    if "true_label" in df.columns:
        true_labels = df["true_label"]
    else:
        st.info("To view metrics, add a column 'true_label' with real fetal health labels.")
        st.stop()

    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    mcc = matthews_corrcoef(true_labels, predictions)

    # Display metrics
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**MCC:** {mcc:.4f}")

    # ---------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(true_labels, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[1,2,3], yticklabels=[1,2,3], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ---------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------
    st.subheader("Classification Report")

    report = classification_report(true_labels, predictions, output_dict=False)
    st.text(report)
