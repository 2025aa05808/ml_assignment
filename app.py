import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

st.title("Heart Disease Prediction – ML Models")

# Correct feature list based on your uploaded dataset
expected_features = [
    'Age',
    'Sex',
    'ChestPainType',
    'RestingBP',
    'Cholesterol',
    'FastingBS',
    'RestingECG',
    'MaxHR',
    'ExerciseAngina',
    'Oldpeak',
    'ST_Slope'
]

models = {
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "nb_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

# Load scaler if exists
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    scaling_available = True
except:
    scaling_available = False

model_choice = st.selectbox("Choose a Model", list(models.keys()))

file = st.file_uploader("Upload Test CSV (same structure as training heart.csv)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Uploaded Data Sample:")
    st.write(df.head())

    # Separate target if present
    if "HeartDisease" in df.columns:
        y_true = df["HeartDisease"]
        df = df.drop("HeartDisease", axis=1)
    else:
        y_true = None

    # Select only the features used during training
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        st.error(f"Your CSV is missing required columns: {missing}")
        st.stop()

    df = df[expected_features]

    # Scale only when required
    if model_choice in ["Logistic Regression", "KNN"]:
        if scaling_available:
            X_input = scaler.transform(df)
        else:
            st.error("Missing scaler.pkl — please save the scaler during training!")
            st.stop()
    else:
        X_input = df.values

    # Load model
    model = pickle.load(open(models[model_choice], "rb"))

    # Predict
    y_pred = model.predict(X_input)

    st.subheader("Predictions")
    st.write(y_pred)

    # If target exists in uploaded file → show evaluation
    if y_true is not None:
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))
    else:
        st.warning("No 'HeartDisease' column found → showing predictions only")
