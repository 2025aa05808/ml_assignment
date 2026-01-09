import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

st.title("Heart Disease Prediction – ML Models")

# Expected columns used during training
expected_features = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
]

# Model files
models = {
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "nb_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

# Load scaler
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    scaling_available = True
except:
    scaling_available = False

model_choice = st.selectbox("Choose a Model", list(models.keys()))

file = st.file_uploader("Upload Test CSV (heart.csv)", type=["csv"])

def normalize_column_name(col):
    """
    Unify column names to match expected features.
    Lowercase, remove spaces, underscores, hyphens.
    """
    return col.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

if file:
    df = pd.read_csv(file)
    st.write("Uploaded Data Sample:")
    st.write(df.head())

    # Clean uploaded column names
    original_cols = df.columns
    cleaned_cols = [normalize_column_name(c) for c in original_cols]
    df.columns = cleaned_cols

    # Expected columns cleaned
    cleaned_expected = [normalize_column_name(c) for c in expected_features]

    # Map cleaned → expected
    column_mapping = {}

    for exp_clean, exp_orig in zip(cleaned_expected, expected_features):
        for df_clean, df_orig in zip(cleaned_cols, original_cols):
            if df_clean == exp_clean:
                column_mapping[df_clean] = exp_orig

    # Apply mapping
    df = df.rename(columns=column_mapping)

    # Keep only expected features
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        st.error(f"Your CSV is missing required columns: {missing}")
        st.stop()

    df = df[expected_features]

    # Scaling if required
    if model_choice in ["Logistic Regression", "KNN"]:
        if not scaling_available:
            st.error("Scaler not found. Train models again.")
            st.stop()
        X_input = scaler.transform(df)
    else:
        X_input = df.values

    # Load model
    model = pickle.load(open(models[model_choice], "rb"))
    y_pred = model.predict(X_input)

    st.subheader("Predictions")
    st.write(y_pred)

    # Evaluate if target exists
    if "target" in original_cols:
        y_true = pd.read_csv(file)["target"]

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))
    else:
        st.warning("No 'target' column found — showing predictions only.")
