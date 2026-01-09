
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Classification", layout="centered")

DATA_PATH = Path("data/heart.csv")

# Sidebar
st.sidebar.title("Controls")
model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes (Gaussian)",
        "Random Forest",
        "XGBoost"
    ],
    index=0
)
st.sidebar.write("Upload **test** CSV (same schema as training)")

uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

st.title("Heart Disease Classification – Interactive App")
st.markdown("""This app trains six classification models on the Heart dataset and lets you evaluate them on an uploaded **test** CSV or on an internal test split.""")

# Load training data
@st.cache_data
def load_training_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_training_data()

# Features & target
X = df.drop(columns=["target"])
y = df["target"].astype(int)

# Column types
num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

to_dense = FunctionTransformer(
    lambda X: X.todense() if hasattr(X, "todense") else X,
    accept_sparse=True
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree":      DecisionTreeClassifier(random_state=42),
    "kNN":                KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Random Forest":      RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost":            XGBClassifier(
                              n_estimators=300,
                              learning_rate=0.05,
                              max_depth=4,
                              subsample=0.9,
                              colsample_bytree=0.9,
                              reg_lambda=1.0,
                              n_jobs=-1,
                              random_state=42,
                              eval_metric="logloss",
                          ),
}

# Train all pipelines once
@st.cache_resource
def fit_pipelines(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipes = {}
    for name, clf in models.items():
        if "Naive Bayes" in name:
            pipe = Pipeline(steps=[("prep", preprocessor), ("to_dense", to_dense), ("clf", clf)])
        else:
            pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pipes[name] = (pipe, X_test, y_test)
    return pipes

pipes = fit_pipelines(X, y)

# Determine evaluation dataset
if uploaded is not None:
    test_df = pd.read_csv(uploaded)
    st.info(f"Using uploaded test CSV with shape: {test_df.shape}")
    # Align columns & types
    missing_cols = set(X.columns) - set(test_df.columns)
    if missing_cols:
        st.error(f"Uploaded CSV is missing columns: {missing_cols}")
        st.stop()
    X_test = test_df[X.columns]
    # If the uploaded file contains 'target', use it for metrics
    y_test = test_df["target"].astype(int) if "target" in test_df.columns else None
else:
    # Fall back to internal split for metrics
    _, X_test_builtin, y_test_builtin = pipes["Logistic Regression"]
    X_test = X_test_builtin
    y_test = y_test_builtin
    st.info("No test CSV uploaded. Using internal 20% test split.")

# Run selected model
pipe, _, _ = pipes[model_name]

# Predict
y_pred = pipe.predict(X_test)

# Probabilities (for AUC)
try:
    y_proba = pipe.predict_proba(X_test)[:, 1]
except Exception:
    y_proba = None

# Metrics
def safe_metric(fn, y_true, y_pred):
    try:
        return fn(y_true, y_pred)
    except Exception:
        return np.nan

acc = safe_metric(accuracy_score, y_test, y_pred) if y_test is not None else np.nan
prec = safe_metric(precision_score, y_test, y_pred) if y_test is not None else np.nan
rec = safe_metric(recall_score, y_test, y_pred) if y_test is not None else np.nan
f1 = safe_metric(f1_score, y_test, y_pred) if y_test is not None else np.nan
mcc = safe_metric(matthews_corrcoef, y_test, y_pred) if y_test is not None else np.nan
auc = roc_auc_score(y_test, y_proba) if (y_test is not None and y_proba is not None) else np.nan

st.subheader("Evaluation Metrics")
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    "Score":  [acc, auc, prec, rec, f1, mcc]
})
st.dataframe(metrics_df.style.format({"Score": "{:.4f}"}), use_container_width=True)

# Confusion Matrix
if y_test is not None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, digits=4))
else:
    st.warning("No ground-truth 'target' found in uploaded CSV. Metrics shown only when 'target' is present.")

st.caption("Tip: Upload only test data on Streamlit free tier to keep resource usage low.")
``
