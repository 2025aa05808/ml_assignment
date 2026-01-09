
# train_and_evaluate.py
import os
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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
import joblib

# ---------------------------
# Paths
# ---------------------------
DATA_PATH = Path("data/heart.csv")
MODEL_DIR = Path("model")
PLOTS_DIR = MODEL_DIR / "plots"
REPORTS_DIR = MODEL_DIR / "reports"

for p in [MODEL_DIR, PLOTS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(DATA_PATH)

# Ensure expected columns exist
expected_cols = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal","target"
]
missing = set(expected_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Features & target
X = df.drop(columns=["target"])
y = df["target"].astype(int)

# Split (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------------------------
# Column types
# ---------------------------
# Numeric features (continuous)
num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Categorical/discrete features (including binary categorical)
cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Preprocessor: scale numeric + one-hot encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# Helper to convert sparse to dense (needed by GaussianNB)
to_dense = FunctionTransformer(
    lambda X: X.todense() if hasattr(X, "todense") else X,
    accept_sparse=True
)

# ---------------------------
# Models dictionary
# ---------------------------
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

# Build pipelines (use dense for NB only)
pipelines = {}
for name, clf in models.items():
    if "Naive Bayes" in name:
        pipelines[name] = Pipeline(steps=[
            ("prep", preprocessor),
            ("to_dense", to_dense),
            ("clf", clf)
        ])
    else:
        pipelines[name] = Pipeline(steps=[
            ("prep", preprocessor),
            ("clf", clf)
        ])

# ---------------------------
# Train, evaluate, save
# ---------------------------
rows = []
for name, pipe in pipelines.items():
    print(f"\nTraining: {name}")
    pipe.fit(X_train, y_train)

    # Predictions
    y_pred = pipe.predict(X_test)

    # Probabilities for AUC (all chosen models support predict_proba)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        # Fallback if predict_proba not available (shouldn't happen here)
        auc = np.nan

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    rows.append({
        "ML Model Name": name,
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    })

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.close()

    # Classification report
    report_text = classification_report(y_test, y_pred, digits=4)
    with open(REPORTS_DIR / f"classification_report_{name.replace(' ', '_')}.txt", "w") as f:
        f.write(report_text)

    # Save fitted pipeline (model + preprocessing)
    joblib.dump(pipe, MODEL_DIR / f"{name.replace(' ', '_')}.joblib")

# ---------------------------
# Comparison table
# ---------------------------
metrics_df = pd.DataFrame(rows)
metrics_df.sort_values(by="AUC", ascending=False, inplace=True)
print("\n=== Metrics Summary ===")
print(metrics_df)

metrics_df.to_csv(MODEL_DIR / "metrics_heart.csv", index=False)
