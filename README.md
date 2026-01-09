
# Heart Disease Classification – ML Assignment 2 (BITS WILP)

> **Live App:** [🔗 Streamlit App URL](https://YOUR-STREAMLIT-APP-LINK)  
> **GitHub Repo:** [🔗 Repository URL](https://YOUR-GITHUB-REPO-LINK)

This project implements six classic machine learning classifiers on the **Heart Disease** dataset and deploys an interactive **Streamlit** web application to evaluate models on uploaded test data.

---

## 1) Problem Statement

Predict whether a patient has **heart disease** (`target` = 1) or **no heart disease** (`target` = 0) using clinical and demographic features (e.g., age, chest pain type, cholesterol).  
The goal is to build multiple classification models, evaluate them using a standard set of metrics, and present the results in a user-friendly web application.

---

## 2) Dataset Description

- **Source:** Heart Disease (UCI), popular Kaggle version (`heart.csv`)
- **Type:** Binary classification
- **Instances:** ~1025 rows
- **Features:** 13 input features + 1 target column (`target`)
- **Columns:**
  - `age` (years), `sex` (0 = female, 1 = male), `cp` (chest pain type),  
    `trestbps` (resting blood pressure), `chol` (serum cholesterol),  
    `fbs` (fasting blood sugar), `restecg` (resting ECG), `thalach` (max heart rate),  
    `exang` (exercise-induced angina), `oldpeak` (ST depression),  
    `slope` (slope of ST segment), `ca` (number of major vessels), `thal` (thalassemia),  
    `target` (0 = no disease, 1 = disease)

> Place the training file at: `data/heart.csv`  
> Ensure your uploaded **test CSV** matches the same schema.

---

## 3) Models Used & Evaluation Metrics

The following **six models** are trained on the same dataset with a consistent preprocessing pipeline:

1. **Logistic Regression**  
2. **Decision Tree Classifier**  
3. **K-Nearest Neighbors (kNN)**  
4. **Naive Bayes (Gaussian)**  
5. **Random Forest (Ensemble)**  
6. **XGBoost (Ensemble)**

**Metrics reported** for each model:

- Accuracy
- ROC–AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### 3.1 Comparison Table

> Fill the table below with values produced by `train_and_evaluate.py`  
> (See `model/metrics_heart.csv` after running the script.)

| **ML Model Name**          | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
|----------------------------|--------------|---------|---------------|------------|--------|---------|
| Logistic Regression        |              |         |               |            |        |         |
| Decision Tree              |              |         |               |            |        |         |
| kNN                        |              |         |               |            |        |         |
| Naive Bayes (Gaussian)     |              |         |               |            |        |         |
| Random Forest (Ensemble)   |              |         |               |            |        |         |
| XGBoost (Ensemble)         |              |         |               |            |        |         |

---

## 4) Observations About Model Performance

> Add concise, dataset-specific insights (bias/variance, class balance, feature importance, etc.).

| **ML Model Name**          | **Observation about model performance** |
|----------------------------|-----------------------------------------|
| Logistic Regression        |                                         |
| Decision Tree              |                                         |
| kNN                        |                                         |
| Naive Bayes (Gaussian)     |                                         |
| Random Forest (Ensemble)   |                                         |
| XGBoost (Ensemble)         |                                         |

Tips for observations:
- Compare **AUC vs. Accuracy** to discuss ranking ability vs. overall correctness.
- Note if **class imbalance** affects Precision/Recall.
- Comment on **overfitting** (e.g., tree depth vs. generalization).
- Discuss **feature importance** for tree/ensemble models.
- Mention how **scaling & one-hot encoding** impacted kNN/LogReg.

---

## 5) Project Structure
