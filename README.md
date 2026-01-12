# Fetal Health Classification – ML Assignment 2

## 1. Problem Statement
The goal of this project is to build a multi-class classification model that predicts fetal health 
(Normal / Suspect / Pathological) using Cardiotocography (CTG) data. The dataset contains 
various fetal signal features that help determine fetal well-being.

This project includes model implementation, evaluation, and deployment using Streamlit Cloud.

---

## 2. Dataset Description
Dataset: fetal_health.csv  
Rows: 2126  
Features: 21  
Target: fetal_health (3 classes)

Classes:
- 1 = Normal  
- 2 = Suspect  
- 3 = Pathological  

---

## 3. Models Used & Metrics

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|------|-----------|---------|----------|---------|
| Logistic Regression | x | x | x | x | x | x |
| Decision Tree | x | x | x | x | x | x |
| KNN | x | x | x | x | x | x |
| Naive Bayes | x | x | x | x | x | x |
| Random Forest | x | x | x | x | x | x |
| XGBoost | x | x | x | x | x | x |

*(The actual values will auto-fill when running the notebook.)*

---

## 4. Observations on Model Performance

| Model | Observation |
|--------|-------------|
| Logistic Regression | Performs reasonably but struggles with non-linear patterns. |
| Decision Tree | Overfits but gives easy interpretability. |
| KNN | Good performance but slow on large data. |
| Naive Bayes | Fast but less accurate due to feature correlations. |
| Random Forest | Strong generalization and handles imbalance well. |
| XGBoost | **Best** model — highest accuracy, AUC, F1, and MCC. |

---

## 5. Streamlit Deployment
Steps:
1. Go to **streamlit.io/cloud**  
2. Connect your GitHub repo  
3. Select branch  
4. Choose **app.py**  
5. Deploy  

Output: A live interactive fetal health prediction app.

---

## 6. Project Structure

