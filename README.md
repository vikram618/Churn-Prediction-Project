# Churn-Prediction-Project
This repository contains my Churn Prediction Project
# 📡 Customer Churn Prediction

A machine learning web application that predicts whether a telecom customer is likely to churn, built using Python and deployed with Streamlit.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges in the telecom industry. This project builds and compares multiple machine learning models to predict churn early, helping businesses take proactive action to retain customers.

---

## 🗂️ Project Structure

```
Telecom Churn Project/
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── P670-dataset.xlsx             # Dataset
├── Finalised_EDA.ipynb           # Exploratory Data Analysis notebook
├── Model_Building_Evaluation.ipynb  # Model building & evaluation notebook
└── README.md                     # Project documentation
```

---

## 🔄 Project Workflow

```
Raw Data → EDA → Preprocessing → Model Building → Evaluation → Deployment
```

1. **EDA** — Explored distributions, correlations, outliers, and churn patterns
2. **Preprocessing** — Handled missing values, clipped outliers, encoded categoricals, scaled features
3. **Model Building** — Trained 6 classifiers and compared performance
4. **Hyperparameter Tuning** — Tuned the best model using GridSearchCV
5. **Deployment** — Deployed as an interactive web app using Streamlit

---

## 🤖 Models Used

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| K-Nearest Neighbors | Instance-based |
| Support Vector Machine | Kernel-based |

The **Tuned Random Forest** was selected as the final model based on ROC-AUC score.

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## 🚀 How to Run the App

**1. Clone or download the project**

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

**4. Open your browser** at `http://localhost:8501` and upload your dataset from the sidebar.

---

## 🛠️ Tech Stack

- **Language** — Python 3.x
- **Data Analysis** — Pandas, NumPy
- **Visualization** — Matplotlib, Seaborn
- **Machine Learning** — Scikit-learn
- **Deployment** — Streamlit

---

## 👥 Team

Developed as part of an academic data science project.
