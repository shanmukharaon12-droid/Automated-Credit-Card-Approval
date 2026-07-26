# Credit Card Approval Prediction System

An end-to-end Machine Learning web application designed to automate credit card approval decisions using applicant financial indicators, demographic profiles, and past-due credit history.

Built using **Python**, **Flask**, **Scikit-Learn**, **XGBoost**, **Pandas**, **Matplotlib**, **Seaborn**, and simulated **IBM Watson Machine Learning** cloud pipelines.

---

## 📌 Project Overview

Banks receive thousands of credit card applications daily. Manually evaluating financial records, credit inquiries, and past loan delinquencies is time-consuming and prone to human error. This project automates the decision process using four classification algorithms:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost Classifier**

The best-performing model is serialized and integrated into an interactive Flask Web Application with real-time risk scoring, applicant eligibility checking, batch compliance screening, and exploratory data analysis (EDA) dashboards.

---

## 🏆 Model Performance Summary

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **91.40%** | **91.44%** | **97.63%** | **94.36%** | **96.82%** | **Selected Production Model** |
| **Random Forest** | 90.60% | 90.09% | 98.42% | 93.98% | 95.78% | Evaluated |
| **XGBoost Classifier** | 89.80% | 89.81% | 97.36% | 93.35% | 95.74% | Evaluated |
| **Decision Tree** | 84.80% | 87.89% | 92.35% | 90.03% | 90.85% | Evaluated |

---

## 🎯 Application Scenarios Supported

- **Scenario 1: Automated Credit Card Application Screening**
  Credit analysts input applicant financial profiles (income, employment duration, credit inquiries) to obtain instant approval predictions.
- **Scenario 2: High-Risk Applicant Identification & Compliance Review**
  Compliance officers batch-screen applicants with past-due loan records using binary feature engineering tags.
- **Scenario 4: Customer Self-Service Eligibility Check**
  Prospective customers evaluate their financial eligibility prior to formal credit application submission.

---

## 🚀 Features

- **Real-Time Prediction Engine**: Instant approval/rejection decision with risk score gauge and factor breakdowns.
- **Batch Applicant Screening**: Filterable table for bulk compliance checks.
- **Visual Analytics & EDA**: Displays count plots, income distribution charts, feature correlation heatmap, confusion matrices, and feature importances.
- **Watson ML Cloud API**: REST endpoint (`/api/predict`) simulating cloud prediction pipeline payloads.

---

## 🛠️ Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shanmukharaon12-droid/Automated-Credit-Card-Approval.git
   cd Automated-Credit-Card-Approval
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Dataset Generator & Model Training Pipeline**:
   ```bash
   python data/generate_dataset.py
   python src/train_models.py
   ```

4. **Launch the Flask Web Application**:
   ```bash
   python app.py
   ```
   Open your browser at `http://127.0.0.1:5000`.

---

## 📂 Repository Structure

```text
Automated-Credit-Card-Approval/
├── app.py                     # Flask web server & REST API
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── data/
│   ├── generate_dataset.py    # Synthetic dataset generator
│   └── credit_card_approval_dataset.csv
├── src/
│   └── train_models.py        # EDA, Preprocessing & 4-Model Training Pipeline
├── model/
│   └── best_model.pkl         # Serialized model, scaler & encoders
├── templates/
│   ├── layout.html            # Base HTML layout
│   ├── index.html             # Home Overview
│   ├── predict.html           # Prediction Interface
│   ├── batch.html             # Batch Screening table
│   └── analytics.html         # EDA plots & metrics dashboard
└── static/
    ├── css/styles.css         # Dark mode styles & glassmorphism theme
    ├── js/main.js             # Client-side interactivity & table search
    └── images/eda/            # Generated EDA plots & visualizations
```
