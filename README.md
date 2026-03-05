# Predicting-Credit-Card-Approvals-Using-Hyper-Parameter-Optimization-Using-ElasticNet-Logistic-Regression

📌 Overview

This project implements a production-style credit risk scoring system to predict the likelihood of loan default using Logistic Regression with ElasticNet regularization. The objective is to identify high-risk borrowers and optimize lending decisions using a data-driven approach.

The model handles:

Real-world tabular financial data

Class imbalance

Categorical feature encoding

Feature scaling

Cross-validated hyperparameter optimization

Threshold tuning for business trade-offs

📊 Problem Statement

Financial institutions must assess whether a borrower is likely to default on a loan. We frame this as a binary classification problem:

0 → Fully Paid

1 → Not Fully Paid (Default)

The goal is to maximize risk detection while minimizing unnecessary rejection of creditworthy applicants.

🛠 Tech Stack

Python

Pandas

NumPy

Scikit-learn

Matplotlib

⚙️ ML Pipeline Architecture

The project uses a fully integrated Scikit-learn Pipeline:

ColumnTransformer

StandardScaler for numerical features

OneHotEncoder for categorical features

Logistic Regression (ElasticNet)

GridSearchCV for hyperparameter tuning

ROC-AUC optimization

Threshold tuning for business alignment

🔎 Model Configuration

Logistic Regression with:

Solver: saga

Penalty: elasticnet

Class Weight: balanced

Hyperparameters tuned:

C (Regularization strength)

l1_ratio (L1 vs L2 mix)

GridSearchCV:

5-fold cross-validation

Scoring metric: ROC-AUC

📈 Model Performance

Metric	                  Value

ROC-AUC                 	~0.68

Recall (Default Class)	  ~0.57

Accuracy	                ~0.64

🧠 Key Learnings

Handling imbalanced financial datasets

Designing production-style ML pipelines

Hyperparameter optimization using GridSearchCV

Interpreting logistic regression coefficients

Translating ML metrics into business impact

🚀 Future Improvements

Compare with XGBoost 

Deploy via FastAPI

Add SHAP explainability

Add CI/CD pipeline
