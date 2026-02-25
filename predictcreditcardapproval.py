
# Credit Risk Prediction using Logistic Regression
# With Hyperparameter Optimization & Imbalance Handling


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


df = pd.read_csv("C:/Users/donsh/Downloads/loan_data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# Target column
target_column = "not.fully.paid"

X = df.drop(target_column, axis=1)
y = df[target_column]
# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(exclude=["object"]).columns

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# 4. Logistic Regression Model
log_reg = LogisticRegression(
    solver="saga",
    penalty="elasticnet",
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", log_reg)
    ]
)

# 5. Hyperparameter Optimization
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__l1_ratio": [0, 0.5, 1]  # 0=L2, 1=L1
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# 6. Model Evaluation-
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

print("\nTest ROC-AUC:", roc_auc_score(y_test, y_probs))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 7. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Credit Risk Model")
plt.show()

# 8. Threshold Tuning (Business Optimization)
custom_threshold = 0.4
y_pred_custom = (y_probs >= custom_threshold).astype(int)

print("\nConfusion Matrix (Threshold = 0.4):\n")
print(confusion_matrix(y_test, y_pred_custom))

print("\nClassification Report (Threshold = 0.4):\n")
print(classification_report(y_test, y_pred_custom))

# 9. Feature Importance (Interpretability)
# Get feature names after preprocessing
ohe_features = best_model.named_steps["preprocessor"] \
    .named_transformers_["cat"] \
    .get_feature_names_out(categorical_features)

all_features = np.concatenate([numerical_features, ohe_features])

coefficients = best_model.named_steps["model"].coef_[0]

feature_importance = pd.DataFrame({
    "Feature": all_features,
    "Coefficient": coefficients
})

feature_importance = feature_importance.sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print("\nTop Influential Features:\n")
print(feature_importance.head(10))




