#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rainfall Prediction and Analysis using Machine Learning
Dataset: IMD Sub-Divisional Rainfall (1901-2017)

Before running:
1. Put the CSV file inside the data/ folder.
2. Update DATA_PATH below if your filename is different.
3. Install dependencies: pip install -r requirements.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. PATHS AND DATASET LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "imd_subdivisional_rainfall_1901_2017.csv"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Put imd_subdivisional_rainfall_1901_2017.csv inside the data/ folder."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\nFIRST FIVE ROWS")
print(df.head())
print("\nDATASET SHAPE:", df.shape)
print("\nCOLUMNS:", df.columns.tolist())
print("\nDATASET INFORMATION")
df.info()
print("\nSTATISTICAL SUMMARY")
print(df.describe(include="all"))
print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())
print("\nDUPLICATE ROWS:", df.duplicated().sum())

# --------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------

df = df.drop_duplicates()

required_columns = ["YEAR", "SUBDIVISION", "ANNUAL"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Required columns missing from dataset: {missing_columns}")

# Convert required numeric columns to numeric values
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
df["ANNUAL"] = pd.to_numeric(df["ANNUAL"], errors="coerce")

# Remove rows where required values are unavailable
df = df.dropna(subset=required_columns).copy()
df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.strip()

# Fill remaining numeric missing values with medians
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill remaining categorical missing values with mode, if any
for col in df.select_dtypes(include="object").columns:
    if df[col].isna().any() and not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# --------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------

# Annual rainfall distribution
plt.figure(figsize=(8, 5))
plt.hist(df["ANNUAL"], bins=20, edgecolor="black")
plt.title("Annual Rainfall Distribution")
plt.xlabel("Annual Rainfall")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_annual_rainfall_distribution.png", dpi=150)
plt.show()

# Year-wise average rainfall
yearly_rainfall = df.groupby("YEAR")["ANNUAL"].mean()
plt.figure(figsize=(10, 5))
plt.plot(yearly_rainfall.index, yearly_rainfall.values)
plt.title("Year-wise Average Rainfall")
plt.xlabel("Year")
plt.ylabel("Average Annual Rainfall")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_year_wise_rainfall.png", dpi=150)
plt.show()

# Subdivision-wise average rainfall
subdivision_rainfall = df.groupby("SUBDIVISION")["ANNUAL"].mean().sort_values()
plt.figure(figsize=(12, 6))
subdivision_rainfall.plot(kind="bar")
plt.title("Average Rainfall by Subdivision")
plt.xlabel("Subdivision")
plt.ylabel("Average Annual Rainfall")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_subdivision_wise_rainfall.png", dpi=150)
plt.show()

# Annual rainfall boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["ANNUAL"])
plt.title("Annual Rainfall Boxplot")
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_annual_rainfall_boxplot.png", dpi=150)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_correlation_heatmap.png", dpi=150)
plt.show()

# --------------------------------------------------
# 4. REGRESSION DATA PREPARATION
# --------------------------------------------------

X_reg = pd.get_dummies(
    df[["YEAR", "SUBDIVISION"]],
    columns=["SUBDIVISION"],
    dtype=int
)
y_reg = df["ANNUAL"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 5. MODEL 1 - LINEAR REGRESSION
# --------------------------------------------------

linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)
linear_pred = linear_model.predict(X_test_reg)

# --------------------------------------------------
# 6. MODEL 2 - DECISION TREE REGRESSION
# --------------------------------------------------

tree_reg_model = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_reg_model.fit(X_train_reg, y_train_reg)
tree_reg_pred = tree_reg_model.predict(X_test_reg)

print("\nREGRESSION MODEL RESULTS")
regression_results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree Regressor"],
    "MAE": [
        mean_absolute_error(y_test_reg, linear_pred),
        mean_absolute_error(y_test_reg, tree_reg_pred)
    ],
    "MSE": [
        mean_squared_error(y_test_reg, linear_pred),
        mean_squared_error(y_test_reg, tree_reg_pred)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test_reg, linear_pred)),
        np.sqrt(mean_squared_error(y_test_reg, tree_reg_pred))
    ],
    "R2 Score": [
        r2_score(y_test_reg, linear_pred),
        r2_score(y_test_reg, tree_reg_pred)
    ]
})
print(regression_results.to_string(index=False))
regression_results.to_csv(BASE_DIR / "regression_model_results.csv", index=False)

# --------------------------------------------------
# 7. CLASSIFICATION TARGET
# --------------------------------------------------

# This is a historical low/high rainfall classification task.
# 0 = Low rainfall; 1 = High rainfall.
rainfall_median = df["ANNUAL"].median()
y_class = (df["ANNUAL"] >= rainfall_median).astype(int)

# One-hot encoded features for KNN and Decision Tree
X_class = pd.get_dummies(
    df[["YEAR", "SUBDIVISION"]],
    columns=["SUBDIVISION"],
    dtype=int
)

# Create one shared train/test split for fair model comparison
all_indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

X_train_class = X_class.iloc[train_idx]
X_test_class = X_class.iloc[test_idx]
y_train_class = y_class.iloc[train_idx]
y_test_class = y_class.iloc[test_idx]

# --------------------------------------------------
# 8. MODEL 3 - KNN CLASSIFIER
# --------------------------------------------------

knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
knn_model.fit(X_train_class, y_train_class)
knn_pred = knn_model.predict(X_test_class)
knn_probability = knn_model.predict_proba(X_test_class)[:, 1]

# --------------------------------------------------
# 9. MODEL 4 - DECISION TREE CLASSIFIER
# --------------------------------------------------

tree_class_model = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_class_model.fit(X_train_class, y_train_class)
tree_class_pred = tree_class_model.predict(X_test_class)
tree_class_probability = tree_class_model.predict_proba(X_test_class)[:, 1]

# --------------------------------------------------
# 10. MODEL 5 - SUPPORT VECTOR MACHINE (SVM)
# --------------------------------------------------

# Use two features so the decision boundary can be plotted.
# Subdivision is converted to numeric category codes.
df["SUBDIVISION_CODE"] = df["SUBDIVISION"].astype("category").cat.codes

X_svm = df[["YEAR", "SUBDIVISION_CODE"]].to_numpy()
X_train_svm = X_svm[train_idx]
X_test_svm = X_svm[test_idx]

svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", probability=True, random_state=42)
)
svm_model.fit(X_train_svm, y_train_class)
svm_pred = svm_model.predict(X_test_svm)
svm_probability = svm_model.predict_proba(X_test_svm)[:, 1]

# --------------------------------------------------
# 11. CLASSIFICATION MODEL PERFORMANCE
# --------------------------------------------------

classification_results = pd.DataFrame({
    "Model": ["KNN Classifier", "Decision Tree Classifier", "SVM"],
    "Accuracy": [
        accuracy_score(y_test_class, knn_pred),
        accuracy_score(y_test_class, tree_class_pred),
        accuracy_score(y_test_class, svm_pred)
    ],
    "Precision": [
        precision_score(y_test_class, knn_pred, zero_division=0),
        precision_score(y_test_class, tree_class_pred, zero_division=0),
        precision_score(y_test_class, svm_pred, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test_class, knn_pred, zero_division=0),
        recall_score(y_test_class, tree_class_pred, zero_division=0),
        recall_score(y_test_class, svm_pred, zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test_class, knn_pred, zero_division=0),
        f1_score(y_test_class, tree_class_pred, zero_division=0),
        f1_score(y_test_class, svm_pred, zero_division=0)
    ],
    "Log Loss": [
        log_loss(y_test_class, knn_model.predict_proba(X_test_class)),
        log_loss(y_test_class, tree_class_model.predict_proba(X_test_class)),
        log_loss(y_test_class, svm_model.predict_proba(X_test_svm))
    ]
})

print("\nCLASSIFICATION MODEL RESULTS")
print(classification_results.to_string(index=False))
classification_results.to_csv(BASE_DIR / "classification_model_results.csv", index=False)

# --------------------------------------------------
# 12. SVM DECISION BOUNDARY AND SCATTER PLOT
# --------------------------------------------------

# Display a manageable sample of points for plotting.
# The model itself is trained using the training split.
plot_sample = min(1500, len(X_svm))
rng = np.random.RandomState(42)
plot_indices = rng.choice(len(X_svm), size=plot_sample, replace=False)
X_plot = X_svm[plot_indices]
y_plot = y_class.iloc[plot_indices].to_numpy()

plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(
    svm_model,
    X_svm,
    response_method="predict",
    alpha=0.35,
    cmap="Pastel1",
    xlabel="Year",
    ylabel="Subdivision Code"
)
plt.scatter(
    X_plot[:, 0],
    X_plot[:, 1],
    c=y_plot,
    s=25,
    edgecolors="black",
    cmap="coolwarm"
)
plt.title("SVM Decision Boundary - Rainfall Dataset")
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_svm_decision_boundary.png", dpi=150)
plt.show()

# --------------------------------------------------
# 13. ROC CURVE - SVM
# --------------------------------------------------

fpr_svm, tpr_svm, _ = roc_curve(y_test_class, svm_probability)
svm_auc = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(8, 5))
plt.plot(fpr_svm, tpr_svm, label=f"SVM AUC = {svm_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_svm_roc_curve.png", dpi=150)
plt.show()

# --------------------------------------------------
# 14. ROC CURVE COMPARISON
# --------------------------------------------------

fpr_knn, tpr_knn, _ = roc_curve(y_test_class, knn_probability)
fpr_tree, tpr_tree, _ = roc_curve(y_test_class, tree_class_probability)

plt.figure(figsize=(8, 5))
plt.plot(fpr_knn, tpr_knn, label=f"KNN AUC = {auc(fpr_knn, tpr_knn):.2f}")
plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree AUC = {auc(fpr_tree, tpr_tree):.2f}")
plt.plot(fpr_svm, tpr_svm, label=f"SVM AUC = {svm_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_roc_curve_comparison.png", dpi=150)
plt.show()

# --------------------------------------------------
# 15. CONFUSION MATRICES
# --------------------------------------------------

classifier_predictions = {
    "KNN": knn_pred,
    "Decision Tree": tree_class_pred,
    "SVM": svm_pred
}

for model_name, predictions in classifier_predictions.items():
    cm = confusion_matrix(y_test_class, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low Rainfall", "High Rainfall"]
    )
    display.plot(cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(PLOT_DIR / f"confusion_matrix_{safe_name}.png", dpi=150)
    plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY.")
print("Plots are saved in the plots/ folder.")
print("Model result CSV files are saved in the project folder.")
