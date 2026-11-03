# Rainfall Prediction and Analysis Using Machine Learning

## Project overview
This project explores historical Indian sub-divisional rainfall data (1901–2017) using Python. It includes data cleaning, exploratory data analysis (EDA), regression, classification, and model evaluation.

## Dataset
Expected CSV filename: `imd_subdivisional_rainfall_1901_2017.csv`

Place the dataset in the `data/` folder. The dataset is not included in this repository. Add the original dataset source and follow its licence/redistribution terms before sharing it.

The code expects these columns:
- `YEAR`
- `SUBDIVISION`
- `ANNUAL`

## Machine learning models
1. Linear Regression
2. Decision Tree Regressor
3. K-Nearest Neighbors (KNN) Classifier
4. Decision Tree Classifier
5. Support Vector Machine (SVM) Classifier

## EDA and visualizations
- Annual rainfall distribution
- Year-wise average rainfall
- Subdivision-wise average rainfall
- Annual rainfall boxplot
- Correlation heatmap
- SVM decision boundary
- SVM ROC curve
- ROC curve comparison
- Confusion matrices

Plots are saved in the `plots/` folder when the script is run.

## Evaluation metrics
Regression: MAE, MSE, RMSE, R-squared.

Classification: Accuracy, Precision, Recall, F1-score, Log Loss, ROC-AUC, and Confusion Matrix.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

Put the CSV file in `data/`, then run:

```bash
python rainfall_project.py
```

## Important note
The classification section creates a historical binary label using the median annual rainfall:
- `0` = Low rainfall
- `1` = High rainfall

This is a historical low/high rainfall classification exercise, not a validated forecast of future rainfall. The SVM decision-boundary plot uses `YEAR` and a numeric subdivision code for visualization.
