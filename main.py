# main.py

import pandas as pd
import numpy as np
import os
from utils.preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from models.linear_model import train_linear_regression
from models.gradient_boosting_model import train_gradient_boosting

# Set seed for reproducibility
np.random.seed(42)

# Load datasets
train_path = os.path.join("data", "train.csv")
test_path = os.path.join("data", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Rename columns
column_map = {
    'X0': 'App Name',
    'X1': 'Category',
    'X2': 'Reviews',
    'X3': 'Size',
    'X4': 'Installs',
    'X5': 'Type',
    'X6': 'Price',
    'X7': 'Content Rating',
    'X8': 'Genres',
    'X9': 'Last Updated',
    'X10': 'Current Version',
    'X11': 'Android Version',
    'Y': 'App Rating'
}

train.rename(columns=column_map, inplace=True)
test.rename(columns=column_map, inplace=True)

# Show basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:\n", train.columns)
print("\nMissing values:\n", train.isnull().sum())

test_app_names = test['App Name'].copy()
train = preprocess_data(train)
test = preprocess_data(test)

# Align columns (very important)
X = train.drop(columns=['App Rating'], errors='ignore')
y = train['App Rating']

# Ensure test has same features as X
test = test.reindex(columns=X.columns, fill_value=0)

# Fill missing values
X = X.fillna(0)
test = test.fillna(0)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
y_train = y_train.fillna(y_train.median())
y_val = y_val.fillna(y_val.median())

# === Linear Regression ===
lr_model = train_linear_regression(X_train, y_train, X_val, y_val)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_val)

rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
r2_lr = r2_score(y_val, y_pred_lr)

print(f"Linear Regression RMSE: {rmse_lr:.4f}")
print(f"Linear Regression R² Score: {r2_lr:.4f}")

# === Gradient Boosting Regressor ===
gbr_model = train_gradient_boosting(X_train, y_train, X_val, y_val)

# Predict on test data
test_predictions_lr = lr_model.predict(test)
test_predictions_gbr = gbr_model.predict(test)

# Clip predictions
test_predictions_lr = np.clip(test_predictions_lr, 1.0, 5.0)
test_predictions_gbr = np.clip(test_predictions_gbr, 1.0, 5.0)

# Load sample submission or create new one
submission_path = os.path.join("data", "sample_submission.csv")
if os.path.exists(submission_path):
    submission = pd.read_csv(submission_path)
else:
    submission = pd.DataFrame({'App Name': test_app_names, 'App Rating': [0] * len(test_app_names)})

# Save Linear Regression submission
submission_lr = submission.copy()
submission_lr['App Rating'] = test_predictions_lr
submission_lr.to_csv("submission_lr.csv", index=False)
print("✅ Submission file saved as submission_lr.csv")

# Save GBR submission
submission_gbr = submission.copy()
submission_gbr['App Rating'] = test_predictions_gbr
submission_gbr.to_csv("submission_gbr.csv", index=False)
print("✅ Submission file saved as submission_gbr.csv")