#maingbr.py
import pandas as pd
import numpy as np
import os
from utils.preprocess import *
from sklearn.model_selection import train_test_split
from models.gradient_boosting_model import train_gradient_boosting
from visualize import *

np.random.seed(42)

train_path = os.path.join("data", "train.csv")
test_path = os.path.join("data", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

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

train['Price'] = train['Price'].apply(clean_price)

test_app_names = test['App Name'].copy()

train = preprocess_data(train)
test = preprocess_data(test)
train = feature_engineering(train)
test = feature_engineering(test)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:\n", train.columns)
print("\nMissing values:\n", train.isnull().sum())

def RunGradientBoosting():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)
    train_clean = feature_engineering(train)
    test_clean = feature_engineering(test)

    X = train_clean.drop(columns=['App Rating'], errors='ignore')
    y = train_clean['App Rating']
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

    X = pd.get_dummies(X, drop_first=True)
    test_clean = pd.get_dummies(test_clean, drop_first=True)

    X, test_clean = X.align(test_clean, join='left', axis=1, fill_value=0)
    print(f"Train:\n{X.head()}")
    print(f"Test:\n{test_clean.head()}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    gbr_model = train_gradient_boosting(X_train, y_train, X_val, y_val)

    test_predictions = gbr_model.predict(test_clean)
    test_predictions = np.clip(test_predictions, 1.0, 5.0)

    submission_path = os.path.join("data", "SampleSubmission.csv")

    if os.path.exists(submission_path):
        submission = pd.read_csv(submission_path)
    else:
        submission = pd.DataFrame({'row_id': range(len(test_app_names)), 'Y': [0] * len(test_app_names)})

    submission['Y'] = test_predictions

    submission.to_csv("submission_gbr.csv", index=False)
    print("✅ Submission file saved as submission_gbr.csv")
    
    plot_residuals(y_val, gbr_model.predict(X_val))
    plot_predictions_vs_actual(y_val, gbr_model.predict(X_val))

if __name__ == "__main__":

    RunGradientBoosting()
    print("Done")