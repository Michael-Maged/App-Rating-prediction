#maingbr.py

import pandas as pd
import numpy as np
import os
from utils.preprocess import preprocess_data, feature_engineering
from sklearn.model_selection import train_test_split
from models.gradient_boosting_model import train_gradient_boosting
from models.gradient_boosting_model import train_gradient_boosting_with_gridsearchcv
from models.gradient_boosting_model import train_gradient_boosting_with_randomizedsearchcv
from models.gradient_boosting_model import train_gradient_boosting_with_earlystopping
from sklearn.preprocessing import OneHotEncoder


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
train = feature_engineering(train)
test = feature_engineering(test)


def RunGradientBoosting():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)
    
    train_clean = feature_engineering(train_clean)
    test_clean = feature_engineering(test_clean)

    # Align columns
    X = train_clean.drop(columns=['App Rating'], errors='ignore')
    y = train_clean['App Rating']
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

    # One-Hot Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    test_clean = pd.get_dummies(test_clean, drop_first=True)

    # Align columns again after encoding
    X, test_clean = X.align(test_clean, join='left', axis=1, fill_value=0)

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting model
    gbr_model = train_gradient_boosting(X_train, y_train, X_val, y_val)

    # Predict on test data
    test_predictions = gbr_model.predict(test_clean)
    test_predictions = np.clip(test_predictions, 1.0, 5.0)

    # Save submission
    submission_path = os.path.join("data", "SampleSubmission.csv")

    # Check if the sample submission file exists
    if os.path.exists(submission_path):
        submission = pd.read_csv(submission_path)
    else:
        # Create a new DataFrame if the file doesn't exist
        submission = pd.DataFrame({'row_id': range(len(test_app_names)), 'Y': [0] * len(test_app_names)})

    # Ensure the submission DataFrame has the correct structure
    submission['Y'] = test_predictions

    # Save the submission file
    submission.to_csv("submission_gbr.csv", index=False)
    print("✅ Submission file saved as submission_gbr.csv")
    

def RunGradientBoostingGridSearchCV():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)

    # Align columns
    X = train_clean
    y = train_clean['App Rating']
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

    X = train_clean.drop(columns=['App Rating'], errors='ignore')
    X = X.fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    y_train = y_train.fillna(y_train.median())
    y_val = y_val.fillna(y_val.median())

    if 'App Rating' in test_clean.columns:
        test_clean = test_clean.drop(columns=['App Rating'])
    test_clean = test_clean.fillna(0)

    # Train Gradient Boosting model with hyperparameter tuning
    gbr_model = train_gradient_boosting_with_gridsearchcv(X_train, y_train, X_val, y_val)

    # Predict on test data
    test_predictions = gbr_model.predict(test_clean)
    test_predictions = np.clip(test_predictions, 1.0, 5.0)

    # Save submission
    submission_path = os.path.join("data", "SampleSubmission.csv")

    # Check if the sample submission file exists
    if os.path.exists(submission_path):
        submission = pd.read_csv(submission_path)
    else:
        # Create a new DataFrame if the file doesn't exist
        submission = pd.DataFrame({'row_id': range(len(test_app_names)), 'Y': [0] * len(test_app_names)})

    # Ensure the submission DataFrame has the correct structure
    submission['Y'] = test_predictions

    # Save the submission file
    submission.to_csv("submission_gbr_gridsearchcv.csv", index=False)
    print("✅ Submission file saved as submission_gbr_gridsearchcv.csv")
    
def RunGradientBoostingRandomizedSearchCV():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)

    # Align columns
    X = train_clean
    y = train_clean['App Rating']
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

    X = train_clean.drop(columns=['App Rating'], errors='ignore')
    X = X.fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    y_train = y_train.fillna(y_train.median())
    y_val = y_val.fillna(y_val.median())

    if 'App Rating' in test_clean.columns:
        test_clean = test_clean.drop(columns=['App Rating'])
    test_clean = test_clean.fillna(0)

    # Train Gradient Boosting model with hyperparameter tuning
    gbr_model = train_gradient_boosting_with_randomizedsearchcv(X_train, y_train, X_val, y_val)

    # Predict on test data
    test_predictions = gbr_model.predict(test_clean)
    test_predictions = np.clip(test_predictions, 1.0, 5.0)

    # Save submission
    submission_path = os.path.join("data", "SampleSubmission.csv")

    # Check if the sample submission file exists
    if os.path.exists(submission_path):
        submission = pd.read_csv(submission_path)
    else:
        # Create a new DataFrame if the file doesn't exist
        submission = pd.DataFrame({'row_id': range(len(test_app_names)), 'Y': [0] * len(test_app_names)})

    # Ensure the submission DataFrame has the correct structure
    submission['Y'] = test_predictions

    # Save the submission file
    submission.to_csv("submission_gbr_randomizedsearchcv.csv", index=False)
    print("✅ Submission file saved as submission_gbr_randomizedsearchcv.csv")

def RunGradientBoostingEarlyStopping():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)

    # Align columns
    X = train_clean
    y = train_clean['App Rating']
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

    X = train_clean.drop(columns=['App Rating'], errors='ignore')
    X = X.fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    y_train = y_train.fillna(y_train.median())
    y_val = y_val.fillna(y_val.median())

    if 'App Rating' in test_clean.columns:
        test_clean = test_clean.drop(columns=['App Rating'])
    test_clean = test_clean.fillna(0)

    # Train Gradient Boosting model with early stopping
    gbr_model = train_gradient_boosting_with_earlystopping(X_train, y_train, X_val, y_val)

    # Predict on test data
    test_predictions = gbr_model.predict(test_clean)
    test_predictions = np.clip(test_predictions, 1.0, 5.0)

    # Save submission
    submission_path = os.path.join("data", "SampleSubmission.csv")

    # Check if the sample submission file exists
    if os.path.exists(submission_path):
        submission = pd.read_csv(submission_path)
    else:
        # Create a new DataFrame if the file doesn't exist
        submission = pd.DataFrame({'row_id': range(len(test_app_names)), 'Y': [0] * len(test_app_names)})

    # Ensure the submission DataFrame has the correct structure
    submission['Y'] = test_predictions

    # Save the submission file
    submission.to_csv("submission_gbr_earlystopping.csv", index=False)
    print("✅ Submission file saved as submission_gbr_earlystopping.csv")

if __name__ == "__main__":

    RunGradientBoosting()
    # RunGradientBoostingGridSearchCV()
    # RunGradientBoostingRandomizedSearchCV()
    # RunGradientBoostingEarlyStopping()
    print("Done")