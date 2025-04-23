# mainlr.py

import pandas as pd
import numpy as np
import os
from utils.preprocess import preprocess_data, build_preprocessing_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from models.linear_model import train_linear_regression
from models.Ridge_Regression import Ridge_regression
from models.StratifiedKFold import  precise_alpha_search, ridge_annealing_search
from sklearn.preprocessing import StandardScaler




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

def RunNormalLinear():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)


    # Align columns (very important)
    X = train_clean.drop(columns=['App Rating'])
    y = train_clean['App Rating']

    # Ensure test has same features as X
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)


    # Already cleaned earlier
    X = X.fillna(0)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    # Ensure y_train and y_val do not contain NaN
    y_train = y_train.fillna(y_train.median())
    y_val = y_val.fillna(y_val.median())

    # Initialize and train the Linear Regression model
    lr_model = train_linear_regression(X_train, y_train, X_val, y_val)
    lr_model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = lr_model.predict(X_val)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"Linear Regression RMSE: {rmse:.4f}")
    print(f"Linear Regression R² Score: {r2:.4f}")

    test_clean = test_clean.fillna(0)

    # Predict on test data
    test_predictions = lr_model.predict(test_clean)

    # Clip ratings to 1-5 range (just in case)
    test_predictions = np.clip(test_predictions, 1.0, 5.0)

    # Load sample submission
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
    submission.to_csv("submission_lr.csv", index=False)
    print("✅ Submission file saved as submission_lr.csv")

def RunRidgeRegression():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)

    # Align columns (very important)
    X = train_clean.drop(columns=['App Rating'])
    y = train_clean['App Rating']

    # Ensure test has same features as X
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

    # Already cleaned earlier
    X = X.fillna(0)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    # Ensure y_train and y_val do not contain NaN
    y_train = y_train.fillna(y_train.median())
    y_val = y_val.fillna(y_val.median())

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)


    test_clean = test_clean.fillna(0)
    test_clean = scaler.transform(test_clean)  # Standardize test data


    # Build and fit preprocessing pipeline
    preprocessor = build_preprocessing_pipeline()

    # Apply transformations (scaling, variance threshold, PCA) on train and test data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(test_clean)


    # Train the Ridge Regression model
    ridge_model = Ridge_regression(X_train_transformed, y_train, X_val_transformed, y_val)

    # Predict on test data
    test_predictions = ridge_model.predict(X_test_transformed)

    # Clip ratings to 1-5 range (just in case)
    test_predictions = np.clip(test_predictions, 1.0, 5.0)

    # Load sample submission
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
    submission.to_csv("submission_Ridge.csv", index=False)
    print("✅ Submission file saved as submission_Ridge.csv")
"""
def run_kfold_ridge():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)

    # Align columns (very important)
    X = train_clean.drop(columns=['App Rating'])
    y = train_clean['App Rating']

    # Ensure test has same features as X
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

    # Already cleaned earlier
    X = X.fillna(0)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    # Ensure y_train and y_val do not contain NaN
    y_train = y_train.fillna(y_train.median())
    y_val = y_val.fillna(y_val.median())

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)


    test_clean = test_clean.fillna(0)
    test_clean = scaler.transform(test_clean)  # Standardize test data


    # Build and fit preprocessing pipeline
    preprocessor = build_preprocessing_pipeline()

    # Apply transformations (scaling, variance threshold, PCA) on train and test data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(test_clean)


    # Train the Ridge Regression model
    ridge_model,scaler = Ridge_regression_KFold(X_train_transformed, y_train, X_val_transformed, y_val)

    # Predict on test data
    test_predictions = ridge_model.predict(X_test_transformed)

    # Clip ratings to 1-5 range (just in case)
    test_predictions = np.clip(test_predictions, 1.0, 5.0)

    # Load sample submission
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
    submission.to_csv("submission_Ridge.csv", index=False)
    print("✅ Submission file saved as submission_Ridge.csv")
"""
def run_kfold_ridge():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)

    # Align columns (very important)
    X = train_clean.drop(columns=['App Rating'])
    y = train_clean['App Rating']

    # Ensure test has same features as X
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

    # Clean data
    X = X.fillna(0)
    y = y.fillna(y.median())
    test_clean = test_clean.fillna(0)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Clean split data
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    y_train = y_train.fillna(y_train.median())
    y_val = y_val.fillna(y_val.median())

    # Build and fit preprocessing pipeline
    preprocessor = build_preprocessing_pipeline()

    # Apply transformations on train and validation data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(test_clean)

    # Train the Ridge Regression model with K-Fold
    ridge_model, scaler = precise_alpha_search(X_train_transformed, y_train, X_val_transformed, y_val)

    # Use the prediction function for test data
    test_predictions = predict_new_data(X_test_transformed, ridge_model, scaler)

    # Clip predictions to valid range
    test_predictions = np.clip(test_predictions, 1.0, 5.0)

    # Create submission
    submission_path = os.path.join("data", "SampleSubmission.csv")

    # Check if the sample submission file exists
    if os.path.exists(submission_path):
        submission = pd.read_csv(submission_path)
    else:
        # Create a new DataFrame if the file doesn't exist
        submission = pd.DataFrame({'row_id': range(len(test_app_names)), 'Y': [0] * len(test_app_names)})

    # Ensure the submission DataFrame has the correct structure
    submission['Y'] = test_predictions

    # Save predictions
    submission.to_csv("submission_Ridge_precise_alpha_search.csv", index=False)
    print("✅ Submission file saved as submission_Ridge_precise_alpha_search.csv")

    # Print model evaluation on validation set
    val_predictions = predict_new_data(X_val_transformed, ridge_model, scaler)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    val_r2 = r2_score(y_val, val_predictions)
    
    print("\nFinal Model Performance on Validation Set:")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation R² Score: {val_r2:.4f}")


def predict_new_data(X_new, model, scaler):
    X_scaled = scaler.transform(X_new)
    return model.predict(X_scaled)
 
if __name__ == "__main__":

    #RunNormalLinear()
    #RunRidgeRegression()
    run_kfold_ridge()
