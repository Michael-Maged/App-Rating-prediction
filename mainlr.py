# mainlr.py

import pandas as pd
import numpy as np
import os
from utils.preprocess import preprocess_data, build_preprocessing_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from models.StratifiedKFold import  precise_alpha_search, ridge_annealing_search




# Set seed for reproducibility
np.random.seed(42)

# Load datasets
train_path = os.path.join("data", "train.csv")
test_path = os.path.join("data", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Rename columns
column_map = {
    'X0': 'App Name',#drop this column
    'X1': 'Category',#drop this column
    'X2': 'Reviews',
    'X3': 'Size',#drop this column
    'X4': 'Installs',
    'X5': 'Type',
    'X6': 'Price', #drop this column
    'X7': 'Content Rating',
    'X8': 'Genres',#drop this column
    'X9': 'Last Updated',#drop this column
    'X10': 'Current Version', #drop this column
    'X11': 'Android Version', #drop this column
    'Y': 'App Rating'
}

train.rename(columns=column_map, inplace=True)
test.rename(columns=column_map, inplace=True)

columns_to_drop = [
    'App Name',          
    'Current Version',   
    'Android Version',
    'Last Updated',
    'Category',
    'Genres',
    'Price',
    'Size',
]

# Drop from both datasets
train.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test.drop(columns=columns_to_drop, inplace=True, errors='ignore')

Test_Ratings = train['App Rating'].copy()

def run_kfold_ridge():
    train_clean = preprocess_data(train)
    test_clean = preprocess_data(test)

    # Align columns (very important)
    X = train_clean.drop(columns=['App Rating'])
    y = train_clean['App Rating']

    # Ensure test has same features as X
    test_clean = test_clean.reindex(columns=X.columns, fill_value=0)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and fit preprocessing pipeline
    preprocessor = build_preprocessing_pipeline()

    # Apply transformations on train and validation data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(test_clean)

    # Train the Ridge Regression model 
    ridge_model, scaler = ridge_annealing_search(X_train_transformed, y_train, X_val_transformed, y_val)

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
        submission = pd.DataFrame({'row_id': range(len(Test_Ratings)), 'Y': [0] * len(Test_Ratings)})

    submission['Y'] = test_predictions

    submission.to_csv("submission_Ridge_annealing_search.csv", index=False)
    print(" Submission file saved as submission_Ridge_annealing_search.csv")

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
    run_kfold_ridge()
