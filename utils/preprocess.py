# utils/preprocess.py

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from datetime import datetime


def clean_size(size_str):
    if pd.isnull(size_str):
        return None  # Or you can return 0 or NaN if you prefer
    if isinstance(size_str, float):  # Check if the value is a float
        return None  # Or 0 or np.nan if you prefer
    size_str = size_str.replace(',', '').strip().upper()
    if 'M' in size_str:
        return float(size_str.replace('M', '')) * 1024  # Convert MB to KB
    elif 'K' in size_str:
        return float(size_str.replace('K', ''))
    return None  # Return None or a default value if the size can't be parsed


def clean_installs(install_str):
    if pd.isnull(install_str):
        return 0
    if isinstance(install_str, str) and install_str.lower() == 'free':
        return 0  # Or np.nan if you prefer
    try:
        # Make sure to convert to string before replacing characters
        return int(str(install_str).replace('+', '').replace(',', ''))
    except ValueError:
        # If the value can't be converted to an integer, return 0 or NaN
        return 0  # or np.nan


def clean_price(price_str):
    if pd.isnull(price_str):
        return 0.0  # Or np.nan if you prefer missing values to be NaN
    if isinstance(price_str, str):
        if price_str.lower() in ['everyone', 'free']:
            return 0.0  # Or np.nan if you prefer
        try:
            return float(price_str.replace('$', '').replace(',', ''))
        except ValueError:
            return 0.0  # Or np.nan if the value can't be converted to a float
    else:
        return 0.0  # Handle the case when the value is a float or other non-string type

def clean_rating(rating, mean_rating):
    
    if pd.isnull(rating) or rating > 5:
        return mean_rating  # Replace with the mean rating
    return rating

def encode_features(df):
    df = df.copy()

    # Drop non-informative columns
    if 'App Name' in df.columns:
        df.drop(columns=['App Name'], inplace=True)

    # Handle missing 'Size' safely (ensure numeric)
    if 'Size' in df.columns:
        df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
        df['Size'] = df['Size'].fillna(df['Size'].median())

    # Handle missing 'App Rating' (only in train)
    if 'App Rating' in df.columns:
        df['App Rating'] = pd.to_numeric(df['App Rating'], errors='coerce')
        df['App Rating'] = df['App Rating'].fillna(df['App Rating'].median())

    # Fill NA in categorical with mode
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # One-Hot Encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    return df


def preprocess_data(df):
    df = df.copy()
    df['Installs'] = df['Installs'].apply(clean_installs)


        # Clean 'App Rating'
    if 'App Rating' in df.columns:
        mean_rating = df['App Rating'].mean(skipna=True)  # Calculate mean rating
        df['App Rating'] = df['App Rating'].apply(lambda x: clean_rating(x, mean_rating))


    cols_all_nan = df.columns[df.isna().all()].tolist()
    print("Dropped columns (all NaN):", cols_all_nan)

    # Drop columns with all NaN values
    df = df.dropna(axis=1, how='all')

    # Fill remaining NaN values with median or 0
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    df = encode_features(df)
    return df


def feature_engineering(df):
    df = df.copy()

    # Create new features
    if 'Last Updated' in df.columns:
        # Convert 'Last Updated' to datetime and calculate days since last update
        df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
        df['Days Since Last Update'] = (datetime.now() - df['Last Updated']).dt.days
        df.drop(columns=['Last Updated'], inplace=True)  # Drop original column

    if 'Reviews' in df.columns and 'Installs' in df.columns:
        # Create a feature for reviews per install
        df['Reviews per Install'] = df['Reviews'] / (df['Installs'] + 1)  # Avoid division by zero

    if 'Price' in df.columns:
        # Create a categorical feature for price ranges
        df['Price Category'] = pd.cut(
            df['Price'],
            bins=[-1, 0, 5, 20, float('inf')],
            labels=['Free', 'Low', 'Medium', 'High']
        )

    if 'Size' in df.columns:
        # Create a feature for app size categories
        df['Size Category'] = pd.cut(
            df['Size'],
            bins=[-1, 1024, 5120, 10240, float('inf')],  # Example bins in KB
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )

    # Drop irrelevant or redundant columns
    if 'App Name' in df.columns:
        df.drop(columns=['App Name'], inplace=True)

    return df


def build_preprocessing_pipeline():
    """
    Builds a preprocessing pipeline with scaling and PCA.

    Returns:
        Pipeline: A scikit-learn pipeline for preprocessing.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features     
        ('variance_threshold', VarianceThreshold(threshold=1e-5)),  # Remove low-variance features
        ('pca', PCA(n_components=0.95))  # Reduce dimensionality, keeping 95% variance
    ])
    return pipeline

def build_preprocessing_pipeline2(X, use_pca=True):
    """
    Builds a preprocessing pipeline with scaling, one-hot encoding, 
    and optional PCA.
    
    Parameters:
        X (DataFrame): The input data.
        use_pca (bool): Whether to apply PCA for dimensionality reduction.
    
    Returns:
        Pipeline: A scikit-learn pipeline for preprocessing.
    """
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    transformers = [
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values for numerical columns
            ('scaler', StandardScaler())  # Scale numerical features
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values for categorical columns
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
        ]), categorical_cols)
    ]

    preprocessor = ColumnTransformer(transformers)

    steps = [('preprocessor', preprocessor)]

    if use_pca:
        steps.append(('variance_threshold', VarianceThreshold(threshold=1e-5)))  # Remove low-variance features
        steps.append(('pca', PCA(n_components=0.95)))  # PCA for dimensionality reduction

    return Pipeline(steps)