# utils/preprocess.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


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
    df['Size'] = df['Size'].apply(clean_size)
    df['Installs'] = df['Installs'].apply(clean_installs)
    df['Price'] = df['Price'].apply(clean_price)
    
    # Drop columns with all NaN values
    df = df.dropna(axis=1, how='all')
    
    # Fill remaining NaN values with median or 0
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    df = encode_features(df)
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
