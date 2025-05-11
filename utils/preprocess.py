# utils/preprocess.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from datetime import datetime


def clean_size(size_str):
    if pd.isnull(size_str):
        return None  
    if isinstance(size_str, float):  
        return None 
    size_str = size_str.replace(',', '').strip().upper()
    if 'M' in size_str:
        return float(size_str.replace('M', '')) * 1024  
    elif 'K' in size_str:
        return float(size_str.replace('K', ''))
    return None  

def clean_installs(install_str):
    if pd.isnull(install_str):
        return 0
    if isinstance(install_str, str) and install_str.lower() == 'free':
        return 0  
    try:
        return int(str(install_str).replace('+', '').replace(',', ''))
    except ValueError:
        return 0  

def clean_price(price_str):
    if pd.isnull(price_str):
        return 0.0  
    if isinstance(price_str, str):
        if price_str.lower() in ['everyone', 'free']:
            return 0.0
        try:
            return float(price_str.replace('$', '').replace(',', ''))
        except ValueError:
            return 0.0  
    else:
        return 0.0  

def clean_rating(rating, mean_rating):
    if pd.isnull(rating) or rating > 5:
        return mean_rating 
    return rating


def preprocess_data(df):
    df = df.copy()
    df['Installs'] = df['Installs'].apply(clean_installs)

        # Clean 'App Rating'
    if 'App Rating' in df.columns:
        mean_rating = df['App Rating'].mean(skipna=True)  # Calculate mean rating
        df['App Rating'] = df['App Rating'].apply(lambda x: clean_rating(x, mean_rating))

    # Fill remaining NaN values with median or 0
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    if 'Size' in df.columns:
        df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
        df['Size'] = df['Size'].fillna(df['Size'].median())

    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
            
    df = df.dropna(axis=1, how='all')
    df = pd.get_dummies(df, drop_first=True)   
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
    pipeline = Pipeline([
        ('scaler', StandardScaler()),    
        ('variance_threshold', VarianceThreshold(threshold=1e-5)),  
        ('pca', PCA(n_components=0.95))  
    ])
    return pipeline
