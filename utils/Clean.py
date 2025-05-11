import numpy as np
import pandas as pd

column_map = {
'X1': 'Category',
'X5': 'Type',
'X7': 'Content Rating',
'X8': 'Genres',
}

def clean_reviews(reviews):
    """Clean the Reviews column by converting to numeric and handling missing values."""
    try:
        return int(reviews.replace(',', '')) if isinstance(reviews, str) else reviews
    except ValueError:
        return 0  # Default to 0 if conversion fails

def categorize_reviews(reviews):
    """
    Categorize the Reviews column into bins based on the number of reviews.

    Parameters:
        reviews (int): The number of reviews.

    Returns:
        str: The category of the reviews.
    """
    if pd.isnull(reviews):
        return 'Unknown'
    elif reviews < 100:
        return 'Very Low'
    elif reviews < 1000:
        return 'Low'
    elif reviews < 10000:
        return 'Moderate'
    elif reviews < 100000:
        return 'High'
    elif reviews < 1000000:
        return 'Very High'
    else:
        return 'Massive'

def clean_size(size_str):
    """Clean the Size column by converting to KB."""
    if pd.isnull(size_str) or not isinstance(size_str, str):
        return None
    size_str = size_str.replace(',', '').strip().upper()
    try:
        if 'M' in size_str:
            return float(size_str.replace('M', '')) * 1024  # Convert MB to KB
        elif 'K' in size_str:
            return float(size_str.replace('K', ''))
        else:
            return None
    except ValueError:
        return None

def categorize_size(size_in_kb):
    """Categorize the Size column into bins."""
    if pd.isnull(size_in_kb):
        return 'Unknown'
    elif size_in_kb < 100:
        return 'Tiny'
    elif size_in_kb < 1000:
        return 'Small'
    elif size_in_kb < 5000:
        return 'Medium'
    elif size_in_kb < 20000:
        return 'Large'
    elif size_in_kb < 50000:
        return 'Very Large'
    else:
        return 'Huge'

def clean_installs(install_str):
    """Clean the Installs column by removing symbols and converting to integer."""
    if pd.isnull(install_str):
        return 0
    try:
        return int(install_str.replace('+', '').replace(',', ''))
    except ValueError:
        return 0

def categorize_installs(installs):
    """Categorize the Installs column into bins."""
    if installs < 1000:
        return 'Very Low'
    elif installs < 100000:
        return 'Low'
    elif installs < 1000000:
        return 'Moderate'
    elif installs < 10000000:
        return 'High'
    elif installs < 50000000:
        return 'Very High'
    else:
        return 'Massive'

def clean_price(price_str):
    """Clean the Price column by removing symbols and converting to float."""
    if pd.isnull(price_str):
        return 0.0
    try:
        return float(price_str.replace('$', '').replace(',', ''))
    except ValueError:
        return 0.0

def clean_rating(rating, mean_rating):
    """Clean the App Rating column by replacing invalid values with the mean."""
    if pd.isnull(rating) or rating > 5:
        return mean_rating
    return rating

def Cpreprocess_data(df):

    # Drop unneeded columns
    columns_to_drop = ['App Name', 'Last Updated', 'Current Version', 'Android Version']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Clean and categorize columns
    if 'Reviews' in df.columns:
        df['Reviews'] = df['Reviews'].apply(clean_reviews)
    if 'Size' in df.columns:
        df['Size'] = df['Size'].apply(clean_size)
    if 'Installs' in df.columns:
        df['Installs'] = df['Installs'].apply(clean_installs)
    if 'Price' in df.columns:
        df['Price'] = df['Price'].apply(clean_price)
    if 'App Rating' in df.columns:
        mean_rating = df['App Rating'].mean(skipna=True)
        df['App Rating'] = df['App Rating'].apply(lambda x: clean_rating(x, mean_rating))

    # Drop columns with all NaN values
    df = df.dropna(axis=1, how='all')

    # Fill remaining NaN values with median or mode
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # One-Hot Encode categorical features
    df = pd.get_dummies(df, drop_first=True)
    print(df.head(5))
    return df
