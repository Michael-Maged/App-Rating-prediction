import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_distributions(df):
    numerical_cols = df.select_dtypes(include='number').columns
    for col in numerical_cols:
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

def plot_correlation_matrix(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Predicted vs Actual Ratings")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # diagonal line
    plt.tight_layout()
    plt.show()
    
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.show()

def plot_feature_vs_rating(df, feature_col, rating_col='App Rating'):
    """
    Visualizes a feature against the target rating using a scatter plot.
    Converts object-type columns to numeric if possible.

    Parameters:
    - df: DataFrame containing the data.
    - feature_col: The feature column to plot.
    - rating_col: The target column (default is 'App Rating').

    Returns:
    - None
    """
    if feature_col not in df.columns:
        print(f"Column '{feature_col}' not found in the DataFrame.")
        return


    # Check if the column is numeric after conversion
    if not pd.api.types.is_numeric_dtype(df[feature_col]):
        print(f"Column '{feature_col}' is not numeric and cannot be plotted as a scatter plot.")
        return

    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[feature_col], y=df[rating_col], alpha=0.6)
    plt.xlabel(feature_col)
    plt.ylabel(rating_col)
    plt.title(f'Scatter Plot: {feature_col} vs {rating_col}')
    plt.tight_layout()
    plt.show()