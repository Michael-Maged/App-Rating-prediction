# models/improved_linear_model.py
# RMSE: 0.4616
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def Ridge_regression(X_train, y_train, X_val, y_val, alpha=1.0):

    # Initialize Ridge Regression with regularization
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    # Log results
    print(f"[Improved Linear Regression] RMSE: {rmse:.4f} | R² Score: {r2:.4f}")

    return model