# Best based on Alpha: 5000.0 | RMSE: 0.4394 
# 3ala kaggle: 0.24702
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def Ridge_regression(X_train, y_train, X_val, y_val, param_grid=None):
    # Default param grid if none is provided
    if param_grid is None:
        param_grid = {'alpha': [10.0, 100.0, 500.0 ,1000.0,1500.0 , 5000.0, 10000.0]}

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize Ridge Regression
    ridge = Ridge()

    # Initialize a list to store RMSE for each alpha
    alpha_rmses = []

    # Perform Grid Search with Cross-Validation manually
    for alpha in param_grid['alpha']:
        # Fit the model with the current alpha
        ridge.set_params(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)

        # Predict on the validation set
        y_pred = ridge.predict(X_val_scaled)

        # Calculate RMSE for this model
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        alpha_rmses.append((alpha, rmse))
        print(f"Alpha: {alpha} | RMSE: {rmse:.4f}")

    # Find the alpha with the lowest RMSE
    best_alpha, best_rmse = min(alpha_rmses, key=lambda x: x[1])
    print(f"Best alpha based on RMSE: {best_alpha} | RMSE: {best_rmse:.4f}")

    # Train the final model with the best alpha
    best_model = Ridge(alpha=best_alpha)
    best_model.fit(X_train_scaled, y_train)

    # Predict on the validation set using the best model
    y_pred = best_model.predict(X_val_scaled)

    # Calculate final metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    # Log results
    print(f"[Improved Linear Regression] RMSE: {rmse:.4f} | R² Score: {r2:.4f}")
    return best_model
