# Best based on Alpha: 5000.0 | RMSE: 0.4394 
# 3ala kaggle: 0.24702
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.model_selection import GridSearchCV

def Ridge_regression(X_train, y_train, X_val, y_val, param_grid=None):
    if param_grid is None:
        param_grid = {'alpha': np.linspace(1000, 100000, 50)}

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid={'alpha': param_grid['alpha']}, scoring='neg_mean_squared_error', cv=5, n_jobs=2)
    grid.fit(X_train_scaled, y_train)

    best_alpha = grid.best_params_['alpha']
    best_rmse = np.sqrt(-grid.best_score_)

    print(f"Best alpha based on RMSE: {best_alpha} | RMSE: {best_rmse:.4f}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val_scaled)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"[Improved Linear Regression] RMSE: {rmse:.4f} | R² Score: {r2:.4f}")
    return best_model
