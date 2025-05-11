# models/gradient_boosting_model.py
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# RMSE: 0.4281, R^2 0.0761
def train_gradient_boosting(X_train, y_train, X_val, y_val):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"Gradient Boosting RMSE: {rmse:.4f}")
    print(f"Gradient Boosting R² Score: {r2:.4f}")

    return model
