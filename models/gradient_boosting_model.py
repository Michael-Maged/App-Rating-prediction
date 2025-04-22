# models/gradient_boosting_model.py
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint
import numpy as np

# RMSE: 0.4292
# 3ala kaggle: 0.25177
def train_gradient_boosting(X_train, y_train, X_val, y_val):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Optional: Evaluate on validation set here
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"Gradient Boosting RMSE: {rmse:.4f}")
    print(f"Gradient Boosting R² Score: {r2:.4f}")

    return model

# RMSE: 
# 3ala kaggle:
def train_gradient_boosting_with_gridsearchcv(X_train, y_train, X_val, y_val):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }

    # Initialize the Gradient Boosting Regressor
    model = GradientBoostingRegressor(random_state=42)

    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
    )
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    y_pred = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print("GridSearchCV Best Parameters:", grid_search.best_params_)
    print(f"GridSearchCV Gradient Boosting RMSE: {rmse:.4f}")
    print(f"GridSearchCV Gradient Boosting R² Score: {r2:.4f}")

    return best_model

# RMSE: 
# 3ala kaggle:
def train_gradient_boosting_with_randomizedsearchcv(X_train, y_train, X_val, y_val):
    
    param_dist = {
        'n_estimators': randint(100, 400),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'max_features': ['sqrt', 'log2', None]
    }
    
    model = GradientBoostingRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=30,  # number of random combinations to try
        cv=3,
        verbose=1,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        random_state=42
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    
    # Evaluate on validation set
    y_pred = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print("RandomizedSearchCV Best Parameters:", random_search.best_params_)
    print(f"RandomizedSearchCV Gradient Boosting RMSE: {rmse:.4f}")
    print(f"RandomizedSearchCV Gradient Boosting R² Score: {r2:.4f}")
    
    
    return best_model

# RMSE: 0.4144
# 3ala kaggle: 0.27651
def train_gradient_boosting_with_earlystopping(X_train, y_train, X_val, y_val):

    model = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.01,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.2,
        max_depth=7,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"Early Stopping Gradient Boosting RMSE: {rmse:.4f}")
    print(f"Early Stopping Gradient Boosting R² Score: {r2:.4f}")

    return model