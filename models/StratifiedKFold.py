from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Custom RMSE scoring function
def rmsee(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def Ridge_regression_KFold(X_train, y_train, X_val, y_val, param_grid=None):
    if param_grid is None:
        param_grid = {
            'alpha': np.linspace(1000, 100000, 10),  # Reduce steps for speed
            'fit_intercept': [True, False],
            'solver': ['auto']  # 'saga' doesn't support Ridge
        }

    rmse_scorer = make_scorer(rmsee, greater_is_better=False)

    # Downcast to float32 to save memory
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Use KFold instead of StratifiedKFold for regression
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search with limited cores
    grid = GridSearchCV(
        estimator=Ridge(),
        param_grid=param_grid,
        scoring=rmse_scorer,
        cv=cv,
        n_jobs=1,  # Prevent memory errors
        verbose=1
    )
    grid.fit(X_train_scaled, y_train)

    best_alpha = grid.best_params_['alpha']
    best_rmse = np.sqrt(-grid.best_score_)

    print(f"Best alpha based on RMSE: {best_alpha} | RMSE: {best_rmse:.4f}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val_scaled)

    final_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"[Final Model] RMSE: {final_rmse:.4f} | R² Score: {r2:.4f}")
    return best_model

def ridge_annealing_search(X_train, y_train, X_val, y_val, alphas=None):
    if alphas is None:
        # Start high, 3/4 alpha each time
        alphas = [2e5 * (0.5 ** i) for i in range(30)]
    rmses = []
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_rmse = float('inf')
    best_alpha = None
    best_model = None

    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        rmses.append(rmse)
        print(f"Alpha: {alpha:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_model = model

    plt.figure(figsize=(8, 4))
    plt.plot(alphas, rmses, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Validation RMSE')
    plt.title('Ridge Alpha Tuning')
    plt.grid(True)
    plt.show()

    print(f"\n Best Alpha: {best_alpha:.4f} | Best RMSE: {best_rmse:.4f}")
    return best_model, scaler

def precise_alpha_search(X_train, y_train, X_val, y_val):
    alphas = np.linspace(1000, 100000, 10)    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    rmses = []
    log_alphas = np.log10(alphas)

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)

    # Fit a quadratic: RMSE = a*(log_alpha)^2 + b*log_alpha + c
    coeffs = np.polyfit(log_alphas, rmses, 2)
    a, b, c = coeffs

    # Vertex (minimum point) of the parabola
    log_alpha_best = -b / (2 * a)
    alpha_best = 10 ** log_alpha_best

    print(f"🎯 Best Alpha (from curve fit): {alpha_best:.4f}")

    # Optionally: refit final model
    final_model = Ridge(alpha=alpha_best)
    final_model.fit(X_train_scaled, y_train)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(alphas, rmses, 'o', label='Observed RMSE')
    alpha_range_fine = np.logspace(np.log10(min(alphas)), np.log10(max(alphas)), 200)
    rmse_fit = np.polyval(coeffs, np.log10(alpha_range_fine))
    plt.plot(alpha_range_fine, rmse_fit, '-', label='Quadratic Fit')
    plt.axvline(alpha_best, color='red', linestyle='--', label=f'Best α ≈ {alpha_best:.1f}')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Validation RMSE')
    plt.title('Quadratic Fit for Optimal Ridge Alpha')
    plt.legend()
    plt.grid(True)
    plt.show()

    return final_model, scaler
