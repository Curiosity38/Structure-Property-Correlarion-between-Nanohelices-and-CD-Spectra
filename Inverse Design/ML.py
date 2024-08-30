import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.signal import savgol_filter
import lightgbm as lgb
import xgboost as xgb

# Load data
train_data = pd.read_csv('../data/cd_train_0603.csv')
test_data = pd.read_csv('../data/cd_test_0603.csv')

# Features for X and y
feature_Y = [str(i) for i in range(120)]
feature_X = ['pitch', 'fiber_radius', 'n_turns', 'helix_radius']

# Preprocess data
X_train = train_data[feature_Y].values
y_train = train_data[feature_X].values

X_test = test_data[feature_Y].values
y_test = test_data[feature_X].values

# Apply Savitzky-Golay filter to smooth the data
window_length = 15  # 滤波器窗口长度
polyorder = 3  # 多项式拟合阶数

X_train = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=X_train)
X_test = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=X_test)

# Standardize the data
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Initialize models and parameter grids
models = {
    'Linear Regression': (LinearRegression(), {}),
    'Lasso': (Lasso(), {'alpha': [0.01, 0.1, 1, 10]}),
    'Random Forest': (RandomForestRegressor(random_state=42), {'n_estimators': [1000, 3000, 5000], 'max_depth': [7, 10, 15]}),
    #'SVR': (SVR(), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42), {'n_estimators': [1000, 3000, 5000], 'learning_rate': [0.01, 0.1], 'max_depth': [7, 10, 15]}),
    'LightGBM': (lgb.LGBMRegressor(random_state=42), {'n_estimators': [1000, 3000, 5000], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50, 100]}),
    'XGBoost': (xgb.XGBRegressor(random_state=42), {'n_estimators': [1000, 3000, 5000], 'learning_rate': [0.01, 0.1], 'max_depth': [7, 10, 15]})
}

# Train and evaluate models for each target variable
results = {}

for i, target in enumerate(feature_X):
    y_train_target = y_train[:, i]
    y_test_target = y_test[:, i]

    scaler_y = StandardScaler()
    y_train_target = scaler_y.fit_transform(y_train_target.reshape(-1, 1)).ravel()
    y_test_target = scaler_y.transform(y_test_target.reshape(-1, 1)).ravel()

    results[target] = {}

    for name, (model, param_grid) in models.items():
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train_target)
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)

        # Calculate percentage errors
        absolute_errors = np.abs(predictions - y_test_target)
        percentage_errors = absolute_errors / np.abs(y_test_target) * 100
        mean_percentage_error = np.mean(percentage_errors)

        results[target][name] = mean_percentage_error

        print(f"Best parameters for {name} on {target}: {grid_search.best_params_}")

# Print results
for target in feature_X:
    print(f"\nResults for {target}:")
    for model_name in models.keys():
        print(f"{model_name}: {results[target][model_name]:.2f}%")

# If you want to see the overall results in a DataFrame
results_df = pd.DataFrame(results)
print("\nSummary of results (Mean Percentage Error):")
print(results_df)
results_df.to_csv("ML_results.csv")
