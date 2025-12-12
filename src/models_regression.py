"""
Regression model training and baseline models.
"""

"""
Regression models: naive baselines + linear regression.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Naive Baseline Models

def naive_last_month(test_df):
    """
    Naive baseline:
    Predict next value = last month's electricity consumption.
    uses: consumption_lag_1
    """
    df = test_df.copy()
    df["naive_last_month"] = df["consumption_lag_1"]
    return df


def naive_last_year(test_df):
    """
    Seasonal naive baseline:
    Predict next value = same month last year.
    uses: consumption_lag_12
    """
    df = test_df.copy()
    df["naive_last_year"] = df["consumption_lag_12"]
    return df

# Linear Regression Model

def train_linear_regression(train_df, feature_cols):
    """
    Train a Linear Regression model using the selected features.
    Returns the trained model and the X_train, y_train used internally.
    """

    # Target
    y_train = train_df["electricity_consumption"]

    # Features
    X_train = train_df[feature_cols]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def predict_linear_regression(model, test_df, feature_cols):
    """
    Apply a trained Linear Regression model to the test set.
    Returns a copy of the test_df with predictions added.
    """

    df = test_df.copy()
    X_test = df[feature_cols]

    df["prediction_lr"] = model.predict(X_test)
    return df

# MACHINE LEARNING MODELS
# ---------------------------------------------------------
# 3. Random Forest Regressor

def train_random_forest(train_df, feature_cols, n_estimators=300, max_depth=None, random_state=42):
    """
    Train a Random Forest Regressor.
    """

    y_train = train_df["electricity_consumption"]
    X_train = train_df[feature_cols]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    return model


def predict_random_forest(model, test_df, feature_cols):
    """
    Apply the trained Random Forest model and add predictions.
    """

    df = test_df.copy()
    X_test = df[feature_cols]

    df["prediction_rf"] = model.predict(X_test)
    return df

# 4. XGBoost Regressor

def train_xgboost(train_df, feature_cols, random_state=42):
    """
    Train an XGBoost regressor on the training set.
    """

    y_train = train_df["electricity_consumption"]
    X_train = train_df[feature_cols]

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model
    

# XGBoost with GridSearchCV

def predict_xgboost(model, test_df, feature_cols):
    """
    Apply trained XGBoost model and add predictions as 'prediction_xgb'.
    """

    df = test_df.copy()
    X_test = df[feature_cols]
    df["prediction_xgb"] = model.predict(X_test)
    return df

def train_xgboost_gridsearch(train_df, feature_cols, cv=3):
    """
    Run GridSearchCV to tune XGBoost hyperparameters.
    Returns the best model and the GridSearch object.
    """

    y_train = train_df["electricity_consumption"]
    X_train = train_df[feature_cols]

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )

    param_grid = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        verbose=1
    )

    print("\nRunning XGBoost GridSearchCV...")
    grid.fit(X_train, y_train)

    print("\n=== BEST XGBOOST PARAMETERS ===")
    print(grid.best_params_)
    print(f"Best RMSE (CV): {(-grid.best_score_) ** 0.5:.2f}")

    best_model = grid.best_estimator_
    return best_model, grid


# Lasso Regression model
def train_lasso(train_df, feature_cols, alpha=0.1):
    """
    Train a LASSO regression model using L1 regularization.
    """
    y_train = train_df["electricity_consumption"]
    X_train = train_df[feature_cols]

    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(X_train, y_train)

    return model

    
def predict_lasso(model, test_df, feature_cols):
    """
    Apply LASSO model to test data.
    """
    df = test_df.copy()
    X_test = df[feature_cols]

    df["prediction_lasso"] = model.predict(X_test)
    return df

def train_lasso_gridsearch(train_df, feature_cols):
    """
    Tune alpha for LASSO using GridSearchCV.
    Returns the best model and the GridSearchCV object.
    """

    y_train = train_df["electricity_consumption"]
    X_train = train_df[feature_cols]

    # Range of alpha values to test
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
    }

    lasso = Lasso(max_iter=5000)

    grid = GridSearchCV(
        estimator=lasso,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # MSE (lower is better)
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\n=== LASSO GRID SEARCH RESULTS ===")
    print("Best alpha:", grid.best_params_["alpha"])
    print("Best CV RMSE:", (-grid.best_score_) ** 0.5)

    return best_model, grid

