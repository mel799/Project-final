"""
Evaluation metrics for regression models.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def save_scores(score_dict, output_path):
    """
    Save model evaluation results to a text file.
    score_dict = { "Model Name": {"rmse": ..., "mae": ..., "r2": ...}, ... }
    """
    lines = []
    for model_name, metrics in score_dict.items():
        lines.append(f"{model_name} Results:")
        lines.append(f"  RMSE: {metrics['rmse']:.2f}")
        lines.append(f"  MAE : {metrics['mae']:.2f}")
        lines.append(f"  R2  : {metrics['r2']:.4f}")
        lines.append("")  # blank line between models

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Model scores saved to: {output_path}")


def evaluate_baseline(test_df, model_column, y_column="electricity_consumption"):
    """
    Evaluate a baseline model already stored in a test dataframe.
    model_column: the name of the column containing predictions.
    """

    df = test_df.dropna(subset=[model_column]).copy()
    y_true = df[y_column]
    y_pred = df[model_column]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "r2": r2}

def evaluate_linear_regression(test_df, pred_column="prediction_lr",
                               y_column="electricity_consumption"):
    """
    Evaluate a linear regression model.
    """
    df = test_df.dropna(subset=[pred_column]).copy()
    y_true = df[y_column]
    y_pred = df[pred_column]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)


    return {"rmse": rmse, "mae": mae, "r2": r2}

def evaluate_random_forest(test_df, pred_column="prediction_rf",
                           y_column="electricity_consumption"):
    """
    Evaluate a Random Forest regression model.
    """
    df = test_df.dropna(subset=[pred_column]).copy()
    y_true = df[y_column]
    y_pred = df[pred_column]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "r2": r2}

def evaluate_xgboost(test_df, pred_column="prediction_xgb",
                     y_column="electricity_consumption"):
    """
    Evaluate an XGBoost regression model.
    """
    df = test_df.dropna(subset=[pred_column]).copy()
    y_true = df[y_column]
    y_pred = df[pred_column]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "r2": r2}

def evaluate_lasso(test_df, pred_column="prediction_lasso", y_column="electricity_consumption"):
    
    df = test_df.dropna(subset=[pred_column])
    y_true = df[y_column]
    y_pred = df[pred_column]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "r2": r2}

