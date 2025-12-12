import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


def plot_train_test_predictions(
    train_df,
    test_df,
    y_col="electricity_consumption",
    pred_col="prediction",
    model_name="Best Model",
    output_path="results/figures/train_test_predictions.png"
):
    

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # -----------------------------
    # METRICS
    # -----------------------------
    def compute_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    train_rmse, train_mae, train_r2 = compute_metrics(
        train_df[y_col], train_df[pred_col]
    )
    test_rmse, test_mae, test_r2 = compute_metrics(
        test_df[y_col], test_df[pred_col]
    )

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure(figsize=(14, 6))

    # Actual values
    plt.plot(
        train_df["date"], train_df[y_col],
        color="black", linewidth=2, label="Actual (Train)"
    )
    plt.plot(
        test_df["date"], test_df[y_col],
        color="black", linewidth=2, linestyle="--", label="Actual (Test)"
    )

    # Predictions
    plt.plot(
        train_df["date"], train_df[pred_col],
        color="blue", alpha=0.8, label="Prediction (Train)"
    )
    plt.plot(
        test_df["date"], test_df[pred_col],
        color="red", alpha=0.8, label="Prediction (Test)"
    )

    # Vertical split line
    plt.axvline(
        test_df["date"].min(),
        color="gray", linestyle=":", label="Train/Test Split"
    )

    plt.xlabel("Year")
    plt.ylabel("Electricity Consumption (GWh)")
    plt.title(f"{model_name} – Actual vs Predicted Electricity Consumption")
    plt.legend()
    plt.grid(True)

    # -----------------------------
    # METRICS TEXT BOX
    # -----------------------------
    textstr = (
        f"TRAIN\n"
        f"RMSE: {train_rmse:.2f}\n"
        f"MAE : {train_mae:.2f}\n"
        f"R²  : {train_r2:.3f}\n\n"
        f"TEST\n"
        f"RMSE: {test_rmse:.2f}\n"
        f"MAE : {test_mae:.2f}\n"
        f"R²  : {test_r2:.3f}"
    )

    plt.gcf().text(
        0.02, 0.5, textstr,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot to: {output_path}")

