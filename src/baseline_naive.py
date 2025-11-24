import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load featured dataset
df = pd.read_csv(
    "/files/Project-final/data/clean/master_dataset_fe.csv",
    parse_dates=["date"]
)

# Sort it by time
df = df.sort_values("date")

# Define train/test split 80% / 20%
split_date = "2019-01-01"

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

print("Train period:", train["date"].min(), "→", train["date"].max())
print("Test period: ", test["date"].min(), "→", test["date"].max())
print("Train shape:", train.shape)
print("Test shape: ", test.shape)

# Build naive forecasts on the test set

# Naive 1: Last month (uses consumption_lag_1)
test["naive_last_month"] = test["consumption_lag_1"]

# Naive 2: Same month last year (seasonal naive, uses consumption_lag_12)
test["naive_last_year"] = test["consumption_lag_12"]

# Remove rows where lags are missing 
test_eval = test.dropna(subset=["naive_last_month", "naive_last_year"])

# Compute error metrics

y_true = test_eval["electricity_consumption"]

rmse_last_month = np.sqrt(mean_squared_error(y_true, test_eval["naive_last_month"]))
mae_last_month = mean_absolute_error(y_true, test_eval["naive_last_month"])
r2_last_month = r2_score(y_true, test_eval["naive_last_month"])


rmse_last_year = np.sqrt(mean_squared_error(y_true, test_eval["naive_last_year"]))
mae_last_year = mean_absolute_error(y_true, test_eval["naive_last_year"])
r2_last_year = r2_score(y_true, test_eval["naive_last_year"])   

print("\n=== BASELINE NAIVE MODELS (evaluated on TEST set) ===")
print(f"Naive Last Month  → RMSE: {rmse_last_month:.2f}, MAE: {mae_last_month:.2f}, R²: {r2_last_month:.3f}")
print(f"Naive Last Year   → RMSE: {rmse_last_year:.2f}, MAE: {mae_last_year:.2f}, R²: {r2_last_year:.3f}")

# 5. Save predictions for plotting / report
output_path = "/files/Project-final/data/clean/baseline_predictions.csv"
cols_to_save = [
    "date",
    "electricity_consumption",
    "naive_last_month",
    "naive_last_year"
]
test_eval[cols_to_save].to_csv(output_path, index=False)

print("\nBaseline predictions saved to:", output_path)
