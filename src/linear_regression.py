import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("/files/Project-final/data/clean/master_dataset_fe.csv",
                 parse_dates=["date"])
df = df.sort_values("date")

# Split train / test
split_date = "2019-01-01"
train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]

# Target
y_train = train["electricity_consumption"]
y_test = test["electricity_consumption"]

# Features
feature_cols = [
    "temp", "heat_need", "gdp_real", "population",
    "month", "year",
    "consumption_lag_1", "consumption_lag_12",
    "temp_lag_1", "gdp_lag_1"
]

X_train = train[feature_cols]
X_test = test[feature_cols]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("LINEAR REGRESSION RESULTS (Test )")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.4f}")

# Save predictions
test_results = test.copy()
test_results["prediction_lr"] = y_pred
test_results.to_csv("/files/Project-final/data/clean/linear_regression_predictions.csv", index=False)

print("\nPredictions saved.")
