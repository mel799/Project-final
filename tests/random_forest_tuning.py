import sys
from pathlib import Path
import pandas as pd

# Add project root so Python can find src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models_regression import (
    train_random_forest, predict_random_forest
)
from src.models_evaluation import evaluate_random_forest

# Load engineered dataset
df = pd.read_csv("./data/dataset/master_dataset_fe.csv", parse_dates=["date"])
df = df.sort_values("date")

# Split train / test
split_date = "2019-01-01"
train = df[df["date"] < split_date].copy()
test_base = df[df["date"] >= split_date].copy()

# Features used
feature_cols = [
    "temp", "heat_need", "gdp_real", "population",
    "month", "year",
    "consumption_lag_1", "consumption_lag_12",
    "temp_lag_1", "gdp_lag_1"
]

# Hyperparameter values to test
n_values = [50, 100, 200, 300, 500, 800]

scores_rf = {}

for n in n_values:
    print(f"\n=== Testing Random Forest with {n} trees ===")

    # Copy test df (because predictions will be added)
    test = test_base.copy()

    # Train + Predict
    model = train_random_forest(train, feature_cols, n_estimators=n)
    test = predict_random_forest(model, test, feature_cols)

    # Evaluate
    metrics = evaluate_random_forest(test)
    scores_rf[n] = metrics

# Summary
print("\n=== Random Forest Comparison ===")
for n, m in scores_rf.items():
    print(f"{n} trees → RMSE: {m['rmse']:.2f}, MAE: {m['mae']:.2f}, R²: {m['r2']:.3f}")

#run python3 tests/random_forest_tuning.py