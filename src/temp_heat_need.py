import pandas as pd

# Path to your cleaned monthly temperature file
path = "/files/Project-final/data/clean/temperature_monthly.csv"

# Load monthly temperature data
df = pd.read_csv(path, parse_dates=["date"])

# Compute heat need (formerly HDD)
df["heat_need"] = (18 - df["temp"]).clip(lower=0)

# Save updated file
output_path = "/files/Project-final/data/clean/temperature_monthly_heatneed.csv"
df.to_csv(output_path, index=False)

print("Saved:", output_path)
print(df.head())