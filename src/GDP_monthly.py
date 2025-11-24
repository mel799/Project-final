import pandas as pd

path = "/files/Project-final/data/clean/SwissGDP.csv"

# Load the CSV
df = pd.read_csv(path, parse_dates=["date"])

# Set date as index
df = df.set_index("date")

# Convert quarterly → monthly using forward-fill
df_monthly = df.resample("MS").ffill()

# Save monthly version
output_path = "/files/Project-final/data/clean/SwissGDP_monthly.csv"
df_monthly.to_csv(output_path)

print("Saved:", output_path)
print(df_monthly.head(6))
