import pandas as pd

# Path to your yearly population file
path ="/files/Project-final/data/clean/population_yearly.csv"

# Load the yearly population data
df = pd.read_csv(path, parse_dates=["date"])

# Set the date as index
df = df.set_index("date")

# Convert yearly → monthly using forward-fill
df_monthly = df.resample("MS").ffill()

# Save monthly version
output_path = "/files/Project-final/data/clean/population_monthly.csv"
df_monthly.to_csv(output_path)

print("Saved:", output_path)
print(df_monthly.head(12))