import pandas as pd

# Path to your population CSV
path = "/files/Project-final/data/Population/pop_data_y.csv"

# Load the CSV
df = pd.read_csv(path, sep=";")

# Keep only the population at January 1st
df = df[df["REFERENCE_POPULATION"] == "POP_JAN"]

# Keep useful columns only
df = df[["YEAR", "VALUE"]].rename(columns={"VALUE": "population"})

# Create a date column (yearly)
df["date"] = pd.to_datetime(df["YEAR"].astype(str) + "-01-01")

# Final yearly dataset
df_yearly = df[["date", "population"]].sort_values("date")

# Save cleaned yearly population file
output_path = "/files/Project-final/data/clean/population_yearly.csv"
df_yearly.to_csv(output_path, index=False)

print("Saved:", output_path)
print(df_yearly.head())