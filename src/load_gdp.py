import pandas as pd
import os

path ="/files/Project-final/data/GDP/GDP_R.csv"

# 1. Load the CSV (separator is ;)
df = pd.read_csv(path, sep=";")

# 2. Rename columns
df.columns = ["quarter", "gdp_real"]

# 3. Remove spaces in GDP column and convert to float
df["gdp_real"] = df["gdp_real"].astype(str).str.replace(" ", "").astype(float)

# 4. Convert "1980-Q1" → quarterly period
df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

# 5. Convert period to a timestamp (end of quarter)
df["date"] = df["quarter"].dt.to_timestamp()

# 6. Final cleaned dataframe
df = df[["date", "quarter", "gdp_real"]]

# Show first rows
print(df.head())

# Save clean CSV
output_dir = "/files/Project-final/data/clean"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/SwissGDP.csv"

df.to_csv(output_path, index=False)

print("CLEAN CSV SAVED TO:", output_path)
print(df.head())
print(df.tail())