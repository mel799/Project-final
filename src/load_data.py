import pandas as pd
import os

path ="/files/Project-final/data/electricity_consumption/Consommation_F.csv"


df = pd.read_csv(path, sep=";")

print("Original columns:", df.columns)

# The real header is the first row of data
df.columns = df.iloc[0]  # row 0 becomes header
df = df[1:]              # drop header row from data
df = df.reset_index(drop=True)

print("Fixed columns:", df.columns)
print(df.head())

# Rename columns to standard names
df = df.rename(columns={"date": "date", " GWH": "GWH", "GWH": "GWH"})

# Convert date MM-YYYY → datetime
df["date"] = pd.to_datetime(df["date"], format="%m-%Y")

# Convert GWH to numeric
df["GWH"] = pd.to_numeric(df["GWH"], errors="coerce")

# Sort
df = df.sort_values("date")

# Save clean CSV
output_dir = "/files/Project-final/data/clean"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/swiss_consumption_monthly.csv"

df.to_csv(output_path, index=False)

print("CLEAN CSV SAVED TO:", output_path)
print(df.head())
print(df.tail())