import pandas as pd

# Load the master dataset
df = pd.read_csv("/files/Project-final/data/clean/master_dataset.csv", parse_dates=["date"])

#time features:

# Month number
df["month"] = df["date"].dt.month

# Year
df["year"] = df["date"].dt.year

# Season function
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"

df["season"] = df["month"].apply(get_season)

# lag features:

# Electricity demand lags
df["consumption_lag_1"] = df["electricity_consumption"].shift(1)
df["consumption_lag_12"] = df["electricity_consumption"].shift(12)

# Temperature lag
df["temp_lag_1"] = df["temp"].shift(1)

# GDP lag
df["gdp_lag_1"] = df["gdp_real"].shift(1)

# drop missing values created by lag features
df = df.dropna().reset_index(drop=True)

# Save the enhanced dataset
output_path = "/files/Project-final/data/clean/master_dataset_fe.csv"
df.to_csv(output_path, index=False)

print("Feature engineering completed!")
print("Saved at:", output_path)
print(df.head())
