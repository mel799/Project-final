import pandas as pd

# Load Electricity (date, consumption)
df_elec = pd.read_csv("/files/Project-final/data/clean/swiss_consumption_monthly.csv",
                      parse_dates=["date"])

# Load Temperature (date, temp, HDD)
df_temp = pd.read_csv("/files/Project-final/data/clean/temperature_monthly_heatneed.csv",
                      parse_dates=["date"])

# GDP (date, gdp_real)
df_gdp = pd.read_csv("/files/Project-final/data/clean/SwissGDP_monthly.csv",
                     parse_dates=["date"])

# Population (must contain: date, population)
df_pop = pd.read_csv("/files/Project-final/data/clean/population_monthly.csv",
                     parse_dates=["date"])

# === SORT ALL DATASETS BY DATE ===
df_elec = df_elec.sort_values("date")
df_temp = df_temp.sort_values("date")
df_gdp = df_gdp.sort_values("date")
df_pop = df_pop.sort_values("date")

# === MERGE EVERYTHING ===
df_final = (
    df_elec
    .merge(df_temp, on="date", how="left")
    .merge(df_gdp, on="date", how="left")
    .merge(df_pop, on="date", how="left")
)

# === OPTIONAL: REMOVE NA IF ANY ===
df_final = df_final.dropna()

# === SAVE MASTER DATASET ===
output_path = "/files/Project-final/data/clean/master_dataset.csv"
df_final.to_csv(output_path, index=False)

print("Master dataset saved to:", output_path)
print(df_final.head())
print("Shape:", df_final.shape)
