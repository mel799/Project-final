import pandas as pd
path = "/files/Project-final/data/temp/climate_Swiss.txt"

# Read the file correctly:
df = pd.read_csv(
    path,
    sep=r"\s+",        # split on any whitespace
    skiprows=15,       # skip metadata BEFORE the real header
    na_values="NA",    # handle missing values
    engine="python"
)

# Keep only yearly + monthly columns
monthly_cols = ["time","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
df = df[monthly_cols]

# Convert wide → long format
df_long = df.melt(id_vars="time", var_name="month", value_name="temp")

# Map month names to numbers
month_numbers = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                 "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
df_long["month_num"] = df_long["month"].map(month_numbers)

# Create date YYYY-MM
df_long["date"] = pd.to_datetime(df_long["time"].astype(str) + "-" +
                                 df_long["month_num"].astype(str) + "-01")

# Final cleaned dataset
df_clean = df_long[["date","temp"]].sort_values("date")

# Save
output_path = "/files/Project-final/data/clean/temperature_monthly.csv"
df_clean.to_csv(output_path, index=False)

print("Saved:", output_path)
