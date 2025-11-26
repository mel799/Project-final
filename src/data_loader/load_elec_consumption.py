import pandas as pd

def load_consumption():
    """Load and clean monthly Swiss electricity consumption (GWh)."""

    # Relative path to raw file
    path = "data/raw/electricity_consumption/Consommation_F.csv"

    # Read raw CSV
    df = pd.read_csv(path, sep=";")

    # First row contains correct column names
    df.columns = df.iloc[0]    
    df = df[1:].reset_index(drop=True)

    # Standardize column names — rename the consumption column
    df = df.rename(columns={
        "date": "date",
        " GWH": "electricity_consumption",
        "GWH": "electricity_consumption"
    })

    # Convert date "MM-YYYY" to datetime
    df["date"] = pd.to_datetime(df["date"], format="%m-%Y")

    # Convert to numeric
    df["electricity_consumption"] = pd.to_numeric(df["electricity_consumption"], errors="coerce")

    # Sort
    df_clean = df.sort_values("date")

    # Save cleaned version
    output_path = "data/clean/swiss_consumption_monthly.csv"
    df_clean.to_csv(output_path, index=False)

    print(f"Saved electricity consumption data → {output_path}")

    return df_clean
