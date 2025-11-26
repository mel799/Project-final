import pandas as pd

def make_population_monthly():
    """Convert yearly population data to monthly."""

    path = "data/clean/population_yearly.csv"

    # Load the yearly population data
    df = pd.read_csv(path, parse_dates=["date"])

    # Set date as index
    df = df.set_index("date")

    # Convert yearly → monthly (Month Start frequency)
    df_monthly = df.resample("MS").ffill()

    # Reset index
    df_monthly = df_monthly.reset_index()

    # Save monthly file
    output_path = "data/clean/population_monthly.csv"
    df_monthly.to_csv(output_path, index=False)

    print(f"Saved monthly population {output_path}")

    return df_monthly
