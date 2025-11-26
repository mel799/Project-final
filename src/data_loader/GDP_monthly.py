import pandas as pd

def make_gdp_monthly():
    """Convert quarterly GDP data to monthly using forward fill."""

    # Relative path to cleaned quarterly GDP
    path = "data/clean/SwissGDP.csv"

    # Load quarterly GDP
    df = pd.read_csv(path, parse_dates=["date"])

    # Set the date as index
    df = df.set_index("date")

    # Quarterly → Monthly (Month Start frequency)
    df_monthly = df.resample("MS").ffill()

    # Reset index back to a column
    df_monthly = df_monthly.reset_index()

    # Save the monthly version
    output_path = "data/clean/SwissGDP_monthly.csv"
    df_monthly.to_csv(output_path, index=False)

    print(f"Saved monthly GDP: {output_path}")

    return df_monthly
