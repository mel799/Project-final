import pandas as pd

def load_gdp():
    """Load and clean Swiss real GDP quarterly data."""

    # Relative path
    path = "data/raw/GDP/GDP_R.csv"

    # Load CSV
    df = pd.read_csv(path, sep=";")

    # Rename columns
    df.columns = ["quarter", "gdp_real"]

    # Remove spaces in numbers and convert to float
    df["gdp_real"] = df["gdp_real"].astype(str).str.replace(" ", "").astype(float)

    # Convert "1980-Q1" → quarterly period
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

    # Convert period to timestamp (end of quarter)
    df["date"] = df["quarter"].dt.to_timestamp()

    # Clean final dataframe
    df_clean = df[["date", "quarter", "gdp_real"]].sort_values("date")

    # Save cleaned CSV
    output_path = "data/clean/SwissGDP.csv"
    df_clean.to_csv(output_path, index=False)

    print(f"Saved GDP data → {output_path}")

    return df_clean
