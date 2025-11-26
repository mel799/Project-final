import pandas as pd

def load_population():
    """Load and clean yearly Swiss population data."""

    # ← relative path
    path = "data/raw/Population/pop_data_y.csv"

    # Load the CSV
    df = pd.read_csv(path, sep=";")

    # Keep only January 1st population
    df = df[df["REFERENCE_POPULATION"] == "POP_JAN"]

    # Select useful columns
    df = df[["YEAR", "VALUE"]].rename(columns={"VALUE": "population"})

    # Create date column
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + "-01-01")

    # Final cleaned yearly population dataset
    df_yearly = df[["date", "population"]].sort_values("date")

    # Save cleaned version (optional but useful)
    output_path = "data/clean/population_yearly.csv"
    df_yearly.to_csv(output_path, index=False)

    print(f"Saved population data → {output_path}")

    return df_yearly

