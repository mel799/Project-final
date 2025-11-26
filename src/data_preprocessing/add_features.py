import pandas as pd

def add_features(df):
    """Add time-based, seasonal, and lag features to the merged dataset."""

    # Time features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # Season feature
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

    # Lag features
    df["consumption_lag_1"] = df["electricity_consumption"].shift(1)
    df["consumption_lag_12"] = df["electricity_consumption"].shift(12)

    df["temp_lag_1"] = df["temp"].shift(1)
    df["gdp_lag_1"] = df["gdp_real"].shift(1)

    # Remove rows with missing lag values
    df = df.dropna().reset_index(drop=True)

    # Save engineered dataset
    output_path = "data/dataset/master_dataset_fe.csv"
    df.to_csv(output_path, index=False)

    print(f"Feature engineering completed  {output_path}")

    return df
