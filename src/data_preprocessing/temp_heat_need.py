import pandas as pd

def add_heat_need(df):
    """Add heat need feature to the temperature dataset."""
    
    # Heat Need = max(0, 18 - temperature)
    df["heat_need"] = (18 - df["temp"]).clip(lower=0)

    # Save the updated temperature file (optional)
    output_path = "data/dataset/temperature_monthly_heatneed.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved temperature with heat need {output_path}")

    return df
