import pandas as pd

def merge_all(df_temp, df_pop, df_gdp, df_elec):
    """Merge temperature, population, GDP, and electricity consumption into one dataset."""

    # Sort all dataframes by date
    df_temp = df_temp.sort_values("date")
    df_pop  = df_pop.sort_values("date")
    df_gdp  = df_gdp.sort_values("date")
    df_elec = df_elec.sort_values("date")

    # Merge step by step
    df_final = (
        df_elec
        .merge(df_temp, on="date", how="left")
        .merge(df_gdp, on="date", how="left")
        .merge(df_pop, on="date", how="left")
    )

    # Optional: drop NA rows created by missing data
    df_final = df_final.dropna().reset_index(drop=True)

    # Save merged dataset
    output_path = "data/dataset/master_dataset.csv"
    df_final.to_csv(output_path, index=False)

    print(f"Master dataset saved  {output_path}")

    return df_final
