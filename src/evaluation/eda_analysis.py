import pandas as pd
import matplotlib.pyplot as plt
import os


def run_eda(df):
    """
    Perform Exploratory Data Analysis (EDA)
    Outputs:
    - results/eda/summary.txt
    - results/eda/seasonality.png
    - results/eda/electricity_consumption_time_series.png
    """

    output_dir = "results/eda"
    os.makedirs(output_dir, exist_ok=True)

    # ==================================================
    # TEXT SUMMARY
    # ==================================================
    
    with open(f"{output_dir}/summary.txt", "w") as f:

        f.write("=== EXPLORATORY DATA ANALYSIS SUMMARY ===\n\n")

        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write(f"Number of observations: {len(df)}\n")
        f.write(f"Time span: {df['date'].min().date()} → {df['date'].max().date()}\n\n")

        # Missing values
        f.write("MISSING VALUES\n")
        f.write(str(df.isna().sum()))
        f.write("\n\n")

        # Summary statistics
        f.write("SUMMARY STATISTICS (Electricity Consumption)\n")
        f.write(str(df["electricity_consumption"].describe()))
        f.write("\n\n")
        
        # Seasonality
        df["month"] = df["date"].dt.month
        monthly_avg = df.groupby("month")["electricity_consumption"].mean()

        f.write("SEASONALITY INSIGHT\n")
        f.write("Average electricity consumption by month:\n")
        f.write(str(monthly_avg))
        f.write("\n\n")

        f.write(
            "Interpretation:\n"
            "- Strong seasonal pattern with higher consumption in winter.\n"
            "- This justifies the use of monthly features, seasonal variables,\n"
            "  heating degree days, and lag-12 features.\n\n"
        )
    

        # --------------------------------------------------
        # VARIANCE 
        # --------------------------------------------------
        numeric_cols = [
            "electricity_consumption",
            "temp",
            "heat_need",
            "gdp_real",
            "population"
        ]

        f.write("VARIANCE OF NUMERIC VARIABLES\n")
        f.write(str(df[numeric_cols].var()))
        f.write("\n\n")

        # --------------------------------------------------
        # CORRELATION MATRIX
        # --------------------------------------------------
        f.write("CORRELATION MATRIX\n")
        corr_matrix = df[numeric_cols].corr()
        f.write(str(corr_matrix))
        f.write("\n\n")


    # ==================================================
    # PLOT: SEASONALITY
    # ==================================================
    plt.figure(figsize=(10, 4))
    plt.plot(monthly_avg.index, monthly_avg.values, marker="o")
    plt.xticks(range(1, 13))
    plt.xlabel("Month")
    plt.ylabel("Average Electricity Consumption (GWh)")
    plt.title("Average Monthly Electricity Consumption (Seasonality)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seasonality.png")
    plt.close()
    
        # ==================================================
    # TIME SERIES OF ELECTRICITY CONSUMPTION (EDA)
    # ==================================================
    plt.figure(figsize=(12, 4))
    plt.plot(df["date"], df["electricity_consumption"], color="black")
    plt.xlabel("Year")
    plt.ylabel("Electricity Consumption (GWh)")
    plt.title("Swiss Electricity Consumption Over Time")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/electricity_consumption_time_series.png")
    plt.close()

