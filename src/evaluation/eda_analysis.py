import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

plt.style.use("ggplot")


def run_eda(df, do_clustering=False, verbose=False):
    """
    Perform EDA on the merged dataset.
    Saves all plots AND a text summary into results/eda/.
    If verbose=False → nothing prints in the terminal.
    """

    output_dir = "results/eda"
    os.makedirs(output_dir, exist_ok=True)

    # Store EDA text results in a file instead of printing
    with open(f"{output_dir}/summary.txt", "w") as f:

        f.write("=== EDA SUMMARY ===\n\n")

        # ---------------------------------------------
        # BASIC INFO
        # ---------------------------------------------
        f.write("DATA INFO:\n")
        df.info(buf=f)
        f.write("\n\n")

        f.write("MISSING VALUES:\n")
        f.write(str(df.isna().sum()))
        f.write("\n\n")

        f.write("SUMMARY STATISTICS:\n")
        f.write(str(df.describe()))
        f.write("\n\n")

    # ---------------------------------------------
    # TIME SERIES
    # ---------------------------------------------
    plt.figure(figsize=(14,5))
    plt.plot(df["date"], df["electricity_consumption"])
    plt.title("Swiss Electricity Consumption Over Time (GWh)")
    plt.xlabel("Year")
    plt.ylabel("Consumption (GWh)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series_consumption.png")
    plt.close()

    # ---------------------------------------------
    # SEASONALITY
    # ---------------------------------------------
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly_avg = df.groupby("month")["electricity_consumption"].mean()

    plt.figure(figsize=(12,5))
    plt.plot(monthly_avg.index, monthly_avg.values)
    plt.xticks(range(1,13))
    plt.title("Average Electricity Consumption by Month (Seasonality)")
    plt.xlabel("Month")
    plt.ylabel("Average GWh")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_seasonality.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(12,6))
    df.boxplot(column="electricity_consumption", by="month")
    plt.title("Month-to-Month Variability")
    plt.suptitle("")
    plt.xlabel("Month")
    plt.ylabel("Consumption (GWh)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_boxplot.png")
    plt.close()

    # ---------------------------------------------
    # CORRELATION MATRIX
    # ---------------------------------------------
    corr = df[["electricity_consumption", "year", "month"]].corr()

    plt.figure(figsize=(6,4))
    plt.imshow(corr, cmap="coolwarm", interpolation="none")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()

    # ---------------------------------------------
    # OPTIONAL CLUSTERING
    # ---------------------------------------------
    if do_clustering:

        X = df[["electricity_consumption"]].values
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        df["cluster"] = kmeans.labels_

        plt.figure(figsize=(14,5))
        plt.scatter(
            df["date"], df["electricity_consumption"],
            c=df["cluster"], cmap="viridis", s=20
        )
        plt.title("K-means Clustering of Monthly Consumption")
        plt.xlabel("Date")
        plt.ylabel("GWh")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/kmeans_clusters.png")
        plt.close()

        # Save cluster means into summary file
        with open(f"{output_dir}/summary.txt", "a") as f:
            f.write("\nCLUSTER MEANS:\n")
            f.write(str(df.groupby("cluster")["electricity_consumption"].mean()))


    if verbose:
        print("EDA completed. Results saved in results/eda/")
