# ============================================
# 1. IMPORTS
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Use a safe default style
plt.style.use("ggplot")

# ============================================
# 2. LOAD DATA
# ============================================
df = pd.read_csv("/files/Project-final/data/clean/swiss_consumption_monthly.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

print(df.info())
print(df.head())

# ============================================
# 3. BASIC CHECKS
# ============================================
print("\nMissing values:\n", df.isna().sum())
print("\nSummary statistics:\n", df.describe())

# ============================================
# 4. TIME SERIES VISUALIZATION
# ============================================
plt.figure(figsize=(14,5))
plt.plot(df["date"], df["electricity_consumption"])
plt.title("Swiss Electricity Consumption Over Time (GWh)")
plt.xlabel("Year")
plt.ylabel("Consumption (GWh)")
plt.show()

# ============================================
# 5. SEASONALITY ANALYSIS
# ============================================
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Average consumption per month
monthly_avg = df.groupby("month")["electricity_consumption"].mean()

plt.figure(figsize=(12,5))
plt.plot(monthly_avg.index, monthly_avg.values)
plt.xticks(range(1,13))
plt.title("Average Electricity Consumption by Month (Seasonality)")
plt.xlabel("Month")
plt.ylabel("Average GWh")
plt.show()

# Boxplot by month
plt.figure(figsize=(12,6))
df.boxplot(column="electricity_consumption", by="month")
plt.title("Month-to-Month Variability")
plt.suptitle("") 
plt.xlabel("Month")
plt.ylabel("Consumption (GWh)")
plt.show()

# ============================================
# 6. CORRELATION ANALYSIS
# ============================================
corr = df[["electricity_consumption", "year", "month"]].corr()

plt.figure(figsize=(6,4))
plt.imshow(corr, cmap="coolwarm", interpolation="none")
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix")
plt.show()

# ============================================
# 7. OPTIONAL: K-MEANS CLUSTERING
# ============================================
X = df[["electricity_consumption"]].values

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
df["cluster"] = kmeans.labels_

plt.figure(figsize=(14,5))
plt.scatter(df["date"], df["electricity_consumption"], c=df["cluster"], cmap="viridis", s=20)
plt.title("K-means Clustering of Monthly Consumption")
plt.xlabel("Date")
plt.ylabel("GWh")
plt.show()

print("\nCluster Means:\n", df.groupby("cluster")["electricity_consumption"].mean())
