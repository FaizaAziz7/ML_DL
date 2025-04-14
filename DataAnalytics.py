# TASK 1 : FINDING MEAN OF GROUPED DATA

class_intervals = [(0, 10), (10, 20), (20, 30), (30, 40)]
frequencies = [5, 8, 15, 7]
midpoints = [(lower + upper) / 2 for lower, upper in class_intervals]
fx = [f * x for f, x in zip(frequencies, midpoints)]
mean = sum(fx) / sum(frequencies)
print(f"Mean of grouped data: {mean}")




# APPLYING DIFFERENT FUNCTIONS ON DATA SET

import pandas as pd
import numpy as np
from scipy import stats

# Load data from CSV
df = pd.read_csv("shampoo_sales.csv")
# Descriptive statistics
mean = df["Sales"].mean()
median = df["Sales"].median()
mode = df["Sales"].mode().tolist()
variance = df["Sales"].var()
std_dev = df["Sales"].std()
cv = (std_dev / mean) * 100
skewness = stats.skew(df["Sales"])
kurtosis = stats.kurtosis(df["Sales"])
z_scores = stats.zscore(df["Sales"])
percentiles = {
    "25th Percentile": np.percentile(df["Sales"], 25),
    "50th Percentile (Median)": np.percentile(df["Sales"], 50),
    "75th Percentile": np.percentile(df["Sales"], 75),
}
correlation = df["Sales"].corr(pd.Series(range(len(df))))

# Print results
print("----- Descriptive Statistics -----")
print(f"Mean: {mean:.2f}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Coefficient of Variation: {cv:.2f}%")
print(f"Skewness: {skewness:.2f}")
print(f"Kurtosis: {kurtosis:.2f}")
print(f"Correlation Coefficient (with Time): {correlation:.2f}")

print("\n----- Percentiles & Quartiles -----")
for k, v in percentiles.items():
    print(f"{k}: {v}")

print("\nZ-scores:")
print(z_scores)





import matplotlib.pyplot as plt
import seaborn as sns

# HISTOGRAM
plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], kde=True, color='skyblue', bins=10)
plt.title("Histogram of Shampoo Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# BOX PLOT
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Sales"], color='lightgreen')
plt.title("Boxplot of Shampoo Sales")
plt.xlabel("Sales")
plt.show()

# LINE PLOT
plt.figure(figsize=(10, 5))
plt.plot(df["Sales"], marker='o', linestyle='-', color='orange')
plt.title("Shampoo Sales Over Time")
plt.xlabel("Time (Index)")
plt.ylabel("Sales")
plt.grid(True)
plt.show()


# # Z SCORES
#
plt.figure(figsize=(8, 4))
sns.histplot(z_scores, kde=True, color='salmon', bins=10)
plt.title("Z-Score Distribution")
plt.xlabel("Z-score")
plt.ylabel("Frequency")
plt.axvline(0, color='black', linestyle='--')
plt.axvline(2, color='red', linestyle='--', label="Z = 2")
plt.axvline(-2, color='red', linestyle='--', label="Z = -2")
plt.legend()
plt.show()


# TASK 3 CREATING GRAPH LINEARLY


x = [1,2, 3, 4,5]
y =  [2,4,6,8,10]

plt.plot(x,y)
plt.show()