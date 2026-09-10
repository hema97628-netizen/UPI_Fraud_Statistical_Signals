import pandas as pd

# Load dataset
file_path = r"C:\Users\acer\Downloads\train_transaction.csv (1)\train_transaction.csv"

df = pd.read_csv(file_path)

# 1. Basic dataset information
print("========== DATASET INFORMATION ==========")
print("Number of transactions:", len(df))
print("Number of features:", df.shape[1])

# 2. Fraud distribution
print("\n========== FRAUD DISTRIBUTION ==========")
print(df["isFraud"].value_counts())

fraud_percentage = df["isFraud"].mean() * 100

print("Percentage of suspicious/fraud transactions:",
      round(fraud_percentage, 2), "%")

# 3. Missing values
print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print("Total missing values:", missing_values.sum())

print("\nTop 20 columns with missing values:")
print(missing_values.sort_values(ascending=False).head(20))

# 4. Duplicate transactions
print("\n========== DUPLICATES ==========")

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)

# 5. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes.value_counts())

# 6. Transaction amount statistics
print("\n========== TRANSACTION AMOUNT ==========")
print(df["TransactionAmt"].describe())
# 7. Important feature information
print("\n========== IMPORTANT FEATURES ==========")

important_columns = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]

for column in important_columns:
    print(f"\n{column}")
    print("Data type:", df[column].dtype)
    print("Missing values:", df[column].isnull().sum())
    print("Unique values:", df[column].nunique())

# 8. Fraud vs legitimate transaction amount
print("\n========== TRANSACTION AMOUNT BY FRAUD STATUS ==========")

amount_by_fraud = df.groupby("isFraud")["TransactionAmt"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print(amount_by_fraud)
# ============================================================
# PHASE 2: DESCRIPTIVE STATISTICAL ANALYSIS
# ============================================================

print("\n========== PHASE 2: DESCRIPTIVE STATISTICS ==========")

# Transaction Amount statistics
amount = df["TransactionAmt"]

print("\n--- Transaction Amount ---")
print("Mean:", round(amount.mean(), 2))
print("Median:", round(amount.median(), 2))
print("Standard Deviation:", round(amount.std(), 2))

print("25th Percentile (Q1):", round(amount.quantile(0.25), 2))
print("50th Percentile (Median):", round(amount.quantile(0.50), 2))
print("75th Percentile (Q3):", round(amount.quantile(0.75), 2))

print("90th Percentile:", round(amount.quantile(0.90), 2))
print("95th Percentile:", round(amount.quantile(0.95), 2))
print("99th Percentile:", round(amount.quantile(0.99), 2))

# Skewness
print("\n--- Distribution Skewness ---")
print("Transaction Amount Skewness:", round(amount.skew(), 2))

# Fraud vs Legitimate descriptive statistics
print("\n--- Transaction Amount by Fraud Status ---")

fraud_stats = df.groupby("isFraud")["TransactionAmt"].agg(
    ["mean", "median", "std"]
)

print(fraud_stats)

# Transaction frequency using TransactionDT
print("\n--- Transaction Frequency ---")

df["Day"] = df["TransactionDT"] // (24 * 60 * 60)

daily_frequency = df.groupby("Day").size()

print("Number of days:", daily_frequency.nunique())
print("Average daily transaction frequency:",
      round(daily_frequency.mean(), 2))

print("Minimum daily transactions:", daily_frequency.min())
print("Maximum daily transactions:", daily_frequency.max())
# ============================================================
# PHASE 3: OUTLIER DETECTION
# ============================================================

print("\n========== PHASE 3: OUTLIER DETECTION ==========")

# IQR Method for Transaction Amount
Q1 = df["TransactionAmt"].quantile(0.25)
Q3 = df["TransactionAmt"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n--- IQR Method ---")
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower Bound:", round(lower_bound, 2))
print("Upper Bound:", round(upper_bound, 2))

outliers = df[
    (df["TransactionAmt"] < lower_bound) |
    (df["TransactionAmt"] > upper_bound)
]

print("Number of transaction amount outliers:", len(outliers))

outlier_percentage = len(outliers) / len(df) * 100

print("Percentage of outliers:",
      round(outlier_percentage, 2), "%")

print("\nHighest transaction amounts:")
print(
    df[["TransactionID", "TransactionAmt", "isFraud"]]
    .sort_values("TransactionAmt", ascending=False)
    .head(10)
)
# ============================================================
# Z-SCORE OUTLIER DETECTION
# ============================================================

print("\n--- Z-Score Method ---")

mean_amount = df["TransactionAmt"].mean()
std_amount = df["TransactionAmt"].std()

df["Amount_ZScore"] = (
    (df["TransactionAmt"] - mean_amount) / std_amount
)

zscore_outliers = df[
    df["Amount_ZScore"].abs() > 3
]

print("Z-score threshold: |Z| > 3")
print("Number of Z-score outliers:", len(zscore_outliers))

zscore_percentage = len(zscore_outliers) / len(df) * 100

print("Percentage of Z-score outliers:",
      round(zscore_percentage, 2), "%")

print("\nHighest Z-score transactions:")
print(
    df[["TransactionID", "TransactionAmt", "Amount_ZScore", "isFraud"]]
    .sort_values("Amount_ZScore", ascending=False)
    .head(10)
)
# ============================================================
# PHASE 3: VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder
os.makedirs("images", exist_ok=True)

# 1. Box Plot - Transaction Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["TransactionAmt"])
plt.title("Transaction Amount Box Plot")
plt.xlabel("Transaction Amount")
plt.savefig("images/transaction_amount_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

# 1. Box Plot - Transaction Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["TransactionAmt"])
plt.title("Transaction Amount Box Plot")
plt.xlabel("Transaction Amount")
plt.savefig("images/transaction_amount_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Distribution Plot - Transaction Amount
plt.figure(figsize=(10, 6))
sns.histplot(df["TransactionAmt"], bins=50, kde=False)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.savefig("images/transaction_amount_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n========== VISUALIZATIONS SAVED ==========")
print("Box plot saved: images/transaction_amount_boxplot.png")
print("Distribution plot saved: images/transaction_amount_distribution.png")