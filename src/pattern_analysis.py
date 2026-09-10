import pandas as pd
import matplotlib.pyplot as plt
import os

# Dataset path
file_path = r"C:\Users\acer\Downloads\train_transaction.csv (1)\train_transaction.csv"

# Load only required columns
df = pd.read_csv(
    file_path,
    usecols=["TransactionDT", "TransactionAmt", "isFraud", "ProductCD", "card1"]
)

os.makedirs("images", exist_ok=True)

# ============================================================
# PHASE 4: PATTERN ANALYSIS
# ============================================================

print("\n========== PHASE 4: PATTERN ANALYSIS ==========")

# ------------------------------------------------------------
# 1. TRANSACTION AMOUNT VS FRAUD
# ------------------------------------------------------------

amount_fraud = df.groupby("isFraud")["TransactionAmt"].mean()

print("\n--- Average Transaction Amount by Fraud Status ---")
print(amount_fraud)

plt.figure(figsize=(8, 5))
plt.bar(
    ["Legitimate", "Fraud"],
    [amount_fraud[0], amount_fraud[1]]
)
plt.title("Average Transaction Amount: Fraud vs Legitimate")
plt.ylabel("Average Transaction Amount")
plt.savefig(
    "images/amount_vs_fraud.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 2. TIME OF DAY VS FRAUD
# ------------------------------------------------------------

df["Hour"] = (df["TransactionDT"] % (24 * 60 * 60)) // 3600

hour_fraud_rate = df.groupby("Hour")["isFraud"].mean() * 100

print("\n--- Fraud Rate by Hour ---")
print(hour_fraud_rate.round(2))

plt.figure(figsize=(10, 5))
plt.plot(hour_fraud_rate.index, hour_fraud_rate.values, marker="o")
plt.title("Fraud Rate by Time of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Fraud Rate (%)")
plt.xticks(range(0, 24))
plt.savefig(
    "images/time_of_day_vs_fraud.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 3. PRODUCT CATEGORY VS FRAUD
# ------------------------------------------------------------

product_fraud_rate = df.groupby("ProductCD")["isFraud"].mean() * 100

print("\n--- Fraud Rate by Product Category ---")
print(product_fraud_rate.round(2))

plt.figure(figsize=(8, 5))
plt.bar(
    product_fraud_rate.index,
    product_fraud_rate.values
)
plt.title("Fraud Rate by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Fraud Rate (%)")
plt.savefig(
    "images/product_category_vs_fraud.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 4. TRANSACTION FREQUENCY PROXY
# ------------------------------------------------------------

card_frequency = df.groupby("card1").size()

df["TransactionFrequency"] = df["card1"].map(card_frequency)

frequency_fraud = df.groupby("isFraud")["TransactionFrequency"].mean()

print("\n--- Average Transaction Frequency by Fraud Status ---")
print(frequency_fraud.round(2))

# ------------------------------------------------------------
# 5. HIGH-FREQUENCY TRANSACTIONS
# ------------------------------------------------------------

frequency_threshold = df["TransactionFrequency"].quantile(0.99)

high_frequency = df[
    df["TransactionFrequency"] >= frequency_threshold
]

print("\n--- High Frequency Transactions ---")
print("99th percentile frequency threshold:",
      round(frequency_threshold, 2))

print("High-frequency transactions:",
      len(high_frequency))

print("Fraud rate among high-frequency transactions:",
      round(high_frequency["isFraud"].mean() * 100, 2), "%")

print("\n========== PHASE 4 VISUALIZATIONS SAVED ==========")
print("1. images/amount_vs_fraud.png")
print("2. images/time_of_day_vs_fraud.png")
print("3. images/product_category_vs_fraud.png")