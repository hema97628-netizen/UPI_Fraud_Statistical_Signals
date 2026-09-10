import pandas as pd
import os

# Dataset path
file_path = r"C:\Users\acer\Downloads\train_transaction.csv (1)\train_transaction.csv"

# Load required columns
df = pd.read_csv(
    file_path,
    usecols=[
        "TransactionAmt",
        "isFraud",
        "TransactionDT",
        "ProductCD",
        "card1"
    ]
)

print("\n========== PHASE 6: STATISTICAL FRAUD SIGNALS ==========")

# ------------------------------------------------------------
# 1. HIGH VALUE TRANSACTIONS
# ------------------------------------------------------------

amount_threshold = df["TransactionAmt"].quantile(0.99)

high_value = df[df["TransactionAmt"] >= amount_threshold]

high_value_fraud_rate = high_value["isFraud"].mean() * 100

print("\n--- Signal 1: High-Value Transactions ---")
print("99th percentile amount:", round(amount_threshold, 2))
print("High-value transactions:", len(high_value))
print("Fraud rate:", round(high_value_fraud_rate, 2), "%")

# ------------------------------------------------------------
# 2. EARLY MORNING TRANSACTIONS
# ------------------------------------------------------------

df["Hour"] = (df["TransactionDT"] % (24 * 60 * 60)) // 3600

early_morning = df[df["Hour"].between(4, 9)]

early_morning_fraud_rate = early_morning["isFraud"].mean() * 100

print("\n--- Signal 2: Early-Morning Activity ---")
print("Time window: 4 AM - 9 AM")
print("Transactions:", len(early_morning))
print("Fraud rate:", round(early_morning_fraud_rate, 2), "%")

# ------------------------------------------------------------
# 3. PRODUCT CATEGORY C
# ------------------------------------------------------------

product_c = df[df["ProductCD"] == "C"]

product_c_fraud_rate = product_c["isFraud"].mean() * 100

print("\n--- Signal 3: Product Category C ---")
print("Transactions:", len(product_c))
print("Fraud rate:", round(product_c_fraud_rate, 2), "%")

# ------------------------------------------------------------
# 4. TRANSACTION FREQUENCY
# ------------------------------------------------------------

card_frequency = df.groupby("card1").size()

df["TransactionFrequency"] = df["card1"].map(card_frequency)

frequency_threshold = df["TransactionFrequency"].quantile(0.99)

high_frequency = df[
    df["TransactionFrequency"] >= frequency_threshold
]

high_frequency_fraud_rate = high_frequency["isFraud"].mean() * 100

print("\n--- Signal 4: High Transaction Frequency ---")
print("99th percentile frequency:", round(frequency_threshold, 2))
print("High-frequency transactions:", len(high_frequency))
print("Fraud rate:", round(high_frequency_fraud_rate, 2), "%")

# ------------------------------------------------------------
# 5. CREATE FRAUD SIGNAL REPORT
# ------------------------------------------------------------

os.makedirs("reports", exist_ok=True)

report = f"""
STATISTICAL FRAUD SIGNAL REPORT
================================

Dataset Transactions: {len(df)}

SIGNAL 1 - HIGH-VALUE TRANSACTIONS
Threshold: 99th percentile = {amount_threshold:.2f}
Fraud Rate: {high_value_fraud_rate:.2f}%
Severity: MEDIUM
Finding: High-value transactions require monitoring, but high value alone
does not indicate fraud because many extreme transactions are legitimate.

SIGNAL 2 - EARLY-MORNING ACTIVITY
Time Window: 4 AM - 9 AM
Fraud Rate: {early_morning_fraud_rate:.2f}%
Severity: HIGH
Finding: Early-morning transactions show elevated fraud activity compared
with the overall dataset fraud rate.

SIGNAL 3 - PRODUCT CATEGORY C
Fraud Rate: {product_c_fraud_rate:.2f}%
Severity: HIGH
Finding: Product category C shows a substantially higher fraud rate and
should receive increased monitoring.

SIGNAL 4 - HIGH TRANSACTION FREQUENCY
99th Percentile Frequency: {frequency_threshold:.2f}
Fraud Rate: {high_frequency_fraud_rate:.2f}%
Severity: LOW
Finding: High transaction frequency alone does not appear to be a strong
fraud signal in this dataset.

IMPORTANT LIMITATION
--------------------
Direct device-change and account-age variables were not included in the
selected transaction dataset analysis. Therefore, these signals were not
fabricated or inferred.

OVERALL PRIORITY
----------------
1. Product Category C - HIGH
2. Early-Morning Activity - HIGH
3. High-Value Transactions - MEDIUM
4. High Transaction Frequency - LOW
"""

with open(
    "reports/fraud_signal_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\n========== FRAUD SIGNAL REPORT CREATED ==========")
print("Saved: reports/fraud_signal_report.txt")