import pandas as pd
from scipy.stats import ttest_ind

# Dataset path
file_path = r"C:\Users\acer\Downloads\train_transaction.csv (1)\train_transaction.csv"

# Load required columns only
df = pd.read_csv(
    file_path,
    usecols=["TransactionAmt", "isFraud"]
)

print("\n========== PHASE 5: HYPOTHESIS TESTING ==========")

# Separate transaction amounts
legitimate = df[df["isFraud"] == 0]["TransactionAmt"]
fraud = df[df["isFraud"] == 1]["TransactionAmt"]

# Hypotheses
print("\nH0: Transaction amount has no significant difference between legitimate and fraud transactions.")
print("H1: Transaction amount has a significant difference between legitimate and fraud transactions.")

# Welch's t-test
t_stat, p_value = ttest_ind(
    legitimate,
    fraud,
    equal_var=False
)

print("\n--- Welch's Independent T-Test ---")
print("T-statistic:", round(t_stat, 4))
print("P-value:", p_value)

# Significance level
alpha = 0.05

print("\nSignificance level:", alpha)

if p_value < alpha:
    print("Result: Reject H0")
    print("Business Interpretation: Transaction amount shows a statistically significant difference between legitimate and fraud transactions.")
else:
    print("Result: Fail to reject H0")
    print("Business Interpretation: No statistically significant difference was detected.")

print("\n========== HYPOTHESIS TEST COMPLETED ==========")