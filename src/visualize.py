import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = r"C:\Users\acer\Downloads\train_transaction.csv (1)\train_transaction.csv"

df = pd.read_csv(file_path, usecols=["TransactionAmt"])

os.makedirs("images", exist_ok=True)

plot_data = df["TransactionAmt"].sample(
    n=min(50000, len(df)),
    random_state=42
)

plt.figure(figsize=(10, 6))
plt.boxplot(plot_data)
plt.title("Transaction Amount Box Plot")
plt.ylabel("Transaction Amount")
plt.savefig(
    "images/transaction_amount_boxplot.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(plot_data, bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.savefig(
    "images/transaction_amount_distribution.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("VISUALIZATIONS SAVED")