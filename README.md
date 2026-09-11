# Task 3: UPI Fraud Statistical Signals

## 1. Problem Statement

Digital payment platforms process a large number of transactions every day. Fraudulent transactions may show unusual patterns such as abnormal transaction amounts, unusual transaction times, and high-risk transaction categories.

The objective of this project is to perform exploratory and statistical analysis on transaction data to identify suspicious patterns and statistical fraud signals that can support fraud prevention teams.

## 2. Dataset Description

This project uses the IEEE-CIS Fraud Detection transaction dataset.

The selected transaction dataset contains:

- 590,540 transactions
- 394 features
- Fraud indicator: `isFraud`
- Transaction amount: `TransactionAmt`
- Transaction time information: `TransactionDT`
- Product category: `ProductCD`
- Card-related and email-domain features

The dataset is stored outside the project folder because of its large size.

## 3. Statistical Methods

The following methods were used:

- Exploratory Data Analysis (EDA)
- Missing value analysis
- Duplicate detection
- Mean, median and standard deviation
- Quartile and percentile analysis
- Skewness analysis
- IQR outlier detection
- Z-score outlier detection
- Transaction frequency analysis
- Fraud-rate analysis
- Welch's Independent T-Test
- Pattern analysis by time and product category

## 4. Fraud Signal Findings

### Overall Fraud Rate

The dataset contains 20,663 fraudulent transactions out of 590,540 transactions.

Overall fraud rate: **3.50%**

### Major Statistical Signals

| Signal | Fraud Rate | Severity |
|---|---:|---|
| Product Category C | 11.69% | High |
| Early-Morning Activity (4 AM–9 AM) | 7.06% | High |
| High-Value Transactions | 2.66% | Medium |
| High Transaction Frequency | 0.75% | Low |

Product Category C and early-morning activity show stronger fraud signals.

High transaction amount alone is not sufficient to identify fraud.

## 5. Visual Analysis

The project includes the following visualizations:

- Transaction Amount Box Plot
- Transaction Amount Distribution
- Transaction Amount vs Fraud
- Time of Day vs Fraud
- Product Category vs Fraud

All visualizations are stored in the `images/` directory.

## 6. Hypothesis Testing

### Null Hypothesis (H0)

Transaction amount has no significant difference between legitimate and fraudulent transactions.

### Alternative Hypothesis (H1)

Transaction amount has a significant difference between legitimate and fraudulent transactions.

Welch's Independent T-Test was performed.

- T-statistic: **-8.9494**
- P-value: **3.846 × 10⁻¹⁹**
- Significance level: **0.05**

Since the p-value is less than 0.05, the null hypothesis was rejected.

This indicates a statistically significant difference in transaction amounts between legitimate and fraudulent transactions.

However, statistical significance does not mean that every high-value transaction is fraudulent.

## 7. Business Recommendations

1. Apply enhanced monitoring to Product Category C transactions.
2. Increase risk checks for early-morning transactions.
3. Do not block transactions based only on transaction amount.
4. Combine multiple statistical signals before escalating transactions.
5. Use risk-based alerts to prioritize suspicious transactions.
6. Include device-change and account-age information in future analysis.

## 8. Business Conclusion

The analysis indicates that fraudulent activity is associated with specific transaction patterns rather than a single variable.

Product category and transaction time show stronger practical fraud signals, while high transaction amount alone is not sufficient to identify fraud.

A multi-signal risk monitoring approach is recommended for fraud prevention.

## 9. Future Scope

Future improvements can include:

- Device-change analysis
- Account-age analysis
- Advanced anomaly detection
- Machine learning-based fraud prediction
- Real-time fraud monitoring
- Customer risk scoring

## 10. Project Structure

```text
Fraud_Detection_Analysis/
│
├── data/
├── notebook/
├── images/
├── reports/
├── src/
│
│   ├── analysis.py
│   ├── pattern_analysis.py
│   ├── hypothesis_test.py
│   ├── fraud_signals.py
│   ├── business_insights.py
│   └── visualize.py
└── requirements.txt
```