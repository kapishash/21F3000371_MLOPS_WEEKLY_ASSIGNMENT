import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os

# Load original dataset
df = pd.read_csv("data/iris.csv")

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

# Create simulated production data
production = df.copy()

# Simulate distribution drift
production["sepal_length"] += 0.5
production["sepal_width"] += 0.3
production["petal_length"] += 1.0
production["petal_width"] += 0.2

os.makedirs("outputs/drift", exist_ok=True)

production.to_csv(
    "outputs/drift/iris_production.csv",
    index=False
)

print("=== Data Drift Analysis ===")

for feature in features:
    statistic, p_value = ks_2samp(
        df[feature],
        production[feature]
    )

    print(f"\nFeature: {feature}")
    print(f"KS Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: DRIFT DETECTED")
    else:
        print("Result: No significant drift")

    # Plot distributions
    plt.figure(figsize=(8, 5))

    plt.hist(
        df[feature],
        bins=20,
        alpha=0.6,
        label="Training Data"
    )

    plt.hist(
        production[feature],
        bins=20,
        alpha=0.6,
        label="Production Data"
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution Drift: {feature}")
    plt.legend()

    plt.savefig(
        f"outputs/drift/{feature}_drift.png",
        bbox_inches="tight"
    )

    plt.close()

print("\nDrift analysis completed.")
print("Plots saved in outputs/drift/")