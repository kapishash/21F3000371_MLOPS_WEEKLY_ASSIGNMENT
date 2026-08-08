import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/iris.csv"
OUTPUT_DIR = "data/week9"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Reproducible random location assignment
np.random.seed(42)
df["location"] = np.random.randint(0, 2, size=len(df))

output_file = os.path.join(OUTPUT_DIR, "iris_with_location.csv")
df.to_csv(output_file, index=False)

print("Dataset created successfully")
print("Total samples:", len(df))
print("\nLocation distribution:")
print(df["location"].value_counts().sort_index())
print("\nColumns:")
print(df.columns.tolist())
print("\nSaved to:", output_file)
