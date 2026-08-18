import pandas as pd
import json
import os

INPUT_FILE = "data/iris.csv"
OUTPUT_DIR = "data/week10"

V1_FILE = os.path.join(OUTPUT_DIR, "iris_v1_raw.jsonl")
V2_FILE = os.path.join(OUTPUT_DIR, "iris_v2_description.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

# -----------------------------
# V1: Raw Feature Representation
# -----------------------------

with open(V1_FILE, "w") as f:
    for _, row in df.iterrows():

        input_text = (
            f"sepal_length: {row['sepal_length']}, "
            f"sepal_width: {row['sepal_width']}, "
            f"petal_length: {row['petal_length']}, "
            f"petal_width: {row['petal_width']}"
        )

        record = {
            "input_text": input_text,
            "output_text": row["species"]
        }

        f.write(json.dumps(record) + "\n")


# -----------------------------
# V2: Natural Language Description
# -----------------------------

with open(V2_FILE, "w") as f:
    for _, row in df.iterrows():

        input_text = (
            f"A flower specimen has a sepal length of {row['sepal_length']} cm, "
            f"sepal width of {row['sepal_width']} cm, "
            f"petal length of {row['petal_length']} cm, "
            f"and petal width of {row['petal_width']} cm. "
            f"Identify the iris species."
        )

        output_text = f"This is Iris {row['species']}."

        record = {
            "input_text": input_text,
            "output_text": output_text
        }

        f.write(json.dumps(record) + "\n")


print("LLM training datasets created successfully.")
print(f"Original samples: {len(df)}")
print(f"V1 file: {V1_FILE}")
print(f"V2 file: {V2_FILE}")
