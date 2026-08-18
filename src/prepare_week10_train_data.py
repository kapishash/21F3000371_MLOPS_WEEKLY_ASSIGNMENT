import pandas as pd
import json

INPUT_FILE = "data/week10/train.csv"

V1_FILE = "data/week10/iris_v1_train_gemini.jsonl"
V2_FILE = "data/week10/iris_v2_train_gemini.jsonl"

df = pd.read_csv(INPUT_FILE)

# -----------------------------
# V1 — Raw Feature Representation
# -----------------------------

with open(V1_FILE, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():

        input_text = (
            f"sepal_length: {row['sepal_length']}, "
            f"sepal_width: {row['sepal_width']}, "
            f"petal_length: {row['petal_length']}, "
            f"petal_width: {row['petal_width']}"
        )

        record = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": input_text
                        }
                    ]
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": row["species"]
                        }
                    ]
                }
            ]
        }

        f.write(json.dumps(record) + "\n")


# -----------------------------
# V2 — Natural Language Description
# -----------------------------

with open(V2_FILE, "w", encoding="utf-8") as f:
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
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": input_text
                        }
                    ]
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": output_text
                        }
                    ]
                }
            ]
        }

        f.write(json.dumps(record) + "\n")


print("Training datasets created successfully.")
print(f"Training samples: {len(df)}")
print(f"V1: {V1_FILE}")
print(f"V2: {V2_FILE}")
