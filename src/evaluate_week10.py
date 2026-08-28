import pandas as pd
from google import genai
from google.genai import types
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "mlops-500009"
LOCATION = "us-central1"

V1_MODEL = (
    "projects/558559129566/locations/us-central1/"
    "endpoints/3300967552804454400"
)

V2_MODEL = (
    "projects/558559129566/locations/us-central1/"
    "endpoints/432456065146159104"
)

TEST_FILE = "data/week10/test.csv"
OUTPUT_FILE = "data/week10/week10_evaluation_results.csv"

VALID_CLASSES = {"setosa", "versicolor", "virginica"}

# ============================================================
# CLIENT
# ============================================================

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_FILE)

print("=" * 60)
print("WEEK 10 — LLMOPS MODEL EVALUATION")
print("=" * 60)

print(f"Test samples: {len(df)}")
print("\nTest distribution:")
print(df["species"].value_counts())

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_v1_prompt(row):
    return (
        "Classify the Iris flower from these measurements.\n\n"
        f"sepal_length: {row['sepal_length']}, "
        f"sepal_width: {row['sepal_width']}, "
        f"petal_length: {row['petal_length']}, "
        f"petal_width: {row['petal_width']}\n\n"
        "Return only the correct species name:\n"
        "setosa, versicolor, or virginica."
    )


def create_v2_prompt(row):
    return (
        "Classify the Iris flower from this description.\n\n"
        f"A flower specimen has a sepal length of "
        f"{row['sepal_length']} cm, "
        f"sepal width of {row['sepal_width']} cm, "
        f"petal length of {row['petal_length']} cm, "
        f"and petal width of {row['petal_width']} cm. "
        "Identify the iris species.\n\n"
        "Return only the correct species name:\n"
        "setosa, versicolor, or virginica."
    )


def get_response(model, prompt):
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=20,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                ),
            ),
        )

        if response.text:
            return response.text.strip()

        return ""

    except Exception as e:
        print(f"Inference error: {e}")
        return ""


def extract_prediction(raw_output):
    """
    Extract the species for accuracy/precision/recall.

    This is intentionally separate from format compliance.
    """

    text = raw_output.strip().lower()

    for species in VALID_CLASSES:
        if species in text:
            return species

    return "unknown"


def check_format_compliance(raw_output):
    """
    Assignment definition:

    Valid:
        setosa
        versicolor
        virginica

    Invalid:
        SETOSA
        This is Iris setosa.
        {"species": "setosa"}
        setosa.
        any extra text
    """

    return raw_output.strip() in VALID_CLASSES


# ============================================================
# EVALUATION
# ============================================================

results = []

for index, row in df.iterrows():

    expected = row["species"]

    print(f"\nTest {index + 1}/{len(df)}")
    print(
        f"Features: "
        f"{row['sepal_length']}, "
        f"{row['sepal_width']}, "
        f"{row['petal_length']}, "
        f"{row['petal_width']}"
    )
    print(f"Expected: {expected}")

    # -------------------------
    # V1
    # -------------------------

    v1_prompt = create_v1_prompt(row)
    v1_raw = get_response(V1_MODEL, v1_prompt)
    v1_prediction = extract_prediction(v1_raw)
    v1_compliant = check_format_compliance(v1_raw)

    print(f"V1 Raw: {v1_raw}")
    print(f"V1 Prediction: {v1_prediction}")
    print(f"V1 Format: {v1_compliant}")

    # -------------------------
    # V2
    # -------------------------

    v2_prompt = create_v2_prompt(row)
    v2_raw = get_response(V2_MODEL, v2_prompt)
    v2_prediction = extract_prediction(v2_raw)
    v2_compliant = check_format_compliance(v2_raw)

    print(f"V2 Raw: {v2_raw}")
    print(f"V2 Prediction: {v2_prediction}")
    print(f"V2 Format: {v2_compliant}")

    results.append(
        {
            "index": index,
            "sepal_length": row["sepal_length"],
            "sepal_width": row["sepal_width"],
            "petal_length": row["petal_length"],
            "petal_width": row["petal_width"],
            "actual": expected,

            "v1_raw": v1_raw,
            "v1_prediction": v1_prediction,
            "v1_format_compliant": v1_compliant,

            "v2_raw": v2_raw,
            "v2_prediction": v2_prediction,
            "v2_format_compliant": v2_compliant,
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

# ============================================================
# METRICS
# ============================================================

y_true = results_df["actual"]

y_v1 = results_df["v1_prediction"]
y_v2 = results_df["v2_prediction"]

# Accuracy
v1_accuracy = accuracy_score(y_true, y_v1)
v2_accuracy = accuracy_score(y_true, y_v2)

# Precision
v1_precision = precision_score(
    y_true,
    y_v1,
    labels=["setosa", "versicolor", "virginica"],
    average=None,
    zero_division=0,
)

v2_precision = precision_score(
    y_true,
    y_v2,
    labels=["setosa", "versicolor", "virginica"],
    average=None,
    zero_division=0,
)

# Recall
v1_recall = recall_score(
    y_true,
    y_v1,
    labels=["setosa", "versicolor", "virginica"],
    average=None,
    zero_division=0,
)

v2_recall = recall_score(
    y_true,
    y_v2,
    labels=["setosa", "versicolor", "virginica"],
    average=None,
    zero_division=0,
)

# Format compliance
v1_format = results_df["v1_format_compliant"].mean()
v2_format = results_df["v2_format_compliant"].mean()

classes = ["setosa", "versicolor", "virginica"]

# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TASK 4 RESULTS")
print("=" * 60)

print("\nOverall Metrics")
print("-" * 60)

print(
    f"{'Metric':<25}"
    f"{'V1 Raw':>15}"
    f"{'V2 Description':>20}"
)

print("-" * 60)

print(
    f"{'Accuracy':<25}"
    f"{v1_accuracy:>14.2%}"
    f"{v2_accuracy:>19.2%}"
)

print(
    f"{'Format Compliance':<25}"
    f"{v1_format:>14.2%}"
    f"{v2_format:>19.2%}"
)

print("\nPer-Class Precision")
print("-" * 60)

for i, cls in enumerate(classes):
    print(
        f"{cls:<25}"
        f"{v1_precision[i]:>14.2%}"
        f"{v2_precision[i]:>19.2%}"
    )

print("\nPer-Class Recall")
print("-" * 60)

for i, cls in enumerate(classes):
    print(
        f"{cls:<25}"
        f"{v1_recall[i]:>14.2%}"
        f"{v2_recall[i]:>19.2%}"
    )

# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

print("\n")
print("=" * 60)
print("V1 CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_v1,
        labels=classes,
        zero_division=0,
    )
)

print("=" * 60)
print("V2 CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_v2,
        labels=classes,
        zero_division=0,
    )
)

# ============================================================
# WINNER
# ============================================================

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

if v1_accuracy > v2_accuracy:
    print("Winner by accuracy: V1 — Raw Feature Representation")
elif v2_accuracy > v1_accuracy:
    print("Winner by accuracy: V2 — Natural Language Description")
else:
    print("Winner by accuracy: Tie")

print(f"\nV1 Accuracy: {v1_accuracy:.2%}")
print(f"V2 Accuracy: {v2_accuracy:.2%}")

print(f"\nV1 Format Compliance: {v1_format:.2%}")
print(f"V2 Format Compliance: {v2_format:.2%}")

print("\nDetailed predictions saved to:")
print(OUTPUT_FILE)

print("=" * 60)