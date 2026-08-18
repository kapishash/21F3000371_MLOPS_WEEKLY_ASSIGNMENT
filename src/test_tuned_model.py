from google import genai
from google.genai import types

PROJECT_ID = "mlops-500009"
LOCATION = "us-central1"

# Your tuned model endpoint
# MODEL = "projects/558559129566/locations/us-central1/endpoints/8083860725816098816" # v1 model
MODEL = "projects/558559129566/locations/us-central1/endpoints/8831669370192134144" # v2 model

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

tests = [
    (
        "sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2",
        "setosa",
    ),
    (
        "sepal_length: 6.7, sepal_width: 3.1, petal_length: 4.7, petal_width: 1.5",
        "versicolor",
    ),
    (
        "sepal_length: 6.3, sepal_width: 3.3, petal_length: 6.0, petal_width: 2.5",
        "virginica",
    ),
]


def extract_prediction(text):
    text = text.lower()

    if "setosa" in text:
        return "setosa"
    elif "versicolor" in text:
        return "versicolor"
    elif "virginica" in text:
        return "virginica"
    else:
        return "unknown"


correct = 0

for i, (features, expected) in enumerate(tests, 1):

    print(f"\nTest {i}")
    print("Input:", features)
    print("Expected:", expected)

    prompt = f"""
Classify the Iris flower from these measurements.

{features}

Return only the correct species name:
setosa, versicolor, or virginica.
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=20,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                ),
            ),
        )

        raw_output = response.text.strip()

        prediction = extract_prediction(raw_output)

        print("Raw output:", raw_output)
        print("Prediction:", prediction)

        if prediction == expected:
            correct += 1
            print("Result: PASS")
        else:
            print("Result: FAIL")

    except Exception as e:
        print("Inference error:", e)
        print("Result: FAIL")


accuracy = correct / len(tests)

print("\n==============================")
print("Evaluation Results")
print("==============================")
print(f"Correct: {correct}/{len(tests)}")
print(f"Accuracy: {accuracy:.2%}")
print("==============================")