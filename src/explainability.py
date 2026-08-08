import os
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "data/week9/iris_with_location.csv"
OUTPUT_DIR = "outputs/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Use only the original four features
features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

X = df[features]
y = df["species"]

# Train classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Model trained successfully")
print("Classes:", model.classes_)

# SHAP full-dataset explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("SHAP values generated")
print("SHAP shape:", shap_values.shape)

# Generate one summary plot per class
for class_index, class_name in enumerate(model.classes_):

    plt.figure()

    shap.summary_plot(
        shap_values[:, :, class_index],
        X,
        show=False
    )

    plt.title(f"SHAP Summary Plot - {class_name}")

    output_file = os.path.join(
        OUTPUT_DIR,
        f"shap_summary_{class_name}.png"
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")

print("\nAll SHAP plots generated successfully.")
