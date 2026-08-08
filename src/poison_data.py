import os
import numpy as np
import pandas as pd

INPUT_FILE = "data/iris.csv"
OUTPUT_DIR = "data/mlsecops"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

LABEL_COLUMN = "species"

RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

classes = df[LABEL_COLUMN].unique()

rng = np.random.default_rng(RANDOM_SEED)

# Save clean baseline
clean_path = os.path.join(OUTPUT_DIR, "iris_clean.csv")
df.to_csv(clean_path, index=False)

print(f"Clean dataset: {len(df)} samples")
print(f"Classes: {list(classes)}")


def create_poisoned_dataset(df, percentage):
    poisoned = df.copy()

    n_samples = len(poisoned)
    n_poison = int(n_samples * percentage / 100)

    poison_indices = rng.choice(
        n_samples,
        size=n_poison,
        replace=False
    )

    # Replace all four features with random values
    poisoned.loc[poison_indices, "sepal_length"] = rng.uniform(
        0, 10, n_poison
    )

    poisoned.loc[poison_indices, "sepal_width"] = rng.uniform(
        0, 10, n_poison
    )

    poisoned.loc[poison_indices, "petal_length"] = rng.uniform(
        0, 10, n_poison
    )

    poisoned.loc[poison_indices, "petal_width"] = rng.uniform(
        0, 10, n_poison
    )

    # Assign random class labels
    poisoned.loc[poison_indices, LABEL_COLUMN] = rng.choice(
        classes,
        size=n_poison
    )

    return poisoned, poison_indices


for percentage in [5, 10, 50]:

    poisoned_df, indices = create_poisoned_dataset(
        df,
        percentage
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"iris_poisoned_{percentage}.csv"
    )

    poisoned_df.to_csv(output_file, index=False)

    print(
        f"{percentage}% poisoning: "
        f"{len(indices)} samples modified -> {output_file}"
    )
