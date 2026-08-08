import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


DATA_DIR = "data/mlsecops"

DATASETS = {
    "0%": "iris_clean.csv",
    "5%": "iris_poisoned_5.csv",
    "10%": "iris_poisoned_10.csv",
    "50%": "iris_poisoned_50.csv",
}

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET = "species"



mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_mlsecops_poisoning")


for poisoning_level, filename in DATASETS.items():

    print("\n" + "=" * 60)
    print(f"Training with poisoning level: {poisoning_level}")
    print("=" * 60)

    file_path = os.path.join(DATA_DIR, filename)

    df = pd.read_csv(file_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    with mlflow.start_run(run_name=f"iris_{poisoning_level}"):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        mlflow.log_param(
            "poisoning_level",
            poisoning_level
        )

        mlflow.log_param(
            "dataset",
            filename
        )

        mlflow.log_param(
            "n_estimators",
            100
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")


print("\nAll MLSecOps experiments completed.")
