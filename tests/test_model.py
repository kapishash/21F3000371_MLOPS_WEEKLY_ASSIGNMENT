import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


MODEL_PATH = "models/model.pkl"
EVAL_PATH = "artifacts/eval.csv"


def test_model_accuracy():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(EVAL_PATH)

    X = df.drop("species", axis=1)
    y = df["species"]

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    print("Accuracy:", accuracy)

    assert accuracy >= 0.8