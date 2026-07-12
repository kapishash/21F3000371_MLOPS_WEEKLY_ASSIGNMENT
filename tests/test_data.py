import pandas as pd


DATA_PATH = "data/iris.csv"


def test_dataset_exists():

    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0


def test_schema():

    df = pd.read_csv(DATA_PATH)

    expected_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    assert list(df.columns) == expected_columns


def test_missing_values():

    df = pd.read_csv(DATA_PATH)

    assert df.isnull().sum().sum() == 0


def test_feature_ranges():

    df = pd.read_csv(DATA_PATH)

    assert df["sepal_length"].between(0, 10).all()
    assert df["sepal_width"].between(0, 10).all()
    assert df["petal_length"].between(0, 10).all()
    assert df["petal_width"].between(0, 10).all()
