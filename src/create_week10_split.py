import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/iris.csv"

TRAIN_FILE = "data/week10/train.csv"
TEST_FILE = "data/week10/test.csv"

df = pd.read_csv(INPUT_FILE)

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["species"],
)

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))

print("\nTrain distribution:")
print(train_df["species"].value_counts())

print("\nTest distribution:")
print(test_df["species"].value_counts())

print("\nSaved:")
print(TRAIN_FILE)
print(TEST_FILE)
