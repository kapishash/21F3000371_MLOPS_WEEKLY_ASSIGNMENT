import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame

DATA_FILE = "data/week9/iris_with_location.csv"

# Load dataset
df = pd.read_csv(DATA_FILE)

# Original four features only
features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

X = df[features]
y = df["species"]

# Sensitive attribute
sensitive_feature = df["location"]

# Train/test split
X_train, X_test, y_train, y_test, location_train, location_test = train_test_split(
    X,
    y,
    sensitive_feature,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model WITHOUT location
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Overall metrics
print("\n=== Overall Metrics ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print(
    "Precision:",
    precision_score(y_test, y_pred, average="weighted", zero_division=0)
)
print(
    "Recall   :",
    recall_score(y_test, y_pred, average="weighted", zero_division=0)
)

# Fairlearn MetricFrame
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=location_test
)

print("\n=== Fairlearn MetricFrame ===")
print(metric_frame.by_group)

print("\n=== Overall Metrics from MetricFrame ===")
print(metric_frame.overall)

print("\n=== Metric Differences ===")
print(metric_frame.difference(method="between_groups"))
