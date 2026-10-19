
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

project_dir = Path(__file__).resolve().parent.parent

# Load the original dataset so we can split chronologically
file_path = project_dir / "data" / "supplier_orders.xlsx"
df = pd.read_excel(file_path)

# Parse dates and create the target
df["PO Date"] = pd.to_datetime(df["PO Date"], dayfirst=True)
df["Is_Late"] = (df["Days Late"] > 0).astype(int)

features = [
    "PO Type",
    "Supplier Country",
    "Supplier Region",
    "Supplier Tier",
    "Payment Terms",
    "Category",
    "Sub Category",
    "Unit Price",
    "Quantity",
    "Discount Pct",
    "Department",
    "Contract Type",
    "Single Source Flag",
    "Preferred Supplier",
    "Local International",
]

# Sort orders by date
df = df.sort_values("PO Date").dropna(
    subset=features + ["Is_Late", "PO Date"]
)

# Train on earlier orders; test on the latest 20%
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df["Is_Late"]
X_test = test_df[features]
y_test = test_df["Is_Late"]

categorical = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical = X_train.select_dtypes(
    exclude=["object", "str"]
).columns.tolist()

preprocessor = ColumnTransformer([
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical,
    ),
    (
        "numerical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]),
        numerical,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Time-based evaluation")
print("Training date range:", train_df["PO Date"].min(), "to", train_df["PO Date"].max())
print("Testing date range:", test_df["PO Date"].min(), "to", test_df["PO Date"].max())
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["On time or early", "Late"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
