
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

project_dir = Path(__file__).resolve().parent.parent
df = pd.read_excel(project_dir / "data" / "supplier_orders.xlsx")

df["PO Date"] = pd.to_datetime(df["PO Date"], dayfirst=True)
df["Is_Late"] = (df["Days Late"] > 0).astype(int)

# Use a strict date boundary: training dates must be before testing dates
cutoff_date = pd.Timestamp("2024-05-23")

train_df = df[df["PO Date"] < cutoff_date].copy()
test_df = df[df["PO Date"] >= cutoff_date].copy()

print("Training date range:", train_df["PO Date"].min(), "to", train_df["PO Date"].max())
print("Testing date range:", test_df["PO Date"].min(), "to", test_df["PO Date"].max())
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

# Baseline: always predict the most common training class
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(train_df[["PO Date"]], train_df["Is_Late"])

predictions = baseline.predict(test_df[["PO Date"]])

print("\nBaseline accuracy:", round(accuracy_score(test_df["Is_Late"], predictions), 4))
print("\nBaseline classification report:")
print(classification_report(
    test_df["Is_Late"],
    predictions,
    labels=[0, 1],
    target_names=["On time or early", "Late"],
    zero_division=0,
))
