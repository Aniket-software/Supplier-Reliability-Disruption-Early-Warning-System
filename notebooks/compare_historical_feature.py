
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "supplier_orders.xlsx"

df = pd.read_excel(file_path)
df["PO Date"] = pd.to_datetime(df["PO Date"], dayfirst=True)
df["Is_Late"] = (df["Days Late"] > 0).astype(int)

cutoff_date = pd.Timestamp("2024-05-23")
train_df = df[df["PO Date"] < cutoff_date].copy()
test_df = df[df["PO Date"] >= cutoff_date].copy()

# Calculate historical supplier rates using training data only.
# For each test order, use outcomes from dates strictly before its date.
history = train_df.groupby("Supplier ID")["Is_Late"].agg(
    Past_Late_Orders="sum",
    Past_Orders="count",
)

fallback_rate = train_df["Is_Late"].mean()

def add_history(data):
    data = data.copy()
    data = data.merge(
        history,
        left_on="Supplier ID",
        right_index=True,
        how="left",
    )
    data["Historical_Late_Rate"] = (
        data["Past_Late_Orders"] / data["Past_Orders"]
    ).fillna(fallback_rate)
    return data

# Training feature values must use earlier dates, not each row's own outcome.
full_history = df.sort_values("PO Date").copy()
daily = (
    full_history.groupby(["PO Date", "Supplier ID"])["Is_Late"]
    .agg(["sum", "count"])
    .reset_index()
    .rename(columns={"sum": "Daily_Late", "count": "Daily_Orders"})
)
daily["Past_Late"] = (
    daily.groupby("Supplier ID")["Daily_Late"].cumsum() - daily["Daily_Late"]
)
daily["Past_Count"] = (
    daily.groupby("Supplier ID")["Daily_Orders"].cumsum() - daily["Daily_Orders"]
)
daily["Historical_Late_Rate"] = (
    daily["Past_Late"] / daily["Past_Count"].replace(0, pd.NA)
).fillna(fallback_rate)

df = df.merge(
    daily[["PO Date", "Supplier ID", "Historical_Late_Rate"]],
    on=["PO Date", "Supplier ID"],
    how="left",
    validate="many_to_one",
)

train_df = df[df["PO Date"] < cutoff_date].copy()
test_df = df[df["PO Date"] >= cutoff_date].copy()

base_features = [
    "PO Type", "Supplier Country", "Supplier Region", "Supplier Tier",
    "Payment Terms", "Category", "Sub Category", "Unit Price", "Quantity",
    "Discount Pct", "Department", "Contract Type", "Single Source Flag",
    "Preferred Supplier", "Local International",
]

def evaluate(features, label):
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
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numerical),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{label}")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Late precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Late recall:", round(recall_score(y_test, predictions, zero_division=0), 4))

evaluate(base_features, "Random Forest without historical feature")
evaluate(base_features + ["Historical_Late_Rate"],
         "Random Forest with historical feature")
