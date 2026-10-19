
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# 1. Load the original dataset
project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "supplier_orders.xlsx"

df = pd.read_excel(file_path)

df["PO Date"] = pd.to_datetime(df["PO Date"], dayfirst=True)
df["Is_Late"] = (df["Days Late"] > 0).astype(int)
df = df.sort_values("PO Date").reset_index(drop=True)

# 2. Define a strict chronological train/test split
cutoff_date = pd.Timestamp("2024-05-23")

# 3. Calculate daily supplier performance
daily = (
    df.groupby(["PO Date", "Supplier ID"])["Is_Late"]
    .agg(["sum", "count"])
    .reset_index()
    .rename(
        columns={
            "sum": "Daily_Late",
            "count": "Daily_Orders",
        }
    )
)

daily = daily.sort_values(["Supplier ID", "PO Date"])

# Count only orders from strictly earlier dates
daily["Past_Late"] = (
    daily.groupby("Supplier ID")["Daily_Late"].cumsum()
    - daily["Daily_Late"]
)

daily["Past_Orders"] = (
    daily.groupby("Supplier ID")["Daily_Orders"].cumsum()
    - daily["Daily_Orders"]
)

# 4. Calculate the overall late rate from strictly earlier dates
overall_daily = (
    df.groupby("PO Date")["Is_Late"]
    .agg(["sum", "count"])
    .sort_index()
)

overall_daily["Past_Late"] = (
    overall_daily["sum"].cumsum() - overall_daily["sum"]
)

overall_daily["Past_Orders"] = (
    overall_daily["count"].cumsum() - overall_daily["count"]
)

overall_daily["Fallback_Rate"] = (
    overall_daily["Past_Late"]
    / overall_daily["Past_Orders"].replace(0, pd.NA)
)

# 5. Add the historical overall rate to each supplier-date
daily = daily.merge(
    overall_daily[["Fallback_Rate"]],
    left_on="PO Date",
    right_index=True,
    how="left",
)

# 6. Calculate the historical supplier late rate
daily["Historical_Late_Rate"] = (
    daily["Past_Late"]
    / daily["Past_Orders"].replace(0, pd.NA)
).fillna(daily["Fallback_Rate"])

# Use 0 only when absolutely no earlier history exists
daily["Historical_Late_Rate"] = (
    daily["Historical_Late_Rate"].fillna(0.0)
)

# 7. Merge the feature into the original dataset
df = df.merge(
    daily[
        ["PO Date", "Supplier ID", "Historical_Late_Rate"]
    ],
    on=["PO Date", "Supplier ID"],
    how="left",
    validate="many_to_one",
)

# 8. Create train and test sets AFTER adding the feature
train_df = df[df["PO Date"] < cutoff_date].copy()
test_df = df[df["PO Date"] >= cutoff_date].copy()

print("Training dates:",
      train_df["PO Date"].min(), "to", train_df["PO Date"].max())
print("Testing dates:",
      test_df["PO Date"].min(), "to", test_df["PO Date"].max())
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print(
    "Missing historical rates:",
    df["Historical_Late_Rate"].isna().sum(),
)

# 9. Define features available at order creation
base_features = [
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

# 10. Train and evaluate a model
def evaluate(features, label):
    X_train = train_df[features]
    y_train = train_df["Is_Late"]

    X_test = test_df[features]
    y_test = test_df["Is_Late"]

    categorical_columns = X_train.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    numerical_columns = X_train.select_dtypes(
        exclude=["object", "str"]
    ).columns.tolist()

    preprocessing = ColumnTransformer([
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_columns,
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_columns,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessing),
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

    print(f"\n{label}")
    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4),
    )
    print(
        "Late precision:",
        round(
            precision_score(
                y_test, predictions, zero_division=0
            ),
            4,
        ),
    )
    print(
        "Late recall:",
        round(
            recall_score(
                y_test, predictions, zero_division=0
            ),
            4,
        ),
    )
    print(
        "Late F1:",
        round(
            f1_score(
                y_test, predictions, zero_division=0
            ),
            4,
        ),
    )


# 11. Compare both models on exactly the same test period
evaluate(
    base_features,
    "Random Forest without historical feature",
)

evaluate(
    base_features + ["Historical_Late_Rate"],
    "Random Forest with historical feature",
)
