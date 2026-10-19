
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "supplier_orders.xlsx"

df = pd.read_excel(file_path)
df["Is_Late"] = (df["Days Late"] > 0).astype(int)

print("Overall late rate:", round(df["Is_Late"].mean() * 100, 2), "%")

# Compare late rates across selected categories
categorical_columns = [
    "Supplier Tier",
    "Supplier Country",
    "PO Type",
    "Category",
    "Contract Type",
    "Preferred Supplier",
    "Single Source Flag",
]

for column in categorical_columns:
    print(f"\n--- Late rate by {column} ---")

    summary = (
        df.groupby(column, dropna=False)
        .agg(
            Orders=("Is_Late", "size"),
            Late_Rate=("Is_Late", "mean"),
        )
        .reset_index()
    )

    summary["Late_Rate"] = (summary["Late_Rate"] * 100).round(2)

    print(
        summary.sort_values("Late_Rate", ascending=False)
        .to_string(index=False)
    )

print("\n--- Numerical feature comparison ---")

numerical_columns = ["Quantity", "Unit Price", "Discount Pct"]

print(
    df.groupby("Is_Late")[numerical_columns]
    .mean()
    .round(2)
    .rename(index={0: "On time or early", 1: "Late"})
)
