
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "supplier_orders.xlsx"

df = pd.read_excel(file_path)
df["PO Date"] = pd.to_datetime(df["PO Date"], dayfirst=True)
df["Is_Late"] = (df["Days Late"] > 0).astype(int)

# Calculate the overall late rate from training-period orders only
cutoff_date = pd.Timestamp("2024-05-23")

training_history = df[df["PO Date"] < cutoff_date]
fallback_rate = training_history["Is_Late"].mean()

# Calculate supplier history using only strictly earlier dates
df = df.sort_values(["PO Date", "Supplier ID"]).reset_index(drop=True)

daily_supplier = (
    df.groupby(["PO Date", "Supplier ID"], as_index=False)
    .agg(
        Daily_Orders=("Is_Late", "size"),
        Daily_Late_Orders=("Is_Late", "sum"),
    )
)

daily_supplier["Past_Orders"] = (
    daily_supplier.groupby("Supplier ID")["Daily_Orders"].cumsum()
    - daily_supplier["Daily_Orders"]
)

daily_supplier["Past_Late_Orders"] = (
    daily_supplier.groupby("Supplier ID")["Daily_Late_Orders"].cumsum()
    - daily_supplier["Daily_Late_Orders"]
)

daily_supplier["Historical_Late_Rate"] = (
    daily_supplier["Past_Late_Orders"]
    / daily_supplier["Past_Orders"].replace(0, pd.NA)
).fillna(fallback_rate)

# Merge the same-date historical rate back into each order
df = df.drop(columns=["Historical_Late_Rate"], errors="ignore")

df = df.merge(
    daily_supplier[
        ["PO Date", "Supplier ID", "Historical_Late_Rate",
         "Past_Orders", "Past_Late_Orders"]
    ],
    on=["PO Date", "Supplier ID"],
    how="left",
    validate="many_to_one",
)

print("Leakage-aware historical features created!")
print("Training-only fallback rate:", round(fallback_rate, 4))
print("Missing historical rates:", df["Historical_Late_Rate"].isna().sum())

print("\nSample:")
print(
    df[
        ["PO Date", "Supplier ID", "Is_Late",
         "Past_Orders", "Historical_Late_Rate"]
    ].head(15).to_string(index=False)
)
