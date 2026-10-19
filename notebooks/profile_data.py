import pandas as pd
from pathlib import Path

# Locate and load the dataset
project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "supplier_orders.xlsx"

df = pd.read_excel(file_path)

# Convert date columns to datetime
date_columns = [
    "PO Date",
    "Requested Delivery",
    "Actual Delivery",
]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        dayfirst=True,
        errors="coerce"
    )

# Check duplicate purchase orders
print("Duplicate PO numbers:")
print(df["PO Number"].duplicated().sum())

# Check unique suppliers
print("\nUnique suppliers:")
print(df["Supplier ID"].nunique())

# Check delivery outcome distribution
print("\nOn-time delivery distribution:")
print(df["On Time Delivery"].value_counts(dropna=False))

# Check delay statistics
print("\nDays late statistics:")
print(df["Days Late"].describe())

# Check date ranges
print("\nDate ranges:")
for column in date_columns:
    print(column, ":", df[column].min(), "to", df[column].max())

# Check invalid delivery dates
invalid_dates = (
    df["Actual Delivery"] < df["Requested Delivery"]
).sum()

print("\nActual delivery before requested date:", invalid_dates)

# Check date parsing failures
print("\nInvalid or unparsed dates:")
print(df[date_columns].isna().sum())