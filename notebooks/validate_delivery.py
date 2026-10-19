import pandas as pd
from pathlib import Path

# Locate and load the dataset
project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "supplier_orders.xlsx"

df = pd.read_excel(file_path)

# Parse delivery dates
date_columns = ["Requested Delivery", "Actual Delivery"]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        dayfirst=True,
        errors="coerce"
    )

# Calculate delay independently from the dates
df["Calculated Days Late"] = (
    df["Actual Delivery"] - df["Requested Delivery"]
).dt.days

# Check the supplied delay values
delay_mismatch = (
    df["Calculated Days Late"] != df["Days Late"]
)

print("Rows where calculated delay differs from Days Late:")
print(delay_mismatch.sum())

# Define late as delivery after the requested date
df["Calculated Late"] = (
    df["Calculated Days Late"] > 0
)

# Compare calculated late status with the supplied label
df["Reported Late"] = (
    df["On Time Delivery"].astype(str).str.strip().str.lower()
    == "no"
)

label_mismatch = (
    df["Calculated Late"] != df["Reported Late"]
)

print("\nRows where calculated late status differs from On Time Delivery:")
print(label_mismatch.sum())

# Show a few mismatches for inspection
print("\nExample mismatches:")
columns_to_show = [
    "PO Number",
    "Requested Delivery",
    "Actual Delivery",
    "Days Late",
    "Calculated Days Late",
    "On Time Delivery",
]

print(df.loc[label_mismatch, columns_to_show].head(10).to_string(index=False))