import pandas as pd
from pathlib import Path

# Locate and load the dataset
project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "supplier_orders.xlsx"

df = pd.read_excel(file_path)

# Convert delivery dates to datetime
for column in ["Requested Delivery", "Actual Delivery"]:
    df[column] = pd.to_datetime(
        df[column],
        dayfirst=True,
        errors="coerce"
    )

# Recalculate late status from the dates
df["Days Late Calculated"] = (
    df["Actual Delivery"] - df["Requested Delivery"]
).dt.days

df["Is Late"] = (
    df["Days Late Calculated"] > 0
)

# Group records by supplier
supplier_report = (
    df.groupby(["Supplier ID", "Supplier Name"])
    .agg(
        Total_Orders=("PO Number", "count"),
        Late_Orders=("Is Late", "sum"),
        Average_Days_Late=("Days Late Calculated", "mean"),
        Average_Lead_Time=("Lead Time Days", "mean"),
    )
    .reset_index()
)

# Calculate late-delivery percentage
supplier_report["Late_Rate_Pct"] = (
    supplier_report["Late_Orders"]
    / supplier_report["Total_Orders"]
    * 100
)

# Sort suppliers by late-delivery rate
supplier_report = supplier_report.sort_values(
    by="Late_Rate_Pct",
    ascending=False
)

# Display the results
print("\nSUPPLIER RELIABILITY REPORT")
print(supplier_report.round(2).to_string(index=False))

# Save the report
output_path = project_dir / "data" / "supplier_reliability_report.csv"
supplier_report.to_csv(output_path, index=False)

print(f"\nReport saved to: {output_path}")