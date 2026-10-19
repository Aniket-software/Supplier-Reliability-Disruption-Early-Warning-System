import pandas as pd
from pathlib import Path

# Find the project directory
project_dir = Path(__file__).resolve().parent.parent

# Load the dataset
file_path = project_dir / "data" / "supplier_orders.xlsx"
df = pd.read_excel(file_path)

# Display columns relevant to our prediction task
columns_to_inspect = [
    "PO Date",
    "Supplier Tier",
    "Supplier Status",
    "Supplier Risk",
    "Supplier ESG Score",
    "Quantity",
    "Unit Price",
    "Category",
    "Payment Terms",
    "Contract Type",
    "Single Source Flag",
    "Preferred Supplier",
    "Local International",
    "Requested Delivery",
    "Actual Delivery",
    "Days Late",
    "On Time Delivery",
    "Lead Time Days",
]

print("Dataset shape:", df.shape)
print("\nSelected columns:")
print(df[columns_to_inspect].head())

print("\nData types:")
print(df[columns_to_inspect].dtypes)

print("\nTarget distribution:")
print(df["On Time Delivery"].value_counts())