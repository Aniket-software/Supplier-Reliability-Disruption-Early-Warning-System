
import pandas as pd
from pathlib import Path

# Find the project directory
project_dir = Path(__file__).resolve().parent.parent

# Load the dataset
file_path = project_dir / "data" / "supplier_orders.xlsx"
df = pd.read_excel(file_path)

# Define the target: 1 means late, 0 means on time or early
df["Is_Late"] = (df["Days Late"] > 0).astype(int)

# Select information that may be available when an order is placed
feature_columns = [
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

# Keep only the selected features and target
ml_data = df[feature_columns + ["Is_Late"]].copy()

# Remove rows with missing values in selected columns
ml_data = ml_data.dropna()

# Save the prepared dataset
output_path = project_dir / "data" / "ml_dataset.csv"
ml_data.to_csv(output_path, index=False)

print("Machine-learning dataset prepared!")
print("Dataset shape:", ml_data.shape)
print("\nTarget distribution:")
print(ml_data["Is_Late"].value_counts())
print("\nMissing values:", ml_data.isnull().sum().sum())
print("\nSaved to:", output_path)
