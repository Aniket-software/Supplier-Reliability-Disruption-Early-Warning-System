import pandas as pd
from pathlib import Path

# Find the main project folder
project_dir = Path(__file__).resolve().parent.parent

# Build the dataset path
file_path = project_dir / "data" / "supplier_orders.xlsx"

# Load the Excel file
df = pd.read_excel(file_path)

# Inspect the dataset
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
for column in df.columns:
    print(column)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())