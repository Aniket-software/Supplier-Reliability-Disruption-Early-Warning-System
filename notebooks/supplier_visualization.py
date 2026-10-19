import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find the project directory
project_dir = Path(__file__).resolve().parent.parent

# Load the supplier report we already created
report_path = project_dir / "data" / "supplier_reliability_report.csv"
df = pd.read_csv(report_path)

# Sort suppliers from highest to lowest late rate
df = df.sort_values("Late_Rate_Pct", ascending=True)

# Create the chart
plt.figure(figsize=(11, 7))

plt.barh(
    df["Supplier Name"],
    df["Late_Rate_Pct"]
)

plt.xlabel("Late Delivery Rate (%)")
plt.ylabel("Supplier")
plt.title("Supplier Reliability: Historical Late Delivery Rates")
plt.tight_layout()

# Save the chart
output_path = project_dir / "data" / "supplier_late_rates.png"
plt.savefig(output_path, dpi=150)

plt.show()

print(f"Chart saved to: {output_path}")