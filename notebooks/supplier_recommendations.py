
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

# Load the supplier performance report
file_path = project_dir / "data" / "supplier_reliability_report.csv"
df = pd.read_csv(file_path)

# Categorize suppliers using historical late-delivery rates
def classify_risk(rate):
    if rate >= 40:
        return "High"
    elif rate >= 35:
        return "Medium"
    return "Low"

# Assign risk categories
df["Risk_Level"] = df["Late_Rate_Pct"].apply(classify_risk)

# Recommend procurement actions
def recommend_action(risk):
    if risk == "High":
        return "Review supplier performance and request a corrective action plan"
    elif risk == "Medium":
        return "Monitor upcoming orders and confirm delivery dates early"
    return "Maintain regular monitoring and review performance periodically"

df["Recommended_Action"] = df["Risk_Level"].apply(recommend_action)

# Sort highest-risk suppliers first
df = df.sort_values("Late_Rate_Pct", ascending=False)

# Save the report
output_path = project_dir / "data" / "supplier_recommendations.csv"
df.to_csv(output_path, index=False)

print("Supplier recommendations generated!")
print("\nRisk distribution:")
print(df["Risk_Level"].value_counts())

print("\nSupplier recommendations:")
print(
    df[
        [
            "Supplier Name",
            "Late_Rate_Pct",
            "Risk_Level",
            "Recommended_Action",
        ]
    ].to_string(index=False)
)

print("\nSaved to:", output_path)
