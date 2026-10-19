
import pandas as pd
import streamlit as st
from pathlib import Path

# Configure the dashboard
st.set_page_config(
    page_title="Supplier Intelligence Dashboard",
    page_icon="📦",
    layout="wide",
)

# Locate the project files
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# Load reports
supplier_file = DATA_DIR / "supplier_reliability_report.csv"
recommendation_file = DATA_DIR / "supplier_recommendations.csv"

if not supplier_file.exists() or not recommendation_file.exists():
    st.error("Required CSV files are missing. Run the analysis scripts first.")
    st.stop()

suppliers = pd.read_csv(supplier_file)
recommendations = pd.read_csv(recommendation_file)

# Dashboard heading
st.title("📦 Supplier Reliability & Disruption Early-Warning System")
st.caption("Historical delivery performance and procurement decision support")

# Sidebar filters
st.sidebar.header("Filters")

risk_options = ["All"] + sorted(
    recommendations["Risk_Level"].dropna().unique().tolist()
)

selected_risk = st.sidebar.selectbox(
    "Supplier risk level",
    risk_options,
)

filtered = recommendations.copy()

if selected_risk != "All":
    filtered = filtered[filtered["Risk_Level"] == selected_risk]

# Key performance indicators
total_suppliers = suppliers["Supplier ID"].nunique()
total_orders = int(suppliers["Total_Orders"].sum())
overall_late_rate = (
    suppliers["Late_Orders"].sum() / total_orders * 100
)
high_risk_count = int(
    (recommendations["Risk_Level"] == "High").sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Suppliers", total_suppliers)
col2.metric("Purchase Orders", f"{total_orders:,}")
col3.metric("Overall Late Rate", f"{overall_late_rate:.2f}%")
col4.metric("High-Risk Suppliers", high_risk_count)

st.divider()

# Historical performance chart
st.subheader("Historical Late-Delivery Rate by Supplier")

# Sort suppliers from highest to lowest late-delivery rate
chart_data = (
    filtered.sort_values("Late_Rate_Pct", ascending=False)
    .set_index("Supplier Name")
)

st.bar_chart(
    chart_data["Late_Rate_Pct"],
    horizontal=True,
    x_label="Late delivery rate (%)",
    y_label="Supplier",
)
# Recommendations table
st.subheader("Supplier Risk & Recommended Actions")

display_columns = [
    "Supplier Name",
    "Total_Orders",
    "Late_Orders",
    "Late_Rate_Pct",
    "Risk_Level",
    "Recommended_Action",
]

st.dataframe(
    filtered[display_columns].sort_values(
        "Late_Rate_Pct", ascending=False
    ),
    use_container_width=True,
    hide_index=True,
)

# Download filtered report
csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download supplier report",
    data=csv_data,
    file_name="supplier_recommendations.csv",
    mime="text/csv",
)

st.caption(
    "Risk categories are based on historical late-delivery rates. "
    "They are not validated predictions of future supplier disruptions."
)
