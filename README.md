# Supplier Reliability & Disruption Early-Warning System

An end-to-end data analytics and machine learning portfolio project for analyzing supplier delivery performance, identifying procurement risks, and supporting data-driven purchasing decisions.

## Project Overview

Supplier delays can disrupt operations, increase costs, and affect business continuity. This project analyzes historical purchase-order data to identify delivery patterns, compare supplier reliability, and explore whether machine learning can help identify potentially late orders.

## Key Features

- **Supplier Performance Analytics:** Analyze order volumes and historical late-delivery rates across suppliers.
- **Exploratory Data Analysis:** Investigate delivery patterns by supplier tier, country, category, and purchase-order type.
- **Machine Learning:** Compare Logistic Regression and Random Forest classification models.
- **Feature Importance:** Examine which input features influence Random Forest predictions.
- **Supplier Risk Classification:** Categorize suppliers using historical late-delivery-rate thresholds.
- **Procurement Recommendations:** Generate suggested actions based on historical supplier performance.
- **Interactive Dashboard:** Explore KPIs, supplier charts, risk categories, and downloadable reports using Streamlit.
- **Chronological Evaluation:** Evaluate model performance on a later time period to better reflect future prediction.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Excel and CSV data processing
- Git and GitHub

## Project Structure

```text
Supplier Reliability & Disruption Early-Warning System/
├── app.py
├── data/
│   ├── supplier_orders.xlsx
│   ├── supplier_reliability_report.csv
│   ├── supplier_recommendations.csv
│   ├── feature_importance.csv
│   ├── ml_dataset.csv
│   └── supplier_late_rates.png
├── notebooks/
│   ├── inspect_data.py
│   ├── profile_data.py
│   ├── validate_delivery.py
│   ├── supplier_analysis.py
│   ├── supplier_visualization.py
│   ├── prepare_ml_data.py
│   ├── train_model.py
│   ├── train_random_forest.py
│   ├── time_based_evaluation.py
│   ├── analyze_late_patterns.py
│   ├── check_baseline.py
│   ├── historical_supplier_features.py
│   └── final_model_evaluation.py
└── README.md
```

## Running the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Supplier Reliability & Disruption Early-Warning System"
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl streamlit
```

### 4. Generate the analysis reports

Run the analysis scripts from the project root, in the order required by their input files. For example:

```powershell
python .\notebooks\supplier_analysis.py
python .\notebooks\supplier_visualization.py
python .\notebooks\supplier_recommendations.py
```

### 5. Launch the dashboard

```powershell
python -m streamlit run app.py
```

## Current Dataset Findings

The dataset contains 5,200 purchase orders and 15 unique suppliers.

- Overall historical late-delivery rate: approximately 35.87%.
- Highest observed supplier late-delivery rate: approximately 41.14%.
- Lowest observed supplier late-delivery rate: approximately 32.16%.

These statistics describe this dataset and should not be assumed to represent real-world supplier populations.

## Model Evaluation

The project compares Logistic Regression and Random Forest. A chronological evaluation was also performed, using earlier orders for training and later orders for testing.

The current Random Forest baseline without the historical supplier feature achieved:

- Accuracy: 56.91%
- Late-order precision: 35.19%
- Late-order recall: 32.29%
- Late-order F1-score: 33.68%

These results indicate limited predictive performance. The model requires further feature engineering, threshold tuning, and validation before it should be used for operational decisions.

## Limitations

- The dataset contains only 15 suppliers, limiting generalization.
- Historical associations do not prove causal relationships.
- Rule-based supplier risk categories are not the same as validated future-risk predictions.
- Model performance is currently limited.
- Dataset provenance and real-world representativeness should be verified before making business claims.

## Future Improvements

- Tune prediction thresholds based on the cost of missed late orders.
- Add more historical supplier and delivery features.
- Improve time-based validation and model monitoring.
- Add explainable predictions and order-level risk alerts.
- Enhance dashboard design and deployment.

## Disclaimer

This is an educational portfolio project intended to demonstrate data analysis, machine learning, evaluation, and dashboard development. It is not a validated procurement risk management system.
