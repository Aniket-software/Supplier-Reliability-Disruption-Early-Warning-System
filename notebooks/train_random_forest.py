
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "ml_dataset.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=["Is_Late"])
y = df["Is_Late"]

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "str"]
).columns.tolist()

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_columns),
    ("numerical", numerical_pipeline, numerical_columns),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Random Forest training completed!")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["On time or early", "Late"],
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# Extract feature names after preprocessing
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# Extract importance scores from Random Forest
importances = model.named_steps["classifier"].feature_importances_

# Create a readable feature-importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances,
})

importance_df = importance_df.sort_values(
    "Importance", ascending=False
)

print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

# Save the results
output_path = project_dir / "data" / "feature_importance.csv"
importance_df.to_csv(output_path, index=False)

print("\nFeature importance saved to:", output_path)
