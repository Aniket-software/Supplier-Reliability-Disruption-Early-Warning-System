
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Find the project directory
project_dir = Path(__file__).resolve().parent.parent

# Load the prepared dataset
file_path = project_dir / "data" / "ml_dataset.csv"
df = pd.read_csv(file_path)

# Separate input features (X) and target (y)
X = df.drop(columns=["Is_Late"])
y = df["Is_Late"]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

# Prepare categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Prepare numerical features
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Apply the correct preprocessing to each column type
preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_columns),
    ("numerical", numerical_pipeline, numerical_columns)
])

# Create the complete machine-learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on unseen test data
predictions = model.predict(X_test)

# Evaluate performance
print("Training completed!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["On time or early", "Late"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
