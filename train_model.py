import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("loan_approval.csv")


# Input features
X = df[
    [
        "income",
        "creditscore",
        "_employment_type",
        "region"
    ]
]


# Target
y = df["loan_approved"]


# Numerical columns
numeric_features = [
    "income",
    "creditscore"
]


# Categorical columns
categorical_features = [
    "_employment_type",
    "region"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
pipeline.fit(X_train, y_train)


# Test accuracy
accuracy = pipeline.score(X_test, y_test)

print("Model Accuracy:", accuracy)


# Save model
joblib.dump(pipeline, "model_joblib.pkl")

print("Model saved successfully!")