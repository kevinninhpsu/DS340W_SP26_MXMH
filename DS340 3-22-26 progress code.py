"""
Program: Train and evaluate models using MXMH + Spotify engineered features
Purpose: Recreate parent paper models and compare them with an added improvement model
Author: Sanjana Vuppunahalli
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report
)


# --------------------------------------------------
# 1. Load already split datasets
# --------------------------------------------------

train_df = pd.read_csv("datasets/mxmh_spotify_train.csv")
val_df = pd.read_csv("datasets/mxmh_spotify_validation.csv")
test_df = pd.read_csv("datasets/mxmh_spotify_test.csv")

TARGET = "Depression"   # Change if needed

if TARGET not in train_df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in training data.")

# Convert Depression into binary classes for classification
train_df["Depression"] = (train_df["Depression"] >= 5).astype(int)
val_df["Depression"] = (val_df["Depression"] >= 5).astype(int)
test_df["Depression"] = (test_df["Depression"] >= 5).astype(int)


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# --------------------------------------------------
# 3. Encode target if needed
# --------------------------------------------------

label_encoder = None

if y_train.dtype == "object" or str(y_train.dtype) == "category":
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)


# --------------------------------------------------
# 4. Identify numeric and categorical columns
# --------------------------------------------------

numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# --------------------------------------------------
# 5. Build preprocessing pipeline
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# --------------------------------------------------
# 6. Define models and hyperparameter grids
# --------------------------------------------------

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        {
            "model__C": [0.1, 1, 10]
        }
    ),

    "SVM": (
        SVC(class_weight="balanced", random_state=42),
        {
            "model__C": [0.5, 1, 5],
            "model__kernel": ["rbf"]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 25]
        }
    ),

    "MLP": (
        MLPClassifier(max_iter=500, random_state=42),
        {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "model__alpha": [0.0001, 0.001],
            "model__learning_rate_init": [0.001, 0.01]
        }
    ),

    "HistGradientBoosting": (
        HistGradientBoostingClassifier(random_state=42),
        {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [200, 400]
        }
    )
}


# --------------------------------------------------
# 7. Cross-validation setup
# --------------------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# --------------------------------------------------
# 8. Train, tune, and evaluate all models
# --------------------------------------------------

results = []
best_model_name = None
best_pipeline = None
best_val_f1 = -1

for name, (model, param_grid) in models.items():
    print("\n" + "=" * 60)
    print(f"Training: {name}")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_

    val_pred = best_estimator.predict(X_val)
    test_pred = best_estimator.predict(X_test)

    val_accuracy = accuracy_score(y_val, val_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    val_f1 = f1_score(y_val, val_pred, average="macro")
    test_f1 = f1_score(y_test, test_pred, average="macro")

    val_bal_acc = balanced_accuracy_score(y_val, val_pred)
    test_bal_acc = balanced_accuracy_score(y_test, test_pred)

    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nValidation Metrics:")
    print(f"Accuracy:           {val_accuracy:.4f}")
    print(f"F1 Macro:           {val_f1:.4f}")
    print(f"Balanced Accuracy:  {val_bal_acc:.4f}")

    print("\nTest Metrics:")
    print(f"Accuracy:           {test_accuracy:.4f}")
    print(f"F1 Macro:           {test_f1:.4f}")
    print(f"Balanced Accuracy:  {test_bal_acc:.4f}")

    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred))

    results.append({
        "Model": name,
        "Best Parameters": str(grid.best_params_),
        "Validation Accuracy": val_accuracy,
        "Validation F1 Macro": val_f1,
        "Validation Balanced Accuracy": val_bal_acc,
        "Test Accuracy": test_accuracy,
        "Test F1 Macro": test_f1,
        "Test Balanced Accuracy": test_bal_acc
    })

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_name = name
        best_pipeline = best_estimator


# --------------------------------------------------
# 9. Save results table
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Validation F1 Macro", ascending=False)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print(results_df)

results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved results to: model_comparison_results.csv")


# --------------------------------------------------
# 10. Save best model
# --------------------------------------------------

if best_pipeline is not None:
    joblib.dump(best_pipeline, "best_model_pipeline.pkl")
    print(f"\nBest model based on Validation F1 Macro: {best_model_name}")
    print("Saved best model to: best_model_pipeline.pkl")


# --------------------------------------------------
# 11. Optional: Show top feature importances for tree-based best models
# --------------------------------------------------

if best_pipeline is not None and best_model_name in ["Random Forest", "Decision Tree"]:
    print("\n" + "=" * 60)
    print(f"Feature Importance for Best Model: {best_model_name}")

    fitted_preprocessor = best_pipeline.named_steps["preprocess"]
    fitted_model = best_pipeline.named_steps["model"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 20 Features:")
    print(importance_df.head(20))

    importance_df.to_csv("best_model_feature_importance.csv", index=False)
    print("\nSaved feature importances to: best_model_feature_importance.csv")