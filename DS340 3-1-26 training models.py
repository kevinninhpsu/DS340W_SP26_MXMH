"""
Program: Train models using MXMH + Spotify engineered features
Purpose: Continue development using pre-split train/validation/test datasets
"""

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report


# -----------------------------
# Load already split datasets
# -----------------------------

train_df = pd.read_csv("mxmh_spotify_train.csv")
val_df = pd.read_csv("mxmh_spotify_validation.csv")
test_df = pd.read_csv("mxmh_spotify_test.csv")


TARGET = "Depression"   # adjust if using Anxiety or other label


X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# -----------------------------
# Preprocessing pipeline
# -----------------------------

numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# -----------------------------
# Model definitions
# -----------------------------

models = {

    "Logistic Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"model__C": [0.1, 1, 10]}
    ),

    "Random Forest": (
        RandomForestClassifier(class_weight="balanced"),
        {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 25]
        }
    ),

    "SVM": (
        SVC(class_weight="balanced"),
        {
            "model__C": [0.5, 1, 5],
            "model__kernel": ["rbf"]
        }
    ),

    "HistGradientBoosting": (
        HistGradientBoostingClassifier(),
        {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [200, 400]
        }
    )
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -----------------------------
# Train and evaluate
# -----------------------------

for name, (model, param_grid) in models.items():

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\n====================================")
    print(name)
    print("Best parameters:", grid.best_params_)

    val_pred = grid.predict(X_val)
    test_pred = grid.predict(X_test)

    print("\nValidation F1:", f1_score(y_val, val_pred, average="macro"))
    print("Test F1:", f1_score(y_test, test_pred, average="macro"))

    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred))
