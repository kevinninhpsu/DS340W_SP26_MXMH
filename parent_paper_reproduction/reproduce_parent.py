"""
Parent paper reproduction:
Impact of Music on Brain and Mental Health: Classification using Machine Learning Algorithms

This script reproduces the parent paper as closely as possible using:
- MXMH survey dataset
- basic cleaning / outlier removal
- Decision Tree, Random Forest, SVM, and MLP
- 80/20 train-test split
- accuracy, precision, recall, F1, confusion matrices

Folder structure expected:
parent_paper_reproduction/
    reproduce_parent.py
    mxmh_survey_results.csv
    parent_paper.pdf   # optional, not used by code
    outputs/
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mxmh_survey_results.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        col.strip()
        .lower()
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
        for col in df.columns
    ]
    return df


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = standardize_columns(df)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert relevant numeric columns
    numeric_candidates = [
        "age",
        "hours_per_day",
        "bpm",
        "anxiety",
        "depression",
        "insomnia",
        "ocd",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parent-paper-style outlier handling mentioned in methodology
    if "age" in df.columns:
        df = df[(df["age"].isna()) | (df["age"] <= 70)]

    if "bpm" in df.columns:
        df = df[(df["bpm"].isna()) | (df["bpm"] <= 200)]

    if "hours_per_day" in df.columns:
        df = df[(df["hours_per_day"].isna()) | (df["hours_per_day"] <= 15)]

    return df.reset_index(drop=True)


def save_eda_plots(df: pd.DataFrame) -> None:
    sns.set_style("whitegrid")

    if "fav_genre" in df.columns:
        plt.figure(figsize=(10, 5))
        order = df["fav_genre"].fillna("Missing").value_counts().index
        sns.countplot(data=df, x="fav_genre", order=order)
        plt.xticks(rotation=45, ha="right")
        plt.title("Distribution of Favorite Music Genres")
        plt.xlabel("Favorite Genre")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "genre_distribution.png", dpi=300)
        plt.close()

    if "age" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["age"].dropna(), kde=True, bins=25)
        plt.title("Age Distribution of Participants")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=300)
        plt.close()

    for target in ["depression", "anxiety", "insomnia", "ocd"]:
        if "fav_genre" in df.columns and target in df.columns:
            genre_mean = (
                df.groupby("fav_genre", dropna=False)[target]
                .mean()
                .sort_values(ascending=False)
            )

            plt.figure(figsize=(10, 5))
            sns.barplot(x=genre_mean.index, y=genre_mean.values)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Impact of Music Genre on {target.title()} Level")
            plt.xlabel("Music Genre")
            plt.ylabel(f"Average {target.title()} Score")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{target}_by_genre.png", dpi=300)
            plt.close()


def make_binary_target(series: pd.Series, threshold: float = 5.0) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") >= threshold).astype(int)


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [
        col for col in ["age", "hours_per_day", "bpm"] if col in df.columns
    ]

    categorical_features = [
        col
        for col in [
            "primary_streaming_service",
            "while_working",
            "instrumentalist",
            "composer",
            "fav_genre",
            "exploratory",
            "foreign_languages",
            "music_effects",
        ]
        if col in df.columns
    ]

    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", SVC(kernel="rbf", random_state=RANDOM_STATE)),
            ]
        ),
        "Neural Network (MLP)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(100,),
                        max_iter=500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_name: str,
    model_name: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{target_name.title()} - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(OUTPUT_DIR / f"{target_name}_{safe_name}_confusion_matrix.png", dpi=300)
    plt.close()


def evaluate_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    numeric_features, categorical_features = build_feature_lists(df)
    feature_cols = numeric_features + categorical_features

    if not feature_cols:
        raise ValueError("No valid features found for modeling.")

    modeling_df = df[feature_cols + [target_col]].copy()
    modeling_df = modeling_df.dropna(subset=[target_col]).reset_index(drop=True)

    X = modeling_df[feature_cols]
    y = make_binary_target(modeling_df[target_col], threshold=5.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor)

    rows = []

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        save_confusion_matrix(y_test, y_pred, target_col, model_name)

        rows.append(
            {
                "target": target_col,
                "model": model_name,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
            }
        )

    return pd.DataFrame(rows)

def save_model_comparison_graph(results_df: pd.DataFrame) -> None:
    """
    Create a grouped bar chart similar to Fig. 10 in the parent paper.
    Uses one target for comparison. Anxiety is the best choice because
    the parent paper discusses it most clearly.
    """
    target_df = results_df[results_df["target"] == "anxiety"].copy()

    # Match paper order
    model_order = ["SVM", "Neural Network (MLP)", "Random Forest", "Decision Tree"]
    target_df["model"] = pd.Categorical(target_df["model"], categories=model_order, ordered=True)
    target_df = target_df.sort_values("model")

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    x = np.arange(len(target_df))
    width = 0.18

    plt.figure(figsize=(10, 6))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        plt.bar(x + i * width, target_df[metric], width=width, label=label)

    plt.xticks(x + 1.5 * width, target_df["model"], rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.xlabel("Classifier")
    plt.ylabel("Score")
    plt.title("Performance Comparison of Machine Learning Models")
    plt.legend(title="Metrics")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "algorithm_comparison_graph.png", dpi=300)
    plt.close()


def main() -> None:
    print("Loading dataset...")
    df = load_data()
    print(f"Original shape: {df.shape}")

    print("Cleaning data...")
    df = clean_data(df)
    print(f"Cleaned shape: {df.shape}")

    print("Saving EDA plots...")
    save_eda_plots(df)

    targets = [col for col in ["anxiety", "depression", "insomnia", "ocd"] if col in df.columns]
    if not targets:
        raise ValueError("Could not find target columns: anxiety, depression, insomnia, ocd")

    all_results = []

    for target in targets:
        print(f"Running target: {target}")
        results = evaluate_target(df, target)
        all_results.append(results)

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv(OUTPUT_DIR / "parent_reproduction_results.csv", index=False)
    save_model_comparison_graph(final_results) 

    print("\nParent reproduction results:\n")
    print(final_results.to_string(index=False))
    print(f"\nSaved all outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()