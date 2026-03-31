import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================
# FINAL PROJECT PIPELINE: PRESENTATION-READY OUTPUTS ONLY
# Author: Sanjana Vuppunahalli
#
# Uses already-combined train / validation / test splits.
# Keeps the most presentation-useful outputs and styles them.
# ============================================================


# -----------------------------
# USER SETTINGS
# -----------------------------
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs_mxmh_presentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "mxmh_spotify_train.csv"
VALID_PATH = DATA_DIR / "mxmh_spotify_validation.csv"
TEST_PATH = DATA_DIR / "mxmh_spotify_test.csv"

RANDOM_STATE = 42
N_SPLITS = 5
TOP_N_FEATURES = 12
SYMPTOM_THRESHOLD = 5.0
COMPOSITE_THRESHOLD = None
MAIN_TARGET = "overall_distress"   # main target for detailed visuals
FOCUS_TARGETS = ["anxiety", "depression", "overall_distress"]


# -----------------------------
# VISUAL STYLE
# -----------------------------
plt.style.use("seaborn-v0_8-whitegrid")
PRIMARY = "#1f4e79"
SECONDARY = "#f28e2b"
ACCENT = "#d1495b"
MUTED = "#9aa5b1"
DARK = "#1f2933"
POSITIVE = "#2a9d8f"
NEGATIVE = "#e76f51"
CMAP = "Blues"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d9dee3",
    "axes.labelcolor": DARK,
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "font.size": 12,
})


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        col.strip().lower().replace("[", "").replace("]", "")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
        for col in df.columns
    ]
    return df



def save_plot(path: Path, title: str | None = None):
    if title:
        plt.title(title, pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()



def safe_binary_target(series: pd.Series, threshold: float | None = None) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().sum() > 0:
        if threshold is None:
            threshold = numeric_series.median()
        return (numeric_series >= threshold).astype(int)

    lowered = series.astype(str).str.lower().str.strip()
    negative_labels = {
        "poor", "bad", "high", "severe", "depressed", "anxious", "yes", "1", "true"
    }
    return lowered.isin(negative_labels).astype(int)



def get_available_symptom_columns(df: pd.DataFrame) -> list[str]:
    candidates = ["anxiety", "depression", "insomnia", "ocd"]
    return [col for col in candidates if col in df.columns]



def add_engineered_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_candidates = [
        "fav_genre",
        "primary_streaming_service",
        "while_working",
        "instrumentalist",
        "composer",
        "exploratory",
        "foreign_languages",
        "music_effects",
        "permissions",
    ]
    existing = [col for col in text_candidates if col in df.columns]

    if existing:
        df["combined_text"] = df[existing].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        df["combined_text"] = ""

    df["text_length"] = df["combined_text"].astype(str).str.len()
    df["word_count"] = df["combined_text"].astype(str).str.split().str.len()
    return df



def build_targets_for_split(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    symptom_cols = get_available_symptom_columns(df)
    target_names = []

    for col in symptom_cols:
        df[f"target_{col}"] = safe_binary_target(df[col], threshold=SYMPTOM_THRESHOLD)
        target_names.append(col)

    if symptom_cols:
        numeric_symptoms = df[symptom_cols].apply(pd.to_numeric, errors="coerce")
        df["overall_distress_score"] = numeric_symptoms.mean(axis=1)
        threshold = COMPOSITE_THRESHOLD if COMPOSITE_THRESHOLD is not None else df["overall_distress_score"].median()
        df["target_overall_distress"] = (df["overall_distress_score"] >= threshold).astype(int)
        target_names.append("overall_distress")

    return df, target_names



def plot_target_distribution(y: pd.Series, filename: str, title: str):
    counts = y.value_counts().sort_index()
    labels = ["Low", "High"] if set(counts.index.tolist()) <= {0, 1} else counts.index.astype(str)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, counts.values, color=[MUTED, SECONDARY], edgecolor="none")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.ylim(0, max(counts.values) * 1.15)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + max(counts.values) * 0.02, f"{int(height)}", ha="center")

    save_plot(OUTPUT_DIR / filename, title)



def plot_best_f1_by_target(best_rows: pd.DataFrame):
    ordered = best_rows.sort_values("test_f1", ascending=False).copy()
    colors = [SECONDARY if t == "anxiety" else PRIMARY if t == "depression" else ACCENT if t == "overall_distress" else MUTED for t in ordered["target_name"]]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(ordered["target_name"], ordered["test_f1"], color=colors, edgecolor="none")
    plt.ylabel("Best Test F1")
    plt.xlabel("Target")
    plt.ylim(0, max(0.85, ordered["test_f1"].max() + 0.08))

    for bar, val in zip(bars, ordered["test_f1"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center")

    save_plot(OUTPUT_DIR / "01_best_f1_by_target.png", "Best Model Performance by Mental Health Outcome")



def plot_main_target_model_comparison(results_df: pd.DataFrame, target_name: str):
    ordered = results_df.sort_values("test_f1", ascending=False).copy()
    best_model = ordered.iloc[0]["model"]

    colors = []
    for model in ordered["model"]:
        if model == best_model:
            colors.append(SECONDARY)
        elif model in ["linear_svm", "logistic_regression"]:
            colors.append(PRIMARY)
        else:
            colors.append(MUTED)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(ordered["model"], ordered["test_f1"], color=colors, edgecolor="none")
    plt.ylabel("Test F1")
    plt.xlabel("Model")
    plt.xticks(rotation=25, ha="right")

    ymin = max(0, ordered["test_f1"].min() - 0.05)
    ymax = min(1, ordered["test_f1"].max() + 0.08)
    plt.ylim(ymin, ymax)

    for bar, val in zip(bars, ordered["test_f1"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center", fontsize=11)

    save_plot(OUTPUT_DIR / f"02_{target_name}_model_comparison.png", f"{target_name}: Model Comparison on Test Set")



def plot_ablation(ablation_df: pd.DataFrame, target_name: str):
    ordered = ablation_df.copy()
    ordered["display_name"] = ordered["feature_set"].map({
        "survey_only": "Survey Only",
        "survey_plus_spotify": "Survey + Spotify",
    })
    ordered = ordered.sort_values("test_f1", ascending=False)

    colors = [SECONDARY if "Spotify" in name else PRIMARY for name in ordered["display_name"]]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(ordered["display_name"], ordered["test_f1"], color=colors, edgecolor="none")
    plt.ylabel("Test F1")
    plt.xlabel("Feature Set")

    ymin = max(0, ordered["test_f1"].min() - 0.05)
    ymax = min(1, ordered["test_f1"].max() + 0.08)
    plt.ylim(ymin, ymax)

    for bar, val in zip(bars, ordered["test_f1"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center")

    save_plot(OUTPUT_DIR / f"03_{target_name}_ablation.png", f"{target_name}: Survey Only vs Survey + Spotify")



def plot_permutation_importance(importance_df: pd.DataFrame, target_name: str):
    top_df = importance_df.head(TOP_N_FEATURES).sort_values("importance_mean", ascending=True).copy()
    colors = [PRIMARY] * len(top_df)
    for i in range(min(3, len(top_df))):
        colors[-(i + 1)] = SECONDARY

    plt.figure(figsize=(9, 6))
    bars = plt.barh(top_df["feature"], top_df["importance_mean"], color=colors, edgecolor="none")
    plt.xlabel("Permutation Importance (F1 drop)")
    plt.ylabel("Feature")

    for bar, val in zip(bars, top_df["importance_mean"]):
        plt.text(val + 0.0008, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=10)

    save_plot(OUTPUT_DIR / f"04_{target_name}_permutation_importance.png", f"{target_name}: Top Permutation Importance Features")



def plot_confusion_matrix(cm: np.ndarray, filename: str, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=CMAP)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, ["Low", "High"])
    plt.yticks(tick_marks, ["Low", "High"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else DARK
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=18, fontweight="bold")

    save_plot(OUTPUT_DIR / filename, title)



def get_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )



def build_models(preprocessor):
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_split=4,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=250,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }



def get_feature_names_from_pipeline(pipeline: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []
    feature_names.extend(numeric_features)

    if categorical_features:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(categorical_features).tolist()
            feature_names.extend(cat_names)
        except Exception:
            feature_names.extend(categorical_features)

    return feature_names



def get_permutation_importance_df(best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1",
    )

    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)



def evaluate_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, target_name: str):
    rows = []
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fitted_models = {}
    test_predictions = {}

    for model_name, model in models.items():
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "f1": "f1",
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
            },
            return_train_score=False,
        )

        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        test_predictions[model_name] = test_pred

        rows.append({
            "target_name": target_name,
            "model": model_name,
            "cv_f1_mean": np.mean(cv_results["test_f1"]),
            "cv_f1_std": np.std(cv_results["test_f1"]),
            "validation_f1": f1_score(y_val, val_pred),
            "validation_accuracy": accuracy_score(y_val, val_pred),
            "validation_balanced_accuracy": balanced_accuracy_score(y_val, val_pred),
            "test_f1": f1_score(y_test, test_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
            "test_precision": precision_score(y_test, test_pred, zero_division=0),
            "test_recall": recall_score(y_test, test_pred, zero_division=0),
        })

    results_df = pd.DataFrame(rows).sort_values(["validation_f1", "test_f1"], ascending=[False, False])
    return results_df, fitted_models, test_predictions



def run_ablation(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, numeric_features: list[str], categorical_features: list[str], spotify_feature_candidates: list[str], target_name: str):
    spotify_present = [col for col in numeric_features if col in spotify_feature_candidates]
    survey_only_numeric = [col for col in numeric_features if col not in spotify_present]

    rows = []
    feature_sets = {
        "survey_only": survey_only_numeric + categorical_features,
        "survey_plus_spotify": numeric_features + categorical_features,
    }

    for label, feature_list in feature_sets.items():
        X_train_variant = train_df[feature_list].copy()
        X_val_variant = val_df[feature_list].copy()
        X_test_variant = test_df[feature_list].copy()

        numeric_variant = X_train_variant.select_dtypes(include=[np.number]).columns.tolist()
        categorical_variant = [col for col in X_train_variant.columns if col not in numeric_variant]

        variant_preprocessor = get_preprocessor(numeric_variant, categorical_variant)
        variant_model = Pipeline(
            steps=[
                ("preprocessor", variant_preprocessor),
                ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )

        variant_model.fit(X_train_variant, y_train)
        val_pred = variant_model.predict(X_val_variant)
        test_pred = variant_model.predict(X_test_variant)

        rows.append({
            "target_name": target_name,
            "feature_set": label,
            "validation_f1": f1_score(y_val, val_pred),
            "test_f1": f1_score(y_test, test_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
        })

    return pd.DataFrame(rows).sort_values("test_f1", ascending=False)


# -----------------------------
# MAIN WORKFLOW
# -----------------------------
def main():
    print("Loading combined split datasets...")
    train_df = standardize_columns(pd.read_csv(TRAIN_PATH))
    val_df = standardize_columns(pd.read_csv(VALID_PATH))
    test_df = standardize_columns(pd.read_csv(TEST_PATH))

    train_df = add_engineered_text_columns(train_df)
    val_df = add_engineered_text_columns(val_df)
    test_df = add_engineered_text_columns(test_df)

    train_df, target_names = build_targets_for_split(train_df)
    val_df, _ = build_targets_for_split(val_df)
    test_df, _ = build_targets_for_split(test_df)

    if not target_names:
        raise ValueError("No usable target columns found. Expected anxiety, depression, insomnia, or ocd.")

    combined_for_visuals = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_for_visuals.to_csv(OUTPUT_DIR / "combined_dataset_used_for_final_project.csv", index=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Targets: {target_names}")

    spotify_feature_candidates = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "popularity",
    ]

    numeric_candidates = [
        "age",
        "hours_per_day",
        "bpm",
        "text_length",
        "word_count",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "popularity",
    ]

    categorical_candidates = [
        "primary_streaming_service",
        "while_working",
        "instrumentalist",
        "composer",
        "fav_genre",
        "exploratory",
        "foreign_languages",
        "music_effects",
    ]

    numeric_features = [col for col in numeric_candidates if col in train_df.columns]
    categorical_features = [col for col in categorical_candidates if col in train_df.columns]

    all_results = []
    all_ablation = []
    target_best_rows = []

    for target_name in target_names:
        target_col = f"target_{target_name}"
        y_train = train_df[target_col].copy()
        y_val = val_df[target_col].copy()
        y_test = test_df[target_col].copy()

        feature_cols = numeric_features + categorical_features
        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        preprocessor = get_preprocessor(numeric_features, categorical_features)
        models = build_models(preprocessor)

        results_df, fitted_models, test_predictions = evaluate_models(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            target_name=target_name,
        )

        results_df.to_csv(OUTPUT_DIR / f"{target_name}_model_comparison_results.csv", index=False)
        all_results.append(results_df)

        best_model_name = results_df.iloc[0]["model"]
        best_model = fitted_models[best_model_name]
        best_pred = test_predictions[best_model_name]

        target_best_rows.append(results_df.iloc[0].to_dict())

        ablation_df = run_ablation(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            spotify_feature_candidates=spotify_feature_candidates,
            target_name=target_name,
        )
        ablation_df.to_csv(OUTPUT_DIR / f"{target_name}_feature_set_ablation.csv", index=False)
        all_ablation.append(ablation_df)

        if target_name in FOCUS_TARGETS:
            plot_target_distribution(
                combined_for_visuals[target_col],
                filename=f"{target_name}_target_distribution.png",
                title=f"{target_name}: Target Distribution",
            )

        if target_name == MAIN_TARGET:
            plot_main_target_model_comparison(results_df, target_name)
            plot_ablation(ablation_df, target_name)

            importance_df = get_permutation_importance_df(best_model, X_test, y_test)
            importance_df.to_csv(OUTPUT_DIR / f"{target_name}_permutation_importance.csv", index=False)
            plot_permutation_importance(importance_df, target_name)

            cm = confusion_matrix(y_test, best_pred)
            plot_confusion_matrix(
                cm,
                filename=f"05_{target_name}_best_confusion_matrix_{best_model_name}.png",
                title=f"{target_name}: Confusion Matrix ({best_model_name})",
            )

    final_results_df = pd.concat(all_results, ignore_index=True).sort_values(["target_name", "validation_f1", "test_f1"], ascending=[True, False, False])
    final_ablation_df = pd.concat(all_ablation, ignore_index=True)
    best_rows_df = pd.DataFrame(target_best_rows).sort_values("test_f1", ascending=False)

    final_results_df.to_csv(OUTPUT_DIR / "all_targets_model_results.csv", index=False)
    final_ablation_df.to_csv(OUTPUT_DIR / "all_targets_ablation_results.csv", index=False)
    best_rows_df.to_csv(OUTPUT_DIR / "best_model_by_target.csv", index=False)

    plot_best_f1_by_target(best_rows_df)

    summary_lines = [
        "FINAL PROJECT SUMMARY",
        f"Train rows: {len(train_df)}",
        f"Validation rows: {len(val_df)}",
        f"Test rows: {len(test_df)}",
        f"Targets run: {', '.join(target_names)}",
        "",
        "Best model by target:",
        best_rows_df.to_string(index=False),
        "",
        "All results:",
        final_results_df.to_string(index=False),
        "",
        "Ablation results:",
        final_ablation_df.to_string(index=False),
    ]

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("Done.")
    print(f"Presentation outputs saved to: {OUTPUT_DIR.resolve()}")
    print("Recommended slides from this folder:")
    print("01_best_f1_by_target.png")
    print(f"02_{MAIN_TARGET}_model_comparison.png")
    print(f"03_{MAIN_TARGET}_ablation.png")
    print(f"04_{MAIN_TARGET}_permutation_importance.png")
    print(f"05_{MAIN_TARGET}_best_confusion_matrix_<best model>.png")


if __name__ == "__main__":
    main()
