import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_path = "DS340musicdataset.csv"
df = pd.read_csv(data_path)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()

    # Drop duplicates
    d = d.drop_duplicates()

    # Median impute Age
    if "Age" in d.columns:
        d["Age"] = d["Age"].fillna(d["Age"].median())

    # BPM: fill missing using genre-specific mode, fallback to overall median
    if "BPM" in d.columns and "Fav genre" in d.columns:
        overall_bpm_median = d["BPM"].median()
        bpm_mode_by_genre = (
            d.groupby("Fav genre")["BPM"]
            .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan)
        )

        def fill_bpm(row):
            if pd.isna(row["BPM"]):
                genre_mode = bpm_mode_by_genre.get(row["Fav genre"], np.nan)
                return overall_bpm_median if pd.isna(genre_mode) else genre_mode
            return row["BPM"]

        d["BPM"] = d.apply(fill_bpm, axis=1)

    # Mode impute key categoricals (if present)
    categorical_cols = [
        "Primary streaming service", "Music effects",
        "While working", "Instrumentalist", "Composer",
        "Exploratory", "Foreign languages", "Fav genre"
    ]
    for col in categorical_cols:
        if col in d.columns and d[col].isna().any():
            d[col] = d[col].fillna(d[col].mode().iloc[0])

    # Outlier removal
    if "Age" in d.columns:
        d = d[d["Age"] <= 70]
    if "BPM" in d.columns:
        d = d[d["BPM"] <= 200]
    if "Hours per day" in d.columns:
        d = d[d["Hours per day"] <= 15]

    return d.reset_index(drop=True)

df_clean = clean_data(df)

# Paper-style split: 80% train, 20% test (no validation)
train_df, test_df = train_test_split(
    df_clean,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# Save
clean_path = "music_cleaned_full_paper.csv"
train_path = "music_train_80_paper.csv"
test_path  = "music_test_20_paper.csv"

df_clean.to_csv(clean_path, index=False)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

(df_clean.shape, train_df.shape, test_df.shape, clean_path, train_path, test_path)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# =========================
# CONFIG (VS Code friendly)
# =========================
RAW_PATH = "DS340musicdataset.csv"   # or your absolute path
RANDOM_STATE = 42
THRESHOLD = 6  # High if score >= 6 (common for this dataset)

# Outputs (optional)
CLEAN_PATH = "music_cleaned_full_paper.csv"
TRAIN_PATH = "music_train_80_paper.csv"
TEST_PATH  = "music_test_20_paper.csv"


# =========================
# 1) Load + Clean (paper-style)
# =========================
df = pd.read_csv(RAW_PATH)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()

    # Drop duplicates
    d = d.drop_duplicates()

    # Median impute Age
    if "Age" in d.columns:
        d["Age"] = d["Age"].fillna(d["Age"].median())

    # BPM: fill missing using genre-specific mode, fallback to overall median
    if "BPM" in d.columns and "Fav genre" in d.columns:
        overall_bpm_median = d["BPM"].median()
        bpm_mode_by_genre = (
            d.groupby("Fav genre")["BPM"]
            .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan)
        )

        def fill_bpm(row):
            if pd.isna(row["BPM"]):
                genre_mode = bpm_mode_by_genre.get(row["Fav genre"], np.nan)
                return overall_bpm_median if pd.isna(genre_mode) else genre_mode
            return row["BPM"]

        d["BPM"] = d.apply(fill_bpm, axis=1)

    # Mode impute key categorical columns (if present)
    categorical_cols = [
        "Primary streaming service", "Music effects",
        "While working", "Instrumentalist", "Composer",
        "Exploratory", "Foreign languages", "Fav genre"
    ]
    for col in categorical_cols:
        if col in d.columns and d[col].isna().any():
            d[col] = d[col].fillna(d[col].mode().iloc[0])

    # Outlier removal (as described)
    if "Age" in d.columns:
        d = d[d["Age"] <= 70]
    if "BPM" in d.columns:
        d = d[d["BPM"] <= 200]
    if "Hours per day" in d.columns:
        d = d[d["Hours per day"] <= 15]

    return d.reset_index(drop=True)

df_clean = clean_data(df)

# Paper split: 80/20 train/test (NO validation)
train_df, test_df = train_test_split(
    df_clean,
    test_size=0.20,
    random_state=RANDOM_STATE,
    shuffle=True
)

# Save files (optional but recommended for reproducibility)
df_clean.to_csv(CLEAN_PATH, index=False)
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("Cleaned:", df_clean.shape)
print("Train  :", train_df.shape)
print("Test   :", test_df.shape)


# =========================
# 2) Figures 2–7 (EDA)
# =========================
def fig2_genre_distribution(d: pd.DataFrame):
    counts = d["Fav genre"].value_counts()

    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Music Genre")
    plt.ylabel("Count")
    plt.title("Distribution of Favorite Music Genres (Fig. 2)")
    plt.tight_layout()
    plt.show()

def fig3_age_distribution(d: pd.DataFrame):
    plt.figure()
    plt.hist(d["Age"].dropna(), bins=25)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Age Distribution of Participants (Fig. 3)")
    plt.tight_layout()
    plt.show()

def bar_mean_by_genre(d: pd.DataFrame, score_col: str, title: str):
    grp = d.groupby("Fav genre")[score_col]
    means = grp.mean().sort_index()
    stds = grp.std().reindex(means.index)

    plt.figure()
    plt.bar(means.index, means.values, yerr=stds.values, capsize=3)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Music Genre")
    plt.ylabel(f"{score_col} (mean ± std)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

fig2_genre_distribution(df_clean)
fig3_age_distribution(df_clean)

bar_mean_by_genre(df_clean, "Depression", "Impact of Music Genre on Depression Level (Fig. 4)")
bar_mean_by_genre(df_clean, "Anxiety",    "Impact of Music Genre on Anxiety Level (Fig. 5)")
bar_mean_by_genre(df_clean, "Insomnia",   "Impact of Music Genre on Insomnia Level (Fig. 6)")
bar_mean_by_genre(df_clean, "OCD",        "Impact of Music Genre on OCD Level (Fig. 7)")


# =========================
# 3) Modeling setup (paper features)
# =========================
FEATURES = ["Fav genre", "Hours per day", "Music effects"]

def make_binary_target(d: pd.DataFrame, col: str) -> pd.Series:
    return (d[col] >= THRESHOLD).astype(int)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Fav genre", "Music effects"]),
        ("num", StandardScaler(), ["Hours per day"]),
    ]
)

# RF GridSearch (paper mentions GridSearchCV + RF)
rf_param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 8, 16],
    "clf__min_samples_split": [2, 10],
    "clf__min_samples_leaf": [1, 5],
}

def fit_best_rf(train_df: pd.DataFrame, target_col: str):
    X_train = train_df[FEATURES]
    y_train = make_binary_target(train_df, target_col)

    pipe = Pipeline(steps=[
        ("pre", preprocess),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    grid = GridSearchCV(pipe, rf_param_grid, cv=5, n_jobs=-1, scoring="f1")
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# =========================
# 4) Fig. 8 Confusion Matrices (RF best model per target)
# =========================
targets = ["Anxiety", "Depression", "Insomnia", "OCD"]

X_test = test_df[FEATURES]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

best_params_by_target = {}

for i, t in enumerate(targets):
    best_model, params = fit_best_rf(train_df, t)
    best_params_by_target[t] = params

    y_test = make_binary_target(test_df, t)
    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Low (0)", "High (1)"])
    disp.plot(ax=axes[i], colorbar=False)
    axes[i].set_title(f"Confusion Matrix - {t}")

plt.tight_layout()
plt.show()

print("\nBest RF params by target:")
for t in targets:
    print(f"{t}: {best_params_by_target[t]}")


# =========================
# 5) Fig. 9 + Fig. 10 Algorithm comparison (on one target like the paper table)
# =========================
def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    X_train = train_df[FEATURES]
    y_train = make_binary_target(train_df, target_col)

    X_test = test_df[FEATURES]
    y_test = make_binary_target(test_df, target_col)

    models = {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=300,
            early_stopping=True,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline(steps=[("pre", preprocess), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rows.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        })

    out = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return out

# The paper’s comparison table typically shows one condition; use Anxiety by default.
results_table = evaluate_models(train_df, test_df, target_col="Anxiety")

print("\nAlgorithm comparison table (Fig. 9 style) — target = Anxiety:")
print(results_table.to_string(index=False))

# Fig. 10: accuracy bar chart
plt.figure()
plt.bar(results_table["Algorithm"], results_table["Accuracy"])
plt.xticks(rotation=25, ha="right")
plt.ylabel("Accuracy")
plt.title("Comparison of Algorithms (Fig. 10) — Accuracy (Target: Anxiety)")
plt.tight_layout()
plt.show()


