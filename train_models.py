"""
train_models.py
---------------
ML Assignment 2 - Breast Cancer Wisconsin (Diagnostic) classification.

What this script does
  1. Loads the dataset (UCI online first, local copy if offline) - see data_utils.py
  2. Checks the assignment constraints (>= 12 features, >= 500 instances)
  3. Splits 80 / 20 (stratified) and writes the 20 % hold-out to test_data.csv
  4. Trains 5 classifiers (each tuned with 5-fold GridSearchCV on the TRAIN split only)
       Logistic Regression | Decision Tree | KNN | Gaussian Naive Bayes | SVM
  5. Evaluates every model on the test split:
       Accuracy, AUC, Precision, Recall, F1, MCC
  6. Saves the trained pipelines to model/*.pkl plus model/metadata.json
  7. Writes results/metrics_comparison.csv and updates the metrics table in README.md
  8. Pins the installed scikit-learn version in requirements.txt (pickle compatibility)

Run:   python train_models.py              (tries UCI online, falls back to local copy)
       python train_models.py --offline    (skip the internet completely)
"""

import argparse
import json
import platform
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_utils import (CLASS_NAMES, FEATURE_COLUMNS, TARGET_COLUMN, load_dataset,
                        positive_class_scores)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
RESULTS_DIR = PROJECT_DIR / "results"
TEST_CSV = PROJECT_DIR / "test_data.csv"
README = PROJECT_DIR / "README.md"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_FEATURES, MIN_INSTANCES = 12, 500


# --------------------------------------------------------------------------- #
# Model definitions: (display name, file slug, pipeline, hyper-parameter grid)
# Scaling lives INSIDE each pipeline, so the saved .pkl is self-contained and
# the Streamlit app can feed raw test features straight in.
# --------------------------------------------------------------------------- #
def build_model_specs():
    return [
        (
            "Logistic Regression", "logistic_regression",
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        (
            "Decision Tree", "decision_tree",
            Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {"clf__max_depth": [3, 4, 5, 6, 8, None],
             "clf__min_samples_leaf": [1, 3, 5, 10],
             "clf__criterion": ["gini", "entropy"]},
        ),
        (
            "KNN", "knn",
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15],
             "clf__weights": ["uniform", "distance"]},
        ),
        (
            "Naive Bayes", "naive_bayes",
            Pipeline([("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        ),
        (
            "Support Vector Machine", "svm",
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE))]),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]},
        ),
    ]


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_score = positive_class_scores(model, X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def metrics_markdown(df: pd.DataFrame) -> str:
    cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    lines = ["| ML Model Name | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, row in df.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{row[c]:.4f}" for c in cols) + " |")
    return "\n".join(lines)


def update_readme(table_md: str) -> None:
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- METRICS_TABLE_START -->)(.*?)(<!-- METRICS_TABLE_END -->)", re.S)
    if pattern.search(text):
        text = pattern.sub(lambda m: f"{m.group(1)}\n{table_md}\n{m.group(3)}", text)
        README.write_text(text, encoding="utf-8")
        print("[readme] Metrics table in README.md updated.")


def pin_sklearn_version() -> None:
    """Pickled models must be loaded with the same scikit-learn version."""
    if not REQUIREMENTS.exists():
        return
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    new = [f"scikit-learn=={sklearn.__version__}"
           if re.match(r"^\s*scikit-learn\b", l, re.I) else l for l in lines]
    REQUIREMENTS.write_text("\n".join(new) + "\n", encoding="utf-8")
    print(f"[requirements] scikit-learn pinned to {sklearn.__version__}")


def main():
    parser = argparse.ArgumentParser(description="Train the 5 classifiers.")
    parser.add_argument("--offline", action="store_true",
                        help="Do not try the internet; use the local copy.")
    args = parser.parse_args()

    print("=" * 72)
    print(" ML Assignment 2 - Breast Cancer Wisconsin (Diagnostic) ")
    print(f" Run at {datetime.now():%Y-%m-%d %H:%M:%S} on {platform.node()} "
          f"| Python {platform.python_version()} | scikit-learn {sklearn.__version__}")
    print("=" * 72)

    # ---------------- 1. Load + validate dataset ---------------- #
    df, source = load_dataset(prefer_online=not args.offline)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    n_rows, n_feat = X.shape
    print(f"[data] Instances: {n_rows} | Features: {n_feat} | Missing values: {int(X.isna().sum().sum())}")
    print(f"[data] Class balance: Benign={int((y == 0).sum())}, Malignant={int((y == 1).sum())}")
    assert n_feat >= MIN_FEATURES, f"Need >= {MIN_FEATURES} features"
    assert n_rows >= MIN_INSTANCES, f"Need >= {MIN_INSTANCES} instances"
    print(f"[check] Assignment constraints satisfied (>= {MIN_FEATURES} features, >= {MIN_INSTANCES} rows)")

    # ---------------- 2. Train / test split ---------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.map({1: "M", 0: "B"})
    test_df.to_csv(TEST_CSV, index=False)
    print(f"[split] Train: {len(X_train)} rows | Test: {len(X_test)} rows -> saved to {TEST_CSV.name}")

    # ---------------- 3. Train + evaluate ---------------- #
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, model_files, best_params, reports = {}, {}, {}, []
    training_log = []
    for name, slug, pipe, grid in build_model_specs():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        results[name] = evaluate(best, X_test, y_test)
        y_pred = best.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        path = MODEL_DIR / f"{slug}.pkl"
        joblib.dump(best, path)
        model_files[name] = path.name
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        print("\n" + "-" * 72)
        block = [
            f" {name}  (CV F1 = {search.best_score_:.4f})  -> {path.relative_to(PROJECT_DIR).as_posix()}",
            f" Best params: {best_params[name]}",
            " Test metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in results[name].items()),
            f" Confusion matrix [[TN FP] [FN TP]]: {cm.tolist()}",
            f"   -> TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}",
        ]
        print("\n".join(block))
        training_log.extend(block + [""])
        reports.append(f"===== {name} =====\n" + classification_report(
            y_test, y_pred, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]], digits=4))

    # ---------------- 4. Save results ---------------- #
    metrics_df = pd.DataFrame(results).T.round(4)
    metrics_df.index.name = "ML Model Name"
    metrics_df.to_csv(RESULTS_DIR / "metrics_comparison.csv")
    (RESULTS_DIR / "classification_reports.txt").write_text("\n".join(reports), encoding="utf-8")
    (RESULTS_DIR / "training_log.txt").write_text(
        f"Data source: {source}\nRANDOM_STATE = {RANDOM_STATE}\n\n" + "\n".join(training_log),
        encoding="utf-8")

    metadata = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic) - id 17",
        "data_source_used": source,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "positive_class": "Malignant (1)",
        "models": model_files,
        "best_params": best_params,
        "test_metrics": {m: {k: round(float(v), 4) for k, v in r.items()} for m, r in results.items()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    table_md = metrics_markdown(metrics_df)
    print("\n" + "=" * 72)
    print(" COMPARISON TABLE (test set)")
    print("=" * 72)
    print(table_md)
    winner = metrics_df.sort_values(["MCC", "F1", "AUC"], ascending=False).index[0]
    print(f"\n Best model by MCC: {winner}")

    update_readme(table_md)
    pin_sklearn_version()
    print("\nDone. Next: streamlit run app.py")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
