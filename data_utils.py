"""
data_utils.py
-------------
Loading helpers for the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

Load order (first one that works is used):
  1. ONLINE  - UCI ML Repository via the `ucimlrepo` package (dataset id = 17)
  2. ONLINE  - Raw `wdbc.data` file straight from the UCI archive URL
  3. OFFLINE - Local copy saved in  data/wdbc.csv  (refreshed after every online load)
  4. OFFLINE - Copy bundled inside scikit-learn (`load_breast_cancer`) - same UCI data

Target encoding used everywhere in this project:
  diagnosis = 1  -> Malignant (M)  (positive class)
  diagnosis = 0  -> Benign    (B)
"""

from pathlib import Path
import io
import urllib.request

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_COPY = PROJECT_DIR / "data" / "wdbc.csv"

UCI_DATASET_ID = 17
UCI_RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)

# 10 cell-nucleus measurements, each reported as mean / standard error / worst
_MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
FEATURE_COLUMNS = (
    [f"{m}_mean" for m in _MEASUREMENTS]
    + [f"{m}_se" for m in _MEASUREMENTS]
    + [f"{m}_worst" for m in _MEASUREMENTS]
)
TARGET_COLUMN = "diagnosis"
CLASS_NAMES = {0: "Benign", 1: "Malignant"}


# --------------------------------------------------------------------------- #
# Target encoding
# --------------------------------------------------------------------------- #
def encode_target(series: pd.Series) -> pd.Series:
    """Convert M/B, Malignant/Benign, 1/0 (str or int) to integers 1/0."""
    mapping = {"m": 1, "malignant": 1, "1": 1, "1.0": 1,
               "b": 0, "benign": 0, "0": 0, "0.0": 0}
    encoded = series.astype(str).str.strip().str.lower().map(mapping)
    if encoded.isna().any():
        bad = sorted(series[encoded.isna()].astype(str).unique())[:5]
        raise ValueError(
            f"Column '{TARGET_COLUMN}' has unrecognised values {bad}. "
            "Use M/B, Malignant/Benign or 1/0."
        )
    return encoded.astype(int)


# --------------------------------------------------------------------------- #
# Individual sources
# --------------------------------------------------------------------------- #
def _from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo  # imported lazily: optional dependency

    ds = fetch_ucirepo(id=UCI_DATASET_ID)
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0]
    if X.shape[1] != 30:
        raise ValueError(f"Expected 30 features from UCI, got {X.shape[1]}")
    X.columns = FEATURE_COLUMNS  # same order as UCI: 10 means, 10 SEs, 10 worsts
    X[TARGET_COLUMN] = encode_target(y).values
    return X


def _from_uci_url(timeout: int = 20) -> pd.DataFrame:
    with urllib.request.urlopen(UCI_RAW_URL, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), header=None)
    df.columns = ["id", TARGET_COLUMN] + FEATURE_COLUMNS
    df[TARGET_COLUMN] = encode_target(df[TARGET_COLUMN])
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def _from_local_copy() -> pd.DataFrame:
    if not LOCAL_COPY.exists():
        raise FileNotFoundError(f"No local copy at {LOCAL_COPY}")
    df = pd.read_csv(LOCAL_COPY)
    df[TARGET_COLUMN] = encode_target(df[TARGET_COLUMN])
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def _from_sklearn() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=FEATURE_COLUMNS)
    # scikit-learn encodes 0 = malignant, 1 = benign -> flip to our convention
    df[TARGET_COLUMN] = 1 - bunch.target
    return df


def save_local_copy(df: pd.DataFrame) -> None:
    """Store the dataset (with M/B labels, like the UCI original) for offline use."""
    LOCAL_COPY.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out[TARGET_COLUMN] = out[TARGET_COLUMN].map({1: "M", 0: "B"})
    out.to_csv(LOCAL_COPY, index=False)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_dataset(prefer_online: bool = True, verbose: bool = True):
    """Return (dataframe, source_description)."""
    online = [
        ("UCI ML Repository (ucimlrepo, id=17)", _from_ucimlrepo),
        ("UCI archive URL (wdbc.data)", _from_uci_url),
    ]
    offline = [
        (f"Local copy ({LOCAL_COPY.relative_to(PROJECT_DIR)})", _from_local_copy),
        ("scikit-learn bundled copy of UCI WDBC", _from_sklearn),
    ]
    sources = (online + offline) if prefer_online else offline

    errors = []
    for name, loader in sources:
        try:
            df = loader()
            if verbose:
                print(f"[data] Loaded dataset from: {name}")
            if loader in (_from_ucimlrepo, _from_uci_url):
                save_local_copy(df)
                if verbose:
                    print(f"[data] Local copy refreshed at {LOCAL_COPY}")
            elif not LOCAL_COPY.exists():
                save_local_copy(df)
            return df.reset_index(drop=True), name
        except Exception as exc:  # noqa: BLE001 - try the next source
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"[data] Could not load from {name} -> {type(exc).__name__}")

    raise RuntimeError("All data sources failed:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------- #
# Scores for the positive class (used for AUC and ROC curves)
# --------------------------------------------------------------------------- #
def positive_class_scores(model, X):
    """Probability of class 1 if the model supports it, else the decision score.

    SVC is trained without probability=True (deprecated in recent scikit-learn),
    so its decision_function() is used - AUC only needs a ranking score.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)
