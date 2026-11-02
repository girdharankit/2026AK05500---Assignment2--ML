"""
app.py - Streamlit front-end for ML Assignment 2
Breast Cancer Wisconsin (Diagnostic): evaluate 5 pre-trained classifiers on uploaded test data.

The app ONLY loads the saved models from model/*.pkl and evaluates them.
It never retrains on the uploaded data.

Run locally:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve,
)

from data_utils import (CLASS_NAMES, FEATURE_COLUMNS, TARGET_COLUMN, encode_target,
                        positive_class_scores)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_TEST_CSV = APP_DIR / "test_data.csv"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
PALETTE = {"benign": "#3a86ff", "malignant": "#e63946", "ink": "#1d3557"}

st.set_page_config(page_title="2026AK05500 - Machine Learning Assignment 2", page_icon="🔬", layout="wide")

st.markdown(
    f"""
    <style>
      .hero {{padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
              background: linear-gradient(120deg, {PALETTE['ink']} 0%, {PALETTE['benign']} 100%);
              color: #fff;}}
      .hero h1 {{margin: 0; font-size: 1.7rem; color: #fff;}}
      .hero p  {{margin: .3rem 0 0; opacity: .9;}}
      div[data-testid="stMetric"] {{background: rgba(42,157,143,.08); border-radius: 10px;
              padding: .6rem .8rem; border-left: 4px solid {PALETTE['benign']};}}
    </style>
    <div class="hero">
      <h1>🔬 2026AK05500 - ML Assignment 2</h1>
      <p>UCI Breast Cancer Wisconsin (Diagnostic) · 5 pre-trained models ·
         upload test data → instant evaluation (no retraining)</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading saved models…")
def load_models():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["models"].items()}
    return models, meta


@st.cache_data(show_spinner=False)
def read_csv(file_or_path):
    return pd.read_csv(file_or_path)


def prepare_data(raw: pd.DataFrame):
    """Validate columns, encode target, drop incomplete rows. Returns X, y, notes."""
    notes = []
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"The uploaded CSV is missing {len(missing)} required column(s): "
            + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
        )
    extra = [c for c in df.columns if c not in FEATURE_COLUMNS + [TARGET_COLUMN]]
    if extra:
        notes.append(f"Ignored extra column(s): {', '.join(extra)}")

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} row(s) with missing / non-numeric values.")
    if df.empty:
        raise ValueError("No usable rows left after cleaning.")

    y = encode_target(df[TARGET_COLUMN])
    return df[FEATURE_COLUMNS], y, notes


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_score = positive_class_scores(model, X)
    both_classes = y.nunique() == 2
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score) if both_classes else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred) if both_classes else np.nan,
    }
    return metrics, y_pred, y_score


def plot_confusion(y, y_pred, title):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 15}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_roc(curves):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (Recall)")
    ax.set_title("ROC curve – Malignant as positive class", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar: data + model choice
# --------------------------------------------------------------------------- #
try:
    models, meta = load_models()
except FileNotFoundError:
    st.error("Saved models not found in `model/`. Run `python train_models.py` first, "
             "then commit the `model/` folder to GitHub.")
    st.stop()

if meta.get("sklearn_version") != sklearn.__version__:
    st.warning(f"Models were trained with scikit-learn {meta.get('sklearn_version')} but this "
               f"server runs {sklearn.__version__}. Pin the same version in requirements.txt.")

with st.sidebar:
    st.header("① Test data")
    uploaded = st.file_uploader("Upload test CSV (30 features + `diagnosis`)", type=["csv"])
    if SAMPLE_TEST_CSV.exists():
        st.download_button("⬇️ Download sample test_data.csv",
                           SAMPLE_TEST_CSV.read_bytes(), "test_data.csv", "text/csv")
    st.caption("`diagnosis` may be M/B, Malignant/Benign or 1/0 (1 = Malignant).")

    st.header("② Model")
    model_names = list(models.keys())
    chosen = st.selectbox("Choose a classifier", model_names,
                          index=model_names.index("Support Vector Machine")
                          if "Support Vector Machine" in model_names else 0)
    st.caption("Best hyper-parameters (5-fold CV on training data):")
    st.json(meta["best_params"].get(chosen, {}), expanded=False)

    st.divider()
    st.caption(f"Trained {meta.get('trained_at', '?')} · scikit-learn {meta.get('sklearn_version')} "
               f"· {meta.get('train_rows')} train rows")

if uploaded is not None:
    source_label = f"Uploaded file: **{uploaded.name}**"
    raw_df = read_csv(uploaded)
elif SAMPLE_TEST_CSV.exists():
    source_label = "No file uploaded – showing results on the bundled **test_data.csv**"
    raw_df = read_csv(str(SAMPLE_TEST_CSV))
else:
    st.info("Upload a test CSV from the sidebar to begin.")
    st.stop()

try:
    X, y, notes = prepare_data(raw_df)
except ValueError as err:
    st.error(str(err))
    st.markdown("**Expected columns:** " + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS + [TARGET_COLUMN]))
    st.stop()

st.markdown(source_label)
for n in notes:
    st.caption("ℹ️ " + n)
c1, c2, c3 = st.columns(3)
c1.metric("Test rows", len(X))
c2.metric("Malignant (1)", int((y == 1).sum()))
c3.metric("Benign (0)", int((y == 0).sum()))

tab_single, tab_compare, tab_pred, tab_about = st.tabs(
    ["📊 Selected model", "🏆 Compare all 5 models", "🧾 Predictions", "📚 About the dataset"])

# --------------------------------------------------------------------------- #
# Tab 1 - selected model
# --------------------------------------------------------------------------- #
with tab_single:
    model = models[chosen]
    metrics, y_pred, y_score = compute_metrics(model, X, y)

    st.subheader(f"{chosen} – evaluation metrics")
    cols = st.columns(len(METRIC_NAMES))
    for col, name in zip(cols, METRIC_NAMES):
        val = metrics[name]
        col.metric(name, "n/a" if pd.isna(val) else f"{val:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion(y, y_pred, chosen), clear_figure=True)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=[0, 1],
                                       target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        st.caption(f"Missed cancers (false negatives): **{fn}** · False alarms (false positives): **{fp}**")

# --------------------------------------------------------------------------- #
# Tab 2 - all models side by side
# --------------------------------------------------------------------------- #
with tab_compare:
    rows, curves = {}, {}
    for name, mdl in models.items():
        m, _, score = compute_metrics(mdl, X, y)
        rows[name] = m
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, score)
            curves[name] = (fpr, tpr, m["AUC"])
    comp = pd.DataFrame(rows).T[METRIC_NAMES]
    comp.index.name = "ML Model Name"

    st.subheader("Comparison table (on the data shown above)")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cdeee9"),
                 use_container_width=True)
    if comp["MCC"].notna().any():
        best = comp.sort_values(["MCC", "F1"], ascending=False).index[0]
        st.success(f"🏆 Best model on this data (highest MCC, ties broken by F1): **{best}**")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        comp[["Accuracy", "F1", "MCC"]].plot.barh(ax=ax, color=["#264653", "#2a9d8f", "#e9c46a"])
        ax.set_xlim(max(0.0, np.nanmin(comp[["Accuracy", "F1", "MCC"]].values) - 0.05), 1.0)
        ax.set_title("Accuracy / F1 / MCC", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with right:
        if curves:
            st.pyplot(plot_roc(curves), clear_figure=True)

    st.download_button("⬇️ Download comparison as CSV", comp.round(4).to_csv().encode(),
                       "model_comparison.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Tab 3 - row-level predictions
# --------------------------------------------------------------------------- #
with tab_pred:
    out = X.copy()
    out["actual"] = y.map(CLASS_NAMES).values
    out["predicted"] = pd.Series(y_pred, index=X.index).map(CLASS_NAMES)
    out["correct"] = np.where(out["actual"] == out["predicted"], "✅", "❌")
    only_wrong = st.toggle("Show only misclassified rows", value=False)
    view = out[out["correct"] == "❌"] if only_wrong else out
    st.dataframe(view[["actual", "predicted", "correct"] + FEATURE_COLUMNS[:6]],
                 use_container_width=True, height=380)
    st.download_button(f"⬇️ Download {chosen} predictions", out.to_csv(index=False).encode(),
                       f"predictions_{chosen.lower().replace(' ', '_')}.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Tab 4 - dataset info
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
**UCI Breast Cancer Wisconsin (Diagnostic)** – 569 fine-needle-aspirate (FNA) samples of breast
masses, 30 real-valued features, binary target (**Malignant = 1**, **Benign = 0**).

Ten measurements of the cell nuclei are computed from each digitised image – radius, texture,
perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal
dimension – and each is reported as its **mean**, **standard error (_se)** and **worst** (mean of
the three largest values), giving 30 features.

* Models were trained on a stratified 80 % split; the remaining 20 % is `test_data.csv`.
* Every saved model is a scikit-learn *Pipeline* (scaling included), so raw features can be uploaded.
* Positive class for Precision / Recall / F1 / AUC is **Malignant**.
        """
    )
    st.dataframe(X.describe().T.round(3), use_container_width=True, height=300)
