# 2026AK05500– Assignment 2 ML 

**M.Tech (AIML), BITS Pilani WILP – Machine Learning, Assignment 2 - Ankit Girdhar (2026AK05500)**

- **Live Streamlit app:** `https://<your-app-name>.streamlit.app`   ← replace after deployment
- **GitHub repository:** `https://github.com/<your-username>/<your-repo-name>`   ← replace

---

## a. Problem statement

Breast cancer is diagnosed by examining cells taken from a breast mass with a fine-needle
aspirate (FNA). The goal of this assignment is to **predict whether a breast mass is Malignant
(cancerous) or Benign** from 30 numeric measurements of the cell nuclei in the FNA image –
a **binary classification** problem.

Five classification models are trained on the same dataset, evaluated with six metrics
(Accuracy, AUC, Precision, Recall, F1, MCC), compared, and deployed in an interactive
Streamlit web application that evaluates the saved models on uploaded test data without
retraining.

Because a missed cancer (false negative) is far more costly than a false alarm, **Recall** on the
Malignant class and the balanced **MCC** score are given extra weight when choosing the best model.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) – WDBC |
| Source | UCI Machine Learning Repository, dataset id 17 – <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic> |
| Instances | **569** (requirement: ≥ 500) |
| Features | **30** real-valued features (requirement: ≥ 12) – the `ID` column is dropped |
| Target | `diagnosis` – **M = Malignant (1)**, **B = Benign (0)** |
| Class balance | 357 Benign (62.7 %) / 212 Malignant (37.3 %) |
| Missing values | None |

**Features.** Ten characteristics are computed for each cell nucleus in the digitised image:
radius, texture, perimeter, area, smoothness, compactness, concavity, concave points,
symmetry and fractal dimension. For each image the **mean**, **standard error (`_se`)** and
**worst** (mean of the three largest values) of every characteristic are recorded,
giving 3 × 10 = 30 features (e.g. `radius_mean`, `radius_se`, `radius_worst`).

**Loading strategy (online first, offline fallback)** – implemented in `data_utils.py`:
1. UCI ML Repository through the `ucimlrepo` package (`fetch_ucirepo(id=17)`)
2. Raw `wdbc.data` file from the UCI archive URL
3. Local copy `data/wdbc.csv` (refreshed automatically after every successful online load)
4. scikit-learn's bundled copy of the same UCI dataset (`load_breast_cancer`)

**Pre-processing.** Stratified 80/20 train-test split (`random_state = 42`) → 455 training rows,
114 test rows (saved as `test_data.csv`). Features are standardised with `StandardScaler` inside
each model's pipeline for scale-sensitive models (Logistic Regression, KNN, SVM). Hyper-parameters
were tuned with 5-fold stratified `GridSearchCV` (scoring = F1) on the **training data only**.

## c. GitHub repository link

`https://github.com/<your-username>/<your-repo-name>`   ← replace

```
project-folder/
│-- app.py                    # Streamlit application
│-- train_models.py           # trains, tunes, evaluates and saves the 5 models
│-- data_utils.py             # dataset loading (UCI online → local copy) & helpers
│-- ML_Assignment2_Colab.ipynb  # notebook to run training in Google Colab / Jupyter
│-- requirements.txt
│-- README.md
│-- test_data.csv             # 20 % hold-out test set (30 features + diagnosis)
│-- data/wdbc.csv             # local copy of the full dataset (offline fallback)
│-- model/                    # saved trained models
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── metadata.json         # feature list, best hyper-parameters, test metrics
│-- results/
│   ├── metrics_comparison.csv
│   └── classification_reports.txt
└── .streamlit/config.toml    # app theme
```

## d. Models used

All five models were trained on the same training split and evaluated on the same 114-row test set.
Positive class = **Malignant**.

| Model | Pipeline | Best hyper-parameters (5-fold CV) |
|---|---|---|
| Logistic Regression | StandardScaler → LogisticRegression | C = 1 |
| Decision Tree | DecisionTreeClassifier | criterion = entropy, max_depth = 4, min_samples_leaf = 1 |
| KNN | StandardScaler → KNeighborsClassifier | n_neighbors = 3, weights = uniform |
| Naive Bayes (Gaussian) | GaussianNB | var_smoothing = 1e-9 |
| Support Vector Machine | StandardScaler → SVC (RBF kernel) | C = 10, gamma = scale |

### Comparison table (test set, n = 114)

<!-- METRICS_TABLE_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9431 | 1.0000 | 0.8095 | 0.8947 | 0.8536 |
| KNN | 0.9386 | 0.9825 | 0.9730 | 0.8571 | 0.9114 | 0.8688 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9927 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
<!-- METRICS_TABLE_END -->

Confusion matrices on the test set (72 Benign, 42 Malignant):

| Model | TN | FP | FN (missed cancers) | TP |
|---|---|---|---|---|
| Logistic Regression | 71 | 1 | 3 | 39 |
| Decision Tree | 72 | 0 | 8 | 34 |
| KNN | 71 | 1 | 6 | 36 |
| Naive Bayes | 72 | 0 | 7 | 35 |
| Support Vector Machine | 72 | 0 | 3 | 39 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This model did really well even though it is a simple linear model. It got 96.49% accuracy, F1 of 0.9512 and MCC of 0.9245, which puts it in second place. Its AUC of 0.9960 was the highest of all five models. Out of 42 malignant cases in the test set it missed 3, and it wrongly marked 1 benign case as malignant. I think it works well because after scaling the data the two classes can be separated almost with a straight line. |
| Decision Tree | The decision tree gave the weakest results in my comparison, with 92.98% accuracy, F1 of 0.8947 and MCC of 0.8536. It did not give any false alarm (precision 1.0) but it missed 8 malignant cases, so recall was only 0.8095. For cancer detection this is the worst kind of mistake. A single tree splits the data on one feature at a time, so the boundary it learns is not smooth. I kept max depth at 4 to avoid overfitting, and the AUC (0.9431) was also the lowest. |
| KNN | KNN gave 93.86% accuracy, F1 of 0.9114 and MCC of 0.8688, so it was in the middle. It missed 6 malignant cases and had 1 false alarm. Since KNN works on distance, scaling the features was very important, otherwise features like area with big values would control the result. The best k was 3, which means only a few neighbours decide the class, so noisy points near the boundary can cause mistakes. |
| Naive Bayes | Naive Bayes got the same accuracy as KNN (93.86%), F1 of 0.9091 and MCC of 0.8715. Precision was 1.0 and AUC was high at 0.9934, but it missed 7 malignant cases (recall 0.8333). This model assumes all features are independent of each other, which is not true in this dataset because radius, perimeter and area are strongly related. Because of this its probabilities are too confident, and the default 0.5 cut-off misses some cancer cases. It is very fast and good as a baseline. |
| Support Vector Machine | SVM was the best model in my results. It had the highest accuracy (97.37%), F1 (0.9630) and MCC (0.9442). It did not give any false positive and missed only 3 malignant cases, with recall of 0.9286. AUC was 0.9927, almost same as logistic regression. The RBF kernel can learn a slightly curved boundary between the classes, and with C = 10 it fitted the data well without overfitting. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF)** gave the best overall performance because it was top in accuracy, F1 and MCC, had no false alarms and missed the least number of cancers. Logistic Regression was very close and had the best AUC, so it can be a good choice where we need a model that is easy to explain. In medical diagnosis missing a cancer is more serious than a false alarm, so the threshold could be lowered to improve recall further. |

## e. Streamlit app features

- **Dataset upload (CSV)** – upload test data in the sidebar; if nothing is uploaded the bundled `test_data.csv` is used, and it can be downloaded as a template.
- **Model selection dropdown** – choose any of the 5 saved models.
- **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- **Confusion matrix and classification report** for the selected model.
- **Compare all 5 models** – metrics table with best values highlighted, bar chart, ROC curves and the winner on the uploaded data.
- **Predictions** – row-level actual vs predicted, filter to misclassified rows, CSV download.
- Input validation (missing columns, non-numeric values, M/B or 1/0 labels). The app only **loads** the saved models – it never retrains on the uploaded data.

## f. How to run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python train_models.py            # re-train (add --offline to skip the internet)
streamlit run app.py              # opens http://localhost:8501
```
