# 2026AK05500- Assignment 2 ML

**M.Tech (AIML) – Machine Learning, Assignment 2 - Ankit Girdhar (2026AK05500)**

- **Live Streamlit app:** https://2026ak05500-app.streamlit.app/
- **GitHub repository:** https://github.com/girdharankit/2026AK05500---Assignment2--ML

---

## a. Problem statement

In this assignment I have built a model to predict whether a breast tumour is **Malignant** (cancer) or **Benign** (not cancer). The prediction is made from 30 measurements of cell nuclei, which are taken from a digitised image of a fine needle aspirate (FNA) of the breast mass. Since there are only two possible outputs, this is a **binary classification** problem.

I trained five different classification models on the same data, checked each of them using six evaluation metrics (Accuracy, AUC, Precision, Recall, F1 and MCC) and compared the results. After that I made a Streamlit web app where test data can be uploaded and the saved models are used to give the results. The app only loads the trained models, it does not train them again.

For this problem I gave more importance to **Recall** for the malignant class and to **MCC**. The reason is that if a cancer case is predicted as benign (false negative), it is much more serious than a false alarm.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) (WDBC) |
| Source | UCI Machine Learning Repository, dataset id 17: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Number of instances | **569** (assignment needs at least 500) |
| Number of features | **30** numeric features (assignment needs at least 12). The ID column is not used. |
| Target column | `diagnosis`: **M = Malignant (1)**, **B = Benign (0)** |
| Class distribution | 357 Benign (62.7%) and 212 Malignant (37.3%) |
| Missing values | None |

**About the features.** For every cell nucleus in the image, 10 properties are measured: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. For each image the dataset gives the **mean**, the **standard error** (`_se`) and the **worst** value (average of the three largest values) of these 10 properties. So in total there are 10 x 3 = 30 features, for example `radius_mean`, `radius_se` and `radius_worst`.

**How the data is loaded** (code in `data_utils.py`). The script first tries to download the data online and if that is not possible it uses a local copy:
1. From the UCI repository using the `ucimlrepo` package (`fetch_ucirepo(id=17)`)
2. From the raw `wdbc.data` file on the UCI website
3. From the local file `data/wdbc.csv` (this file gets updated every time the online download works)
4. From the copy of the same dataset that comes with scikit-learn (`load_breast_cancer`)

**Pre-processing.** I split the data into 80% training and 20% testing using a stratified split (`random_state = 42`), so the class ratio stays the same in both parts. This gave 455 rows for training and 114 rows for testing. The test rows are saved as `test_data.csv`. For Logistic Regression, KNN and SVM I used `StandardScaler` inside the pipeline, because these models depend on the scale of the features. Hyper-parameters for every model were selected with 5-fold stratified `GridSearchCV` (scoring = F1) using **only the training data**.

## c. GitHub repository link

https://github.com/girdharankit/2026AK05500---Assignment2--ML

```
bc-classifier-app/
|-- app.py                      # Streamlit app
|-- train_models.py             # trains, tunes, evaluates and saves the 5 models
|-- data_utils.py               # loads the dataset (UCI online)
|-- ML_Assignment2_Colab.ipynb  
|-- requirements.txt
|-- README.md
|-- test_data.csv               # 20% test data (30 features + diagnosis)
|-- data/wdbc.csv               # local copy of the full dataset
|-- model/                      # saved models
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   |-- svm.pkl
|   `-- metadata.json           # feature names, best hyper-parameters, test scores
|-- results/                    # metrics table, classification reports, training log
`-- .streamlit/config.toml      
```

## d. Models used

All five models were trained on the same training data and tested on the same 114 test rows. Malignant is taken as the positive class.

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

Confusion matrix values on the test set (72 Benign, 42 Malignant):

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

- **Upload test data (CSV):** test data can be uploaded from the sidebar. If no file is uploaded, the app uses the `test_data.csv` from the repo, and this file can also be downloaded from the sidebar to use as a sample.
- **Model selection dropdown:** any of the 5 saved models can be selected.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC are shown for the selected model.
- **Confusion matrix and classification report** for the selected model.
- **Compare all 5 models:** one table with all metrics (best values highlighted), a bar chart, ROC curves and the best model on the uploaded data.
- **Predictions:** actual vs predicted for every row, option to show only wrong predictions, and CSV download.
- The app checks the uploaded file (missing columns, non-numeric values, labels as M/B or 1/0). It only **loads** the saved models and never trains on the uploaded data.

## f. How to run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python train_models.py            # train again (add --offline to skip the internet)
streamlit run app.py              # opens http://localhost:8501
```
