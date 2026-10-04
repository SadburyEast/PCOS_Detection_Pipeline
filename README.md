# PCOS Prediction Pipeline

A modular, leakage-safe machine learning pipeline that predicts **Polycystic Ovary Syndrome (PCOS)** from clinical, hormonal and ultrasound data. One command cleans the data, tunes four models, picks the best one, and shows how the models compare.

Built for a biohackathon. Designed to be reused on any tabular classification problem by changing a few constants.

```bash
python main.py
```

<!-- Add a screenshot after your first run, e.g.:
![Model comparison](outputs/plots/metric_comparison.png)
-->

## What it does

```mermaid
flowchart LR
    A[Excel / CSV] --> B[Extract]
    B --> C[Clean]
    C --> D[Stratified split]
    D --> E["Impute → Scale / One-hot → SMOTE → Model<br/>(inside each CV fold)"]
    E --> F[Grid search x 4 models]
    F --> G[Evaluate on held-out test set]
    G --> H[Best model + params + pop-up plots]
```

- **Tunes and compares** Logistic Regression, Random Forest, Gradient Boosting and XGBoost with 5-fold stratified cross-validated grid search.
- **Reports** the best model, its best hyperparameters, and ROC-AUC / accuracy / precision / recall / F1 for every model.
- **Compares feature importance** across models, with one-hot columns folded back into their original feature.
- **Opens pop-up plots** for metric comparison, ROC curves, feature importance, and the best model's confusion matrix. They are also saved as PNGs.
- **Saves** the best model (`.joblib`), a `results.json` summary, and a feature importance CSV.

## Engineering decisions worth noting

| Problem | What I did |
|---|---|
| **Data leakage from resampling.** Applying SMOTE before cross-validation lets synthetic copies of validation rows leak into training and inflates scores. | SMOTE, imputation, scaling and encoding all live inside an `imblearn` pipeline, so they are re-fit on the training part of every fold only. |
| **An ID column acting as a feature.** The row index (`Sl. No`) ranked among the top Random Forest features in my first notebook, a sign row order was leaking the label. | Identifier columns are dropped up front. |
| **Tiny test set (~108 rows).** Picking the winner on it is partly luck. | The best model is chosen by cross-validated ROC-AUC by default (`--select-by test` to override), and both scores are reported. |
| **Messy real-world data.** Column names with stray whitespace, text in numeric columns, blood pressure recorded in tens, extreme lab outliers. | Handled in one stateless, testable `Preprocessor` class, driven by constants in `util.py`. |

## Project structure

```
├── main.py                 # CLI entry point
├── pipeline.py             # Orchestrates the full run
├── util.py                 # Logger + dataset constants (columns, caps, rules)
├── processer/
│   ├── extract.py          # Load .xlsx / .csv
│   └── preprocess.py       # Stateless cleaning
├── model/
│   ├── models.py           # Model classes + grid search (sklearn / imblearn pipelines)
│   ├── evaluate.py         # Metrics, best-model selection, report
│   └── visualize.py        # Pop-up and saved plots
└── requirements.txt
```

Adding a model means adding one small class to `model/models.py` and one line to the registry.

## Quick start

```bash
pip install -r requirements.txt

# Put the dataset at datasets/(Main_Dataset)_PCOS_data_without_infertility.xlsx, or:
python main.py --data-path path/to/data.xlsx --sheet-name Full_new
```

| Flag | Purpose |
|---|---|
| `--models random_forest xgboost` | Choose which models to tune |
| `--select-by {cv,test}` | How the best model is picked (default `cv`) |
| `--no-smote` | Use class weights instead of SMOTE |
| `--no-show` | Save plots without opening windows |
| `--cv-folds`, `--test-size`, `--top-n` | Tweak validation and plot settings |

## Results

<!-- Replace with your real numbers after running `python main.py` -->

| Model | CV ROC-AUC | Test ROC-AUC | Test Accuracy |
|---|---|---|---|
| Random Forest | 0.958 | 0.939 | 0.908 |
| Logistic Regression | 0.949 | 0.952 | 0.909 |
| Gradient Boosting | 0.946 | 0.954 | 0.917 |
| XGBoost | 0.944 | 0.944 | 0.908 |

**Best model:** Random Forest · **Best parameters:** "max_depth": 10, "min_samples_leaf": 1, "n_estimators": 300 

**Top predictive features:** Follicle Numbers, Hair Growth and Cycle (Regular/Irregular)

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · matplotlib

## Limitations and next steps

- The dataset is small (hundreds of patients), so test-set metrics have wide error bars. Treat them as indicative, not definitive.
- Not externally validated. This is a research and learning project, **not a medical device** or diagnostic tool.
- Next: permutation importance and SHAP for explanations, probability calibration, and validation on an independent cohort.
