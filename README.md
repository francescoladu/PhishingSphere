# PhishingSphere

Machine learning pipeline for **phishing website detection**, developed for the *Data Mining & Machine Learning* course of the M.Sc. in Artificial Intelligence and Data Engineering at the University of Pisa.

The project studies how repeated and conflicting feature profiles affect model evaluation on the **UCI Phishing Websites** dataset, comparing Decision Tree and Random Forest models under three different data representations.

## Project Overview

The task is a binary classification problem:

- `-1` — phishing
- `1` — legitimate

The dataset contains **11,055 observations** and **30 pre-computed discrete predictors** describing:

- URL and address-bar characteristics;
- abnormal links and resource behavior;
- HTML / JavaScript behavior;
- domain and reputation information.

The project focuses not only on predictive performance, but also on **leakage-aware model selection, duplicate handling, explainability, and error analysis**.

## Experimental Setup

Three representations of the dataset are evaluated:

| Experiment | Representation | Description |
|---|---|---|
| **Exp. 1** | Raw data | Keeps all original observations |
| **Exp. 2** | Exact deduplication | Removes repeated rows sharing predictors and target |
| **Exp. 3** | Weighted profile deduplication | Groups identical predictor profiles, keeps the majority label and preserves its support through `sample_weight` |

Experiment 3 is used for the detailed final analysis because it removes repeated profiles while preserving information about their empirical frequency.

## Methodology

The modeling pipeline includes:

- **Mutual Information** feature selection with `SelectKBest`;
- **Decision Tree** as a baseline;
- **Random Forest** as the ensemble model;
- **Nested Stratified Cross-Validation**
  - 10 outer folds for performance estimation;
  - 5 inner folds for feature selection and hyperparameter optimization;
- **Grid Search** for Decision Tree;
- **Randomized Search** for Random Forest;
- **1-standard-error rule** for model simplification among competitive configurations;
- **Wilcoxon signed-rank test** for paired DT/RF comparisons;
- held-out test evaluation;
- **Permutation Importance** for global explainability;
- **TreeSHAP** for local explanations;
- feature-level error analysis using out-of-fold predictions.

All data-dependent feature-selection and model-selection operations are fitted inside the appropriate training folds to avoid validation leakage.

## Main Results

Random Forest achieved higher nested-CV accuracy than Decision Tree in all three experiments.

| Experiment | RF nested-CV accuracy | RF test accuracy |
|---|---:|---:|
| Exp. 1 — Raw | 0.9699 | 0.9760 |
| Exp. 2 — Exact deduplication | 0.9455 | 0.9487 |
| Exp. 3 — Weighted deduplication | 0.9578 | 0.9612 |

For **Experiment 3**, the final Random Forest obtained:

- **Weighted accuracy:** 0.9612
- **Macro F1-score:** 0.9606
- **Phishing precision:** 0.9581
- **Phishing recall:** 0.9538
- **ROC-AUC:** 0.9937

The final configuration retained all 30 predictors.

Global explainability identified `URL_of_Anchor` and `SSLfinal_State` as the two features with the largest effect on validation performance. Local SHAP analysis showed how different values of the same features can push individual predictions toward phishing or legitimate.

## Repository Structure

```text
.
├── Makefile
├── code
│   ├── data_preprocessing
│   │   ├── config.py
│   │   ├── data_processing.py
│   │   ├── main.py
│   │   ├── plots.py
│   │   └── outputs/
│   │
│   ├── model_selection
│   │   ├── config.py
│   │   ├── diagnostics.py
│   │   ├── engine.py
│   │   ├── main.py
│   │   ├── paths.py
│   │   ├── plots.py
│   │   ├── summaries.py
│   │   ├── utils.py
│   │   └── outputs/
│   │
│   ├── final_evaluation
│   │   ├── config.py
│   │   ├── main.py
│   │   └── outputs/
│   │
│   ├── explainability
│   │   ├── config.py
│   │   ├── generate_false_negative_shap.py
│   │   ├── main.py
│   │   ├── permutation_importance.py
│   │   ├── plots.py
│   │   ├── shap_force.py
│   │   └── outputs/
│   │
│   └── shared
│       ├── config.py
│       └── modeling.py
│
└── scripts
    └── clean_and_split.py
```

### Main modules

`data_preprocessing/` contains the exploratory data analysis, Mutual Information ranking, feature distributions, conflicting-profile statistics, and Spearman correlation analysis.

`model_selection/` contains nested cross-validation, feature selection, hyperparameter optimization, statistical comparison, permutation importance, and out-of-fold diagnostics.

`final_evaluation/` refits the selected configurations on the full development data and evaluates them on the corresponding held-out test set.

`explainability/` generates the global permutation-importance plots and local SHAP explanations.

`shared/` contains configuration and modeling utilities shared across the pipeline.

## Running the Project

The repository includes a `Makefile` that exposes the complete workflow for each experiment.

### 1. Create the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install the Python dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn shap
```

The Makefile also requires `make` and `curl`.

### 3. Explore the available commands

```bash
make help
```

### 4. Run a complete experiment

```bash
make experiment1
make experiment2
make experiment3
```

Each command performs dataset preparation, EDA, model selection, held-out evaluation, and explainability for the corresponding representation.

Individual stages can also be executed separately:

```bash
make install3
make analyze3
make train3
make evaluate3
make explain3
```

Replace `3` with `1` or `2` for the other experiments.

## Reproducibility

The project uses a shared random seed:

```text
RANDOM_STATE = 42
```

The Makefile automatically downloads the UCI Phishing Websites dataset and generates the experiment-specific development and test files.

Generated artifacts are saved under the corresponding `outputs/` directories and include:

- EDA statistics and plots;
- feature-selection rankings;
- hyperparameter-search results;
- fold-level model scores;
- Wilcoxon test results;
- out-of-fold predictions;
- test predictions and metrics;
- confusion matrices and ROC curves;
- permutation importance;
- SHAP explanations.

## Dataset

R. Mohammad and L. McCluskey, **Phishing Websites**, UCI Machine Learning Repository, 2012.  
DOI: `10.24432/C51W2X`
