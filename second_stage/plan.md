# Project Plan: Loan Status Classification

## Scope and Requirements (from screenshot)
- Prepare data for training (aggregation if needed).
- Train 3-4 models with scikit-learn.
- Train 2-3 models from other libraries.
- Manual train/test split (optional validation).
- Pick metrics appropriate for classification.
- Manual implementation of the metrics formulas.
- Manual cross-validation with stratification.
- Compare results with tables, plots, confusion matrices, etc.

## Dataset Choice
- Primary dataset: data/splited_qualitive/loan_data_automl_standard_splitted.csv
- Target: loan_status
- Check class balance and missing values.

## Data Preparation Steps
1. Load dataset and verify target column.
2. Check class distribution (imbalance).
3. Split into train/val/test with stratification (e.g., 70/15/15 or 80/20).
4. If scaling is needed, fit scaler on train only, transform val/test.

## Models to Train
### scikit-learn (3-4 models)
- LogisticRegression (baseline, interpretable)
- RandomForestClassifier
- GradientBoostingClassifier or HistGradientBoostingClassifier
- SVC (RBF or linear) or KNeighborsClassifier

### Other libraries (2-3 models)
- XGBoost (xgboost.XGBClassifier)
- LightGBM (lightgbm.LGBMClassifier)
- CatBoost (catboost.CatBoostClassifier)

## Metrics (manual formulas)
- Confusion matrix (TP, FP, TN, FN)
- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- Optional: ROC AUC and PR AUC (plots)

## Cross-Validation (manual stratified)
- StratifiedKFold
- Manual loop over folds
- Compute metrics per fold
- Report mean +/- std per metric

## Final Evaluation
- Pick best model based on CV
- Retrain on full training data
- Evaluate once on test set

## Report Outputs
### Tables
- Model comparison on validation/test (Accuracy, Precision, Recall, F1, Specificity)
- Cross-validation summary (mean +/- std)

### Plots
- Confusion matrix for top models
- ROC and PR curves with AUC
- Bar chart comparing F1 or recall across models
- Feature importance (tree-based) or coefficients (logistic regression)
