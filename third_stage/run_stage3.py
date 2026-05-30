from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from flaml import AutoML
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from xgboost import DMatrix, XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from second_stage.metrics import compute_metrics, confusion_matrix_tuple

RANDOM_STATE = 42
TARGET = "loan_status"
OUTPUT_DIR = PROJECT_ROOT / "third_stage" / "outputs"
MODEL_DIR = PROJECT_ROOT / "third_stage" / "models"


BASELINE_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage 3 XGBoost optimization experiments.")
    parser.add_argument("--cv", type=int, default=5, help="Number of StratifiedKFold folds.")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--automl-time-budget", type=int, default=180, help="FLAML time budget in seconds.")
    parser.add_argument("--skip-grid", action="store_true", help="Skip Grid Search.")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Optuna.")
    parser.add_argument("--skip-automl", action="store_true", help="Skip FLAML AutoML.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller tuning budget for smoke tests.")
    return parser.parse_args()


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    split_dir = PROJECT_ROOT / "third_stage" / "splits"
    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")

    train_val = pd.concat([train, val], ignore_index=True)
    X_train_val = train_val.drop(columns=[TARGET])
    y_train_val = train_val[TARGET]
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]
    return X_train_val, y_train_val, X_test, y_test, train, val


def make_xgb(**params: Any) -> XGBClassifier:
    merged = BASELINE_PARAMS.copy()
    merged.update(params)
    return XGBClassifier(**merged)


def evaluate_model(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_count: int,
    train_time: float,
    params: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = compute_metrics(y_test, pred, positive_label=1)
    tn, fp, fn, tp = confusion_matrix_tuple(y_test, pred, positive_label=1)
    metrics.update(
        {
            "model": name,
            "threshold": threshold,
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "train_time_sec": train_time,
            "feature_count": feature_count,
            "best_params": json.dumps(params or {}, sort_keys=True),
        }
    )
    return metrics


def cv_f1(model: Any, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> tuple[float, float]:
    scores = cross_val_score(model, X, y, scoring="f1", cv=cv, n_jobs=1)
    return float(scores.mean()), float(scores.std())


def save_correlation_artifacts(X: pd.DataFrame) -> pd.DataFrame:
    corr = X.corr(method="pearson")
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(13, 10))
    sns.heatmap(corr, cmap="vlag", center=0, square=False, linewidths=0.1)
    plt.title("Pearson correlation matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()

    pairs = []
    columns = corr.columns.tolist()
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(corr.loc[left, right])
            if abs(value) >= 0.8:
                pairs.append({"feature_1": left, "feature_2": right, "correlation": value})
    pairs_df = pd.DataFrame(pairs).sort_values("correlation", key=lambda s: s.abs(), ascending=False)
    pairs_df.to_csv(OUTPUT_DIR / "high_correlation_pairs.csv", index=False)
    return pairs_df


def save_feature_importance(model: XGBClassifier, feature_names: list[str], prefix: str) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / f"{prefix}_feature_importance.csv", index=False)

    top = importance.head(15).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"], color="#2f6f9f")
    plt.title(f"Top feature importance: {prefix}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_feature_importance.png", dpi=180)
    plt.close()
    return importance


def correlation_pruned_features(high_corr_pairs: pd.DataFrame, importance: pd.DataFrame, all_features: list[str]) -> list[str]:
    if high_corr_pairs.empty:
        return all_features

    rank = dict(zip(importance["feature"], importance["importance"]))
    to_drop: set[str] = set()
    for row in high_corr_pairs.itertuples(index=False):
        left = row.feature_1
        right = row.feature_2
        if left in to_drop or right in to_drop:
            continue
        drop = left if rank.get(left, 0.0) < rank.get(right, 0.0) else right
        to_drop.add(drop)
    return [feature for feature in all_features if feature not in to_drop]


def run_select_k_best(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> tuple[dict[str, Any], list[str], pd.DataFrame]:
    candidate_k = [8, 10, 12, 14, 16, 18, 20, X.shape[1]]
    candidate_k = sorted({k for k in candidate_k if k <= X.shape[1]})
    rows = []
    for k in candidate_k:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        model = make_xgb()
        mean_f1, std_f1 = cv_f1(model, pd.DataFrame(X_selected, columns=selector.get_feature_names_out()), y, cv)
        rows.append({"k": k, "cv_f1_mean": mean_f1, "cv_f1_std": std_f1})

    results = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
    results.to_csv(OUTPUT_DIR / "select_k_best_cv_results.csv", index=False)
    best_k = int(results.iloc[0]["k"])

    selector = SelectKBest(score_func=f_classif, k=best_k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    score_table = pd.DataFrame(
        {
            "feature": X.columns,
            "score": selector.scores_,
            "p_value": selector.pvalues_,
            "selected": selector.get_support(),
        }
    ).sort_values("score", ascending=False)
    score_table.to_csv(OUTPUT_DIR / "select_k_best_feature_scores.csv", index=False)

    model = make_xgb()
    start = time.perf_counter()
    model.fit(X[selected_features], y)
    train_time = time.perf_counter() - start
    metrics = evaluate_model(
        "xgboost_select_k_best",
        model,
        X_test[selected_features],
        y_test,
        feature_count=len(selected_features),
        train_time=train_time,
        params={"k": best_k, "selected_features": selected_features, **BASELINE_PARAMS},
    )
    joblib.dump(model, MODEL_DIR / "xgboost_select_k_best.joblib")
    return metrics, selected_features, results


def fit_and_evaluate_xgb(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
) -> tuple[XGBClassifier, dict[str, Any]]:
    model = make_xgb(**params)
    start = time.perf_counter()
    model.fit(X, y)
    train_time = time.perf_counter() - start
    metrics = evaluate_model(name, model, X_test, y_test, X.shape[1], train_time, {**BASELINE_PARAMS, **params})
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    return model, metrics


def run_grid_search(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, cv: StratifiedKFold, quick: bool) -> dict[str, Any]:
    grid = {
        "n_estimators": [250, 400] if quick else [300, 500],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "scale_pos_weight": [1.0, float((y == 0).sum() / (y == 1).sum())],
    }
    estimator = make_xgb(n_jobs=1)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    start = time.perf_counter()
    search.fit(X, y)
    train_time = time.perf_counter() - start

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results.to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)
    joblib.dump(search.best_estimator_, MODEL_DIR / "xgboost_grid_search.joblib")
    return evaluate_model(
        "xgboost_grid_search",
        search.best_estimator_,
        X_test,
        y_test,
        X.shape[1],
        train_time,
        {**BASELINE_PARAMS, **search.best_params_},
    )


def run_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
    trials: int,
    quick: bool,
) -> dict[str, Any]:
    if quick:
        trials = min(trials, 10)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 4.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, float((y == 0).sum() / (y == 1).sum())),
            "n_jobs": 1,
        }
        model = make_xgb(**params)
        mean_f1, _ = cv_f1(model, X, y, cv)
        return mean_f1

    study = optuna.create_study(direction="maximize", study_name="stage3_xgboost_f1")
    start = time.perf_counter()
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    search_time = time.perf_counter() - start

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)
    with open(OUTPUT_DIR / "optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2, sort_keys=True)

    model = make_xgb(**study.best_params)
    start = time.perf_counter()
    model.fit(X, y)
    train_time = time.perf_counter() - start
    joblib.dump(model, MODEL_DIR / "xgboost_optuna.joblib")
    metrics = evaluate_model(
        "xgboost_optuna",
        model,
        X_test,
        y_test,
        X.shape[1],
        train_time + search_time,
        {**BASELINE_PARAMS, **study.best_params},
    )
    return metrics


def run_flaml_automl(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    time_budget: int,
    quick: bool,
) -> dict[str, Any]:
    if quick:
        time_budget = min(time_budget, 45)
    automl = AutoML()
    settings = {
        "time_budget": time_budget,
        "metric": "f1",
        "task": "classification",
        "log_file_name": str(OUTPUT_DIR / "flaml.log"),
        "seed": RANDOM_STATE,
        "verbose": 1,
        "n_jobs": -1,
        "estimator_list": ["xgboost", "lgbm", "rf", "extra_tree", "catboost", "lrl1"],
    }
    start = time.perf_counter()
    automl.fit(X_train=X, y_train=y, **settings)
    train_time = time.perf_counter() - start
    joblib.dump(automl, MODEL_DIR / "flaml_automl.joblib")

    pred = automl.predict(X_test)
    proba = automl.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, pred, positive_label=1)
    tn, fp, fn, tp = confusion_matrix_tuple(y_test, pred, positive_label=1)
    metrics.update(
        {
            "model": "flaml_automl",
            "threshold": np.nan,
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "train_time_sec": train_time,
            "feature_count": X.shape[1],
            "best_params": json.dumps(
                {
                    "best_estimator": automl.best_estimator,
                    "best_config": automl.best_config,
                    "time_budget": time_budget,
                },
                sort_keys=True,
                default=str,
            ),
        }
    )
    with open(OUTPUT_DIR / "flaml_best_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_estimator": automl.best_estimator, "best_config": automl.best_config},
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )
    return metrics


def save_local_explanations(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    case_masks = {
        "true_positive": (y_test.to_numpy() == 1) & (pred == 1),
        "true_negative": (y_test.to_numpy() == 0) & (pred == 0),
        "false_positive": (y_test.to_numpy() == 0) & (pred == 1),
        "false_negative": (y_test.to_numpy() == 1) & (pred == 0),
    }
    selected_indices: list[int] = []
    case_names: list[str] = []
    for case_name, mask in case_masks.items():
        matches = np.flatnonzero(mask)
        if len(matches) > 0:
            selected_indices.append(int(matches[0]))
            case_names.append(case_name)

    if not selected_indices:
        return

    booster = model.get_booster()
    dmatrix = DMatrix(X_test.iloc[selected_indices], feature_names=X_test.columns.tolist())
    contributions = booster.predict(dmatrix, pred_contribs=True)
    rows = []
    for row_id, case_name in enumerate(case_names):
        feature_contrib = pd.Series(contributions[row_id, :-1], index=X_test.columns)
        for feature, contribution in feature_contrib.reindex(feature_contrib.abs().sort_values(ascending=False).index).head(8).items():
            rows.append(
                {
                    "case": case_name,
                    "test_row_position": selected_indices[row_id],
                    "true_label": int(y_test.iloc[selected_indices[row_id]]),
                    "predicted_label": int(pred[selected_indices[row_id]]),
                    "predicted_probability": float(proba[selected_indices[row_id]]),
                    "feature": feature,
                    "feature_value": float(X_test.iloc[selected_indices[row_id]][feature]),
                    "contribution_log_odds": float(contribution),
                    "bias_log_odds": float(contributions[row_id, -1]),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"{model_name}_local_xgb_contributions.csv", index=False)


def write_report_notes(results: pd.DataFrame, high_corr_pairs: pd.DataFrame, select_k_results: pd.DataFrame) -> None:
    best = results.sort_values("f1", ascending=False).iloc[0]
    baseline = results[results["model"] == "xgboost_baseline"].iloc[0]
    select_best = select_k_results.sort_values("cv_f1_mean", ascending=False).iloc[0]
    lines = [
        "# Etap 3 - notatki do raportu",
        "",
        "## Punkt odniesienia",
        f"Baseline XGBoost z etapu 2 uzyskal na zbiorze testowym F1 = {baseline['f1']:.4f}, "
        f"recall = {baseline['recall']:.4f}, precision = {baseline['precision']:.4f}.",
        "",
        "## Optymalizacja cech",
        f"W macierzy korelacji znaleziono {len(high_corr_pairs)} par cech o |r| >= 0.8.",
        f"Najlepszy wariant SelectKBest w CV wybral k = {int(select_best['k'])} "
        f"z F1 CV = {select_best['cv_f1_mean']:.4f} +/- {select_best['cv_f1_std']:.4f}.",
        "",
        "## Najlepszy wariant",
        f"Najwyzsze F1 na zbiorze testowym ma `{best['model']}`: F1 = {best['f1']:.4f}, "
        f"recall = {best['recall']:.4f}, precision = {best['precision']:.4f}, "
        f"ROC-AUC = {best['roc_auc']:.4f}, PR-AUC = {best['pr_auc']:.4f}.",
        "",
        "## Artefakty",
        "- `stage3_results.csv` - tabela zbiorcza metryk.",
        "- `correlation_heatmap.png` i `correlation_matrix.csv` - analiza korelacji.",
        "- `select_k_best_cv_results.csv` - wyniki selekcji cech.",
        "- `*_feature_importance.csv/png` - waznosc cech.",
        "- `*_local_xgb_contributions.csv` - lokalne wyjasnienia predykcji XGBoost.",
    ]
    (OUTPUT_DIR / "report_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    X, y, X_test, y_test, train, val = load_data()
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    all_results: list[dict[str, Any]] = []

    split_summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train), "positive_share": train[TARGET].mean()},
            {"split": "val", "rows": len(val), "positive_share": val[TARGET].mean()},
            {"split": "train_val", "rows": len(X), "positive_share": y.mean()},
            {"split": "test", "rows": len(X_test), "positive_share": y_test.mean()},
        ]
    )
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    high_corr_pairs = save_correlation_artifacts(X)

    baseline_model, baseline_metrics = fit_and_evaluate_xgb("xgboost_baseline", X, y, X_test, y_test, {})
    all_results.append(baseline_metrics)
    baseline_importance = save_feature_importance(baseline_model, X.columns.tolist(), "xgboost_baseline")

    pruned_features = correlation_pruned_features(high_corr_pairs, baseline_importance, X.columns.tolist())
    _, pruned_metrics = fit_and_evaluate_xgb(
        "xgboost_correlation_pruned",
        X[pruned_features],
        y,
        X_test[pruned_features],
        y_test,
        {},
    )
    pruned_metrics["best_params"] = json.dumps({"features": pruned_features, **BASELINE_PARAMS}, sort_keys=True)
    all_results.append(pruned_metrics)

    select_metrics, selected_features, select_k_results = run_select_k_best(X, y, X_test, y_test, cv)
    all_results.append(select_metrics)

    top_k = len(selected_features)
    top_features = baseline_importance.head(top_k)["feature"].tolist()
    _, top_importance_metrics = fit_and_evaluate_xgb(
        "xgboost_top_importance_features",
        X[top_features],
        y,
        X_test[top_features],
        y_test,
        {},
    )
    top_importance_metrics["best_params"] = json.dumps({"features": top_features, **BASELINE_PARAMS}, sort_keys=True)
    all_results.append(top_importance_metrics)

    if not args.skip_grid:
        all_results.append(run_grid_search(X, y, X_test, y_test, cv, args.quick))
    if not args.skip_optuna:
        all_results.append(run_optuna(X, y, X_test, y_test, cv, args.optuna_trials, args.quick))
    if not args.skip_automl:
        all_results.append(run_flaml_automl(X, y, X_test, y_test, args.automl_time_budget, args.quick))

    results = pd.DataFrame(all_results)
    metric_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "threshold",
        "train_time_sec",
        "feature_count",
        "best_params",
    ]
    results = results[metric_columns].sort_values("f1", ascending=False)
    results.to_csv(OUTPUT_DIR / "stage3_results.csv", index=False)

    best_xgb_name = results[results["model"].str.startswith("xgboost")].iloc[0]["model"]
    best_xgb = joblib.load(MODEL_DIR / f"{best_xgb_name}.joblib")
    best_feature_json = json.loads(results[results["model"] == best_xgb_name].iloc[0]["best_params"])
    best_features = best_feature_json.get("features") or best_feature_json.get("selected_features") or X.columns.tolist()
    save_feature_importance(best_xgb, best_features, best_xgb_name)
    save_local_explanations(best_xgb, X_test[best_features], y_test, best_xgb_name)
    write_report_notes(results, high_corr_pairs, select_k_results)

    print(results[["model", "f1", "precision", "recall", "roc_auc", "pr_auc", "feature_count"]].round(4).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
