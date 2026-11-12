"""Train, compare, tune and save the best house-price model.

Run:  python -m src.train
"""
import json
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config as C
from src.evaluate import plot_diagnostics, plot_feature_importance, regression_metrics
from src.preprocess import build_preprocessor, clean_data, load_data


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def candidate_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                              random_state=C.RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=C.RANDOM_STATE),
    }


def main():
    raw = load_data(C.DATA_PATH)
    df = clean_data(raw)
    print(f"Rows: raw={len(raw)}  after cleaning={len(df)}")

    X, y = df[C.FEATURES], df[C.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)

    # 1) compare models with 5-fold CV on the training set
    cv_results = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(make_pipeline(model), X_tr, y_tr, cv=5,
                                 scoring="neg_root_mean_squared_error")
        cv_results[name] = {"cv_rmse_mean": float(-scores.mean()),
                            "cv_rmse_std": float(scores.std())}
        print(f"{name:18s} CV RMSE = {-scores.mean():8.2f} ± {scores.std():.2f}")

    # 2) tune the best candidate
    best_name = min(cv_results, key=lambda k: cv_results[k]["cv_rmse_mean"])
    print(f"Best by CV: {best_name}")
    grids = {
        "GradientBoosting": {"model__n_estimators": [200, 400],
                             "model__learning_rate": [0.05, 0.1],
                             "model__max_depth": [2, 3, 4]},
        "RandomForest": {"model__n_estimators": [200, 400],
                         "model__max_depth": [None, 12, 20]},
        "Ridge": {"model__alpha": [0.1, 1, 10]},
        "LinearRegression": {},
    }
    search = GridSearchCV(make_pipeline(candidate_models()[best_name]),
                          grids[best_name], cv=3,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    print("Best params:", search.best_params_)

    # 3) final evaluation on the untouched hold-out test set
    pred = best.predict(X_te)
    test_metrics = regression_metrics(y_te, pred)
    print("Test metrics:", {k: round(v, 3) for k, v in test_metrics.items()})

    # 4) persist artefacts
    C.MODEL_PATH.parent.mkdir(exist_ok=True)
    C.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best, C.MODEL_PATH)
    plot_diagnostics(y_te, pred, C.REPORTS_DIR / "actual_vs_predicted.png")
    plot_feature_importance(best, C.REPORTS_DIR / "feature_importance.png")
    with open(C.METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "best_params": search.best_params_,
                   "cv_results": cv_results, "test_metrics": test_metrics,
                   "n_train": len(X_tr), "n_test": len(X_te)}, f, indent=2)
    print(f"Saved model -> {C.MODEL_PATH}")


if __name__ == "__main__":
    main()
