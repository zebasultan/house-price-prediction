"""Regression metrics and diagnostic plots."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def plot_diagnostics(y_true, y_pred, out_path):
    """Actual-vs-predicted and residual plots."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_true, y_pred, s=10, alpha=.5)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set(xlabel="Actual price (lakhs)", ylabel="Predicted price (lakhs)",
              title="Actual vs Predicted")
    ax[1].hist(y_true - y_pred, bins=40, color="#4C78A8")
    ax[1].set(xlabel="Residual (lakhs)", ylabel="Count", title="Residual distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_feature_importance(pipeline, out_path, top=15):
    """Plot importances for tree models (no-op for others)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    names = pipeline.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    idx = np.argsort(model.feature_importances_)[-top:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[idx], model.feature_importances_[idx], color="#4C78A8")
    ax.set(title="Top feature importances", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
