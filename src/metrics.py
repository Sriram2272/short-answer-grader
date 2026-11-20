"""Agreement between predicted and human scores."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


def to_half_points(x, max_score: float = 5.0) -> np.ndarray:
    """Clip to [0, max] and round to the 0.5 steps the Mohler graders' average uses."""
    return np.rint(np.clip(np.asarray(x, dtype=float), 0, max_score) * 2) / 2


def report(y_true, y_pred, max_score: float = 5.0) -> dict:
    """Pearson r, RMSE, MAE and quadratic weighted kappa on half-point bands."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, max_score)
    r = pearsonr(y_true, y_pred)[0] if y_pred.std() > 0 and y_true.std() > 0 else float("nan")
    bands = list(range(int(max_score * 2) + 1))
    qwk = cohen_kappa_score((to_half_points(y_true, max_score) * 2).astype(int),
                            (to_half_points(y_pred, max_score) * 2).astype(int),
                            weights="quadratic", labels=bands)
    return {
        "pearson": round(float(r), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "qwk": round(float(qwk), 3),
    }
