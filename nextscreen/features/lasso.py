"""LASSO-based feature importance via cross-validated regularization."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def run_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit a LASSO model and return per-feature coefficient magnitudes.

    Features are standardized (zero mean, unit variance) before fitting
    so that coefficient magnitudes are comparable across features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    y : pd.Series
        Target variable with length n_samples.
    alpha : float or None, optional
        Regularization strength. If ``None`` (default), alpha is selected
        automatically via ``LassoCV`` with *cv* folds.
    cv : int, optional
        Number of cross-validation folds used when *alpha* is ``None``.
        Capped at ``len(X)`` automatically. Default is 5.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['feature', 'coefficient',
        'abs_coefficient', 'rank']``, sorted by descending
        ``abs_coefficient``. Rank 1 is most important.

    Notes
    -----
    Features with zero coefficients are included in the output with the
    highest (worst) rank value. Coefficients are in standardized units.

    Raises
    ------
    ValueError
        If *X* is empty, if *X* and *y* have different lengths, or if
        a manually supplied *alpha* is not positive.
    """
    _validate_inputs(X, y)
    if alpha is not None and alpha <= 0:
        raise ValueError(
            f"alpha must be a positive number; got {alpha}."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_arr = y.to_numpy(dtype=float)

    if alpha is None:
        actual_cv = min(cv, len(X))
        model: Lasso | LassoCV = LassoCV(
            cv=actual_cv, random_state=random_state
        )
        model.fit(X_scaled, y_arr)
        chosen = model.alpha_  # type: ignore[union-attr]
        logger.info("LassoCV selected alpha=%.6f.", chosen)
    else:
        model = Lasso(alpha=alpha, random_state=random_state)
        model.fit(X_scaled, y_arr)

    coefficients: np.ndarray = model.coef_

    result = pd.DataFrame(
        {
            "feature": list(X.columns),
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    result = result.sort_values(
        "abs_coefficient", ascending=False
    ).reset_index(drop=True)
    result["rank"] = (
        result["abs_coefficient"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    n_nonzero = int((result["abs_coefficient"] > 0).sum())
    logger.info(
        "LASSO: %d/%d features with non-zero coefficients.",
        n_nonzero,
        len(X.columns),
    )
    return result


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    """Raise ValueError for common input problems."""
    if X.empty:
        raise ValueError("X must contain at least one row.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length "
            f"({len(X)} vs {len(y)})."
        )
