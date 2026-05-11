"""Tests for nextscreen.features.* individual method modules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nextscreen.features.correlations import run_correlations
from nextscreen.features.lasso import run_lasso
from nextscreen.features.pca import run_pca
from nextscreen.features.random_forest import run_random_forest
from nextscreen.features.shap_analysis import run_shap

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FEAT_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d"]


@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic (X, y) where feat_a is the dominant driver of y.

    y = 2.0 * feat_a + 0.3 * feat_b + N(0, 0.1)
    feat_c and feat_d are pure noise uncorrelated with y.
    """
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "feat_c": rng.normal(0, 1, n),
            "feat_d": rng.normal(0, 1, n),
        }
    )
    y = pd.Series(
        2.0 * X["feat_a"]
        + 0.3 * X["feat_b"]
        + rng.normal(0, 0.1, n),
        name="yield",
    )
    return X, y


@pytest.fixture()
def dominant_pc_X() -> pd.DataFrame:
    """X where the first PC explains >99% of variance.

    All three features are near-perfect copies of a single latent
    variable, so PCA should select n_components=1 with any threshold
    below ~1.0.
    """
    rng = np.random.default_rng(5)
    n = 100
    z = rng.normal(0, 100, n)
    eps = rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "f1": z + eps,
            "f2": z + rng.normal(0, 0.01, n),
            "f3": z + rng.normal(0, 0.01, n),
        }
    )


# ---------------------------------------------------------------------------
# LASSO
# ---------------------------------------------------------------------------


class TestRunLasso:
    def test_returns_dataframe(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_present(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        assert set(result.columns) == {
            "feature", "coefficient", "abs_coefficient", "rank"
        }

    def test_all_features_present(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        assert set(result["feature"]) == set(_FEAT_NAMES)

    def test_rank_column_is_integer(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        assert pd.api.types.is_integer_dtype(result["rank"])

    def test_rank_starts_at_one(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        assert result["rank"].min() == 1

    def test_sorted_by_abs_coefficient_desc(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        vals = result["abs_coefficient"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_abs_coefficient_nonnegative(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        assert (result["abs_coefficient"] >= 0).all()

    def test_feat_a_is_rank_one(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        top = result[result["rank"] == 1].iloc[0]
        assert top["feature"] == "feat_a"

    def test_custom_alpha_accepted(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y, alpha=0.01)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(_FEAT_NAMES)

    def test_no_duplicate_features(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        assert result["feature"].nunique() == len(_FEAT_NAMES)

    def test_index_is_reset(self, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        result = run_lasso(X, y)
        assert list(result.index) == list(range(len(result)))

    def test_zero_alpha_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        with pytest.raises(ValueError, match="positive"):
            run_lasso(X, y, alpha=0.0)

    def test_negative_alpha_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        with pytest.raises(ValueError, match="positive"):
            run_lasso(X, y, alpha=-0.1)

    def test_length_mismatch_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        with pytest.raises(ValueError, match="same length"):
            run_lasso(X, y.iloc[:-1])

    def test_empty_X_raises(self, synthetic_data: tuple) -> None:
        _, y = synthetic_data
        with pytest.raises(ValueError, match="at least one row"):
            run_lasso(pd.DataFrame(columns=_FEAT_NAMES), y)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class TestRunRandomForest:
    def test_returns_dataframe(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        assert isinstance(run_random_forest(X, y), pd.DataFrame)

    def test_output_columns_present(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y)
        assert set(result.columns) == {"feature", "importance", "rank"}

    def test_all_features_present(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y)
        assert set(result["feature"]) == set(_FEAT_NAMES)

    def test_importances_sum_to_one(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y)
        assert math.isclose(
            result["importance"].sum(), 1.0, rel_tol=1e-6
        )

    def test_importances_nonnegative(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y)
        assert (result["importance"] >= 0).all()

    def test_sorted_by_importance_desc(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y)
        vals = result["importance"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_rank_column_is_integer(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y)
        assert pd.api.types.is_integer_dtype(result["rank"])

    def test_rank_starts_at_one(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y)
        assert result["rank"].min() == 1

    def test_top_feature_is_feat_a(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y)
        assert result.iloc[0]["feature"] == "feat_a"

    def test_custom_n_estimators(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y, n_estimators=10)
        assert isinstance(result, pd.DataFrame)

    def test_custom_max_depth(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y, max_depth=3)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_reset(self, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        result = run_random_forest(X, y)
        assert list(result.index) == list(range(len(result)))

    def test_invalid_n_estimators_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        with pytest.raises(ValueError, match="n_estimators"):
            run_random_forest(X, y, n_estimators=0)

    def test_length_mismatch_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        with pytest.raises(ValueError, match="same length"):
            run_random_forest(X, y.iloc[:-1])

    def test_empty_X_raises(self, synthetic_data: tuple) -> None:
        _, y = synthetic_data
        with pytest.raises(ValueError, match="at least one row"):
            run_random_forest(
                pd.DataFrame(columns=_FEAT_NAMES), y
            )


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


class TestRunPca:
    def test_returns_dict(self, synthetic_data: tuple) -> None:
        X, _ = synthetic_data
        assert isinstance(run_pca(X), dict)

    def test_required_keys_present(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        result = run_pca(X)
        expected = {
            "explained_variance_ratio",
            "cumulative_variance",
            "n_components",
            "loadings",
            "feature_rank",
        }
        assert expected.issubset(result.keys())

    def test_loadings_is_dataframe(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        result = run_pca(X)
        assert isinstance(result["loadings"], pd.DataFrame)

    def test_loadings_shape(self, synthetic_data: tuple) -> None:
        X, _ = synthetic_data
        result = run_pca(X)
        n_comp = result["n_components"]
        assert result["loadings"].shape == (len(_FEAT_NAMES), n_comp)

    def test_loadings_index_is_feature_names(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        result = run_pca(X)
        assert list(result["loadings"].index) == _FEAT_NAMES

    def test_loadings_column_names(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        result = run_pca(X)
        n_comp = result["n_components"]
        expected = [f"PC{i + 1}" for i in range(n_comp)]
        assert list(result["loadings"].columns) == expected

    def test_n_components_positive(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        assert run_pca(X)["n_components"] >= 1

    def test_max_components_respected(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        result = run_pca(X, max_components=2)
        assert result["n_components"] <= 2

    def test_cumulative_variance_is_nondecreasing(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        cum = run_pca(X)["cumulative_variance"]
        diffs = np.diff(cum)
        assert (diffs >= -1e-10).all()

    def test_evr_sums_leq_one(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        evr = run_pca(X)["explained_variance_ratio"]
        assert evr.sum() <= 1.0 + 1e-10

    def test_threshold_met_by_dominant_pc(
        self, dominant_pc_X: pd.DataFrame
    ) -> None:
        """With near-perfectly correlated features, 1 PC should suffice."""
        result = run_pca(dominant_pc_X, variance_threshold=0.90)
        assert result["n_components"] == 1

    def test_low_threshold_fewer_components(
        self, synthetic_data: tuple
    ) -> None:
        """A low threshold should yield fewer components than a high one."""
        X, _ = synthetic_data
        n_low = run_pca(X, variance_threshold=0.30)["n_components"]
        n_high = run_pca(X, variance_threshold=0.90)["n_components"]
        assert n_low <= n_high

    def test_feature_rank_columns(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        fr = run_pca(X)["feature_rank"]
        assert set(fr.columns) == {"feature", "max_loading", "rank"}

    def test_feature_rank_all_features(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        fr = run_pca(X)["feature_rank"]
        assert set(fr["feature"]) == set(_FEAT_NAMES)

    def test_feature_rank_starts_at_one(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        fr = run_pca(X)["feature_rank"]
        assert fr["rank"].min() == 1

    def test_invalid_threshold_zero_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        with pytest.raises(ValueError, match="variance_threshold"):
            run_pca(X, variance_threshold=0.0)

    def test_invalid_threshold_gt_one_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        with pytest.raises(ValueError, match="variance_threshold"):
            run_pca(X, variance_threshold=1.1)

    def test_invalid_max_components_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, _ = synthetic_data
        with pytest.raises(ValueError, match="max_components"):
            run_pca(X, max_components=0)

    def test_empty_X_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one row"):
            run_pca(pd.DataFrame(columns=["a", "b"]))


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


class TestRunShap:
    def test_returns_dict(self, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        assert isinstance(run_shap(X, y), dict)

    def test_required_keys_present(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_shap(X, y)
        assert {"shap_values", "feature_importance", "X_background"
                }.issubset(result.keys())

    def test_shap_values_shape(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_shap(X, y)
        assert result["shap_values"].shape == (len(X), len(_FEAT_NAMES))

    def test_feature_importance_columns(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        fi = run_shap(X, y)["feature_importance"]
        assert set(fi.columns) == {
            "feature", "mean_abs_shap", "rank"
        }

    def test_all_features_in_importance(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        fi = run_shap(X, y)["feature_importance"]
        assert set(fi["feature"]) == set(_FEAT_NAMES)

    def test_rank_starts_at_one(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        fi = run_shap(X, y)["feature_importance"]
        assert fi["rank"].min() == 1

    def test_rank_is_integer(self, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        fi = run_shap(X, y)["feature_importance"]
        assert pd.api.types.is_integer_dtype(fi["rank"])

    def test_mean_abs_shap_nonnegative(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        fi = run_shap(X, y)["feature_importance"]
        assert (fi["mean_abs_shap"] >= 0).all()

    def test_sorted_by_mean_abs_shap_desc(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        fi = run_shap(X, y)["feature_importance"]
        vals = fi["mean_abs_shap"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_feat_a_top_ranked(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        fi = run_shap(X, y)["feature_importance"]
        assert fi.iloc[0]["feature"] == "feat_a"

    def test_background_samples_capped(
        self, synthetic_data: tuple
    ) -> None:
        """background_samples > n should be capped at n."""
        X, y = synthetic_data
        result = run_shap(X, y, background_samples=9999)
        assert len(result["X_background"]) == len(X)

    def test_small_background_samples(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_shap(X, y, background_samples=5)
        assert len(result["X_background"]) == 5

    def test_X_background_is_dataframe(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_shap(X, y)
        assert isinstance(result["X_background"], pd.DataFrame)

    def test_X_background_columns_match(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_shap(X, y)
        assert list(result["X_background"].columns) == _FEAT_NAMES

    def test_length_mismatch_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        with pytest.raises(ValueError, match="same length"):
            run_shap(X, y.iloc[:-1])

    def test_empty_X_raises(self, synthetic_data: tuple) -> None:
        _, y = synthetic_data
        with pytest.raises(ValueError, match="at least one row"):
            run_shap(pd.DataFrame(columns=_FEAT_NAMES), y)


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------


class TestRunCorrelations:
    def test_returns_dataframe(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        assert isinstance(run_correlations(X, y), pd.DataFrame)

    def test_index_contains_all_features(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y)
        assert set(result.index) == set(_FEAT_NAMES)

    def test_both_pearson_columns_present(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y, method="both")
        for col in ("pearson_r", "pearson_p", "pearson_significant"):
            assert col in result.columns

    def test_both_spearman_columns_present(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y, method="both")
        for col in (
            "spearman_r", "spearman_p", "spearman_significant"
        ):
            assert col in result.columns

    def test_pearson_only_has_no_spearman_cols(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y, method="pearson")
        assert "spearman_r" not in result.columns
        assert "spearman_p" not in result.columns

    def test_spearman_only_has_no_pearson_cols(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y, method="spearman")
        assert "pearson_r" not in result.columns
        assert "pearson_p" not in result.columns

    def test_rank_column_present(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        assert "rank" in run_correlations(X, y).columns

    def test_rank_starts_at_one(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        assert run_correlations(X, y)["rank"].min() == 1

    def test_rank_is_integer(self, synthetic_data: tuple) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y)
        assert pd.api.types.is_integer_dtype(result["rank"])

    def test_feat_a_is_rank_one(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y)
        assert result.loc["feat_a", "rank"] == 1

    def test_feat_a_significant(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y)
        assert bool(result.loc["feat_a", "pearson_significant"])

    def test_strong_pearson_correlation_magnitude(
        self, synthetic_data: tuple
    ) -> None:
        """feat_a should have |r| > 0.95 given the strong signal."""
        X, y = synthetic_data
        result = run_correlations(X, y, method="pearson")
        assert abs(result.loc["feat_a", "pearson_r"]) > 0.95

    def test_pearson_r_in_minus_one_to_one(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y, method="pearson")
        assert (result["pearson_r"].abs() <= 1.0 + 1e-10).all()

    def test_spearman_r_in_minus_one_to_one(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y, method="spearman")
        assert (result["spearman_r"].abs() <= 1.0 + 1e-10).all()

    def test_significance_flag_dtype_is_bool(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        result = run_correlations(X, y)
        assert result["pearson_significant"].dtype == bool

    def test_custom_significance_threshold(
        self, synthetic_data: tuple
    ) -> None:
        """With threshold=0.0, nothing should be significant."""
        X, y = synthetic_data
        result = run_correlations(X, y, significance_threshold=0.0)
        assert not result["pearson_significant"].any()

    def test_invalid_method_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        with pytest.raises(ValueError, match="Unknown method"):
            run_correlations(X, y, method="kendall")  # type: ignore[arg-type]

    def test_length_mismatch_raises(
        self, synthetic_data: tuple
    ) -> None:
        X, y = synthetic_data
        with pytest.raises(ValueError, match="same length"):
            run_correlations(X, y.iloc[:-1])

    def test_empty_X_raises(self, synthetic_data: tuple) -> None:
        _, y = synthetic_data
        with pytest.raises(ValueError, match="at least one row"):
            run_correlations(
                pd.DataFrame(columns=_FEAT_NAMES), y
            )

    # -- ANOVA / categorical feature path ---------------------------------

    def test_categorical_anova_cols_present(self) -> None:
        """Categorical features should populate eta_squared and anova_p."""
        X = pd.DataFrame(
            {"cat": ["A", "A", "B", "B", "C", "C"]}
        )
        y = pd.Series([1.0, 1.1, 5.0, 5.1, 9.0, 9.1])
        result = run_correlations(X, y, categorical_cols=["cat"])
        for col in ("eta_squared", "anova_p", "anova_significant"):
            assert col in result.columns

    def test_categorical_pearson_is_nan(self) -> None:
        """Pearson and Spearman are NaN for categorical features."""
        X = pd.DataFrame(
            {"cat": ["A", "A", "B", "B", "C", "C"]}
        )
        y = pd.Series([1.0, 1.1, 5.0, 5.1, 9.0, 9.1])
        result = run_correlations(X, y, categorical_cols=["cat"])
        assert math.isnan(result.loc["cat", "pearson_r"])
        assert math.isnan(result.loc["cat", "spearman_r"])

    def test_categorical_eta_squared_range(self) -> None:
        """eta_squared must be in [0, 1]."""
        X = pd.DataFrame(
            {"cat": ["A", "A", "B", "B", "C", "C"]}
        )
        y = pd.Series([1.0, 1.1, 5.0, 5.1, 9.0, 9.1])
        result = run_correlations(X, y, categorical_cols=["cat"])
        eta = result.loc["cat", "eta_squared"]
        assert 0.0 <= eta <= 1.0 + 1e-10

    def test_categorical_significant_when_groups_differ(self) -> None:
        """Clearly separated groups should be flagged as significant."""
        X = pd.DataFrame(
            {"cat": ["A", "A", "B", "B", "C", "C"]}
        )
        y = pd.Series([1.0, 1.1, 5.0, 5.1, 9.0, 9.1])
        result = run_correlations(X, y, categorical_cols=["cat"])
        assert bool(result.loc["cat", "anova_significant"])

    def test_categorical_rank_present(self) -> None:
        """Categorical features still get a rank based on eta."""
        X = pd.DataFrame(
            {"cat": ["A", "A", "B", "B", "C", "C"]}
        )
        y = pd.Series([1.0, 1.1, 5.0, 5.1, 9.0, 9.1])
        result = run_correlations(X, y, categorical_cols=["cat"])
        assert "rank" in result.columns
        assert result.loc["cat", "rank"] >= 1

    def test_mixed_continuous_categorical(self) -> None:
        """Continuous features get Pearson; categoricals get ANOVA."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "cont": rng.normal(0, 1, 12),
                "cat": (["A"] * 4 + ["B"] * 4 + ["C"] * 4),
            }
        )
        y = pd.Series(rng.normal(0, 1, 12))
        result = run_correlations(
            X, y, method="both", categorical_cols=["cat"]
        )
        assert not math.isnan(result.loc["cont", "pearson_r"])
        assert math.isnan(result.loc["cat", "pearson_r"])
        assert not math.isnan(result.loc["cat", "eta_squared"])
