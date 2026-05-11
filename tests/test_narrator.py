"""Tests for nextscreen.interpretation.narrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nextscreen.interpretation.narrator import (
    interpret_correlations,
    interpret_consensus,
    interpret_lasso,
    interpret_pca,
    interpret_random_forest,
    interpret_shap,
)

_CAVEAT = "Domain expertise should guide final decisions"
_TARGET = "yield"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def lasso_result() -> pd.DataFrame:
    """Three features: two nonzero, one zero-weight."""
    return pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c"],
            "coefficient": [1.5, -0.3, 0.0],
            "abs_coefficient": [1.5, 0.3, 0.0],
            "rank": [1, 2, 3],
        }
    )


@pytest.fixture()
def lasso_all_zero() -> pd.DataFrame:
    """All features shrunk to zero by LASSO."""
    return pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b"],
            "coefficient": [0.0, 0.0],
            "abs_coefficient": [0.0, 0.0],
            "rank": [1, 2],
        }
    )


@pytest.fixture()
def rf_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c"],
            "importance": [0.7, 0.2, 0.1],
            "rank": [1, 2, 3],
        }
    )


@pytest.fixture()
def shap_result() -> dict:
    fi = pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c"],
            "mean_abs_shap": [0.5, 0.2, 0.1],
            "rank": [1, 2, 3],
        }
    )
    return {
        "feature_importance": fi,
        "shap_values": np.zeros((5, 3)),
        "X_background": pd.DataFrame(),
    }


@pytest.fixture()
def pca_result() -> dict:
    loadings = pd.DataFrame(
        {
            "PC1": [0.8, 0.3, -0.1],
            "PC2": [0.1, 0.7, 0.2],
        },
        index=["feat_a", "feat_b", "feat_c"],
    )
    feature_rank = pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c"],
            "max_loading": [0.8, 0.7, 0.2],
            "rank": [1, 2, 3],
        }
    )
    return {
        "n_components": 2,
        "explained_variance_ratio": np.array([0.6, 0.2]),
        "cumulative_variance": np.array([0.6, 0.8]),
        "loadings": loadings,
        "feature_rank": feature_rank,
    }


@pytest.fixture()
def corr_result_both() -> pd.DataFrame:
    """Both Pearson and Spearman columns present."""
    return pd.DataFrame(
        {
            "pearson_r": [0.9, 0.05, -0.02],
            "pearson_p": [0.001, 0.80, 0.90],
            "pearson_significant": [True, False, False],
            "spearman_r": [0.85, 0.06, -0.03],
            "spearman_p": [0.002, 0.75, 0.88],
            "spearman_significant": [True, False, False],
            "rank": [1, 2, 3],
        },
        index=pd.Index(
            ["feat_a", "feat_b", "feat_c"], name="feature"
        ),
    )


@pytest.fixture()
def corr_result_pearson_only() -> pd.DataFrame:
    """Only Pearson columns present."""
    return pd.DataFrame(
        {
            "pearson_r": [0.9, 0.04],
            "pearson_p": [0.001, 0.85],
            "pearson_significant": [True, False],
            "rank": [1, 2],
        },
        index=pd.Index(["feat_a", "feat_b"], name="feature"),
    )


@pytest.fixture()
def corr_result_none_significant() -> pd.DataFrame:
    """No features are statistically significant."""
    return pd.DataFrame(
        {
            "pearson_r": [0.04, 0.02],
            "pearson_p": [0.85, 0.92],
            "pearson_significant": [False, False],
            "rank": [1, 2],
        },
        index=pd.Index(["feat_a", "feat_b"], name="feature"),
    )


@pytest.fixture()
def consensus_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c"],
            "lasso_rank": [1, 2, 3],
            "rf_rank": [1, 3, 2],
            "avg_normalized_rank": [1 / 3, 5 / 6, 5 / 6],
            "consensus_rank": [1, 2, 3],
            "n_methods_top_k": [2, 1, 0],
            "importance_label": [
                "strongly important",
                "moderately important",
                "weakly important or inconsistent",
            ],
        }
    )


@pytest.fixture()
def consensus_with_pca_corr() -> pd.DataFrame:
    """Consensus with pca_rank and pearson_rank columns."""
    return pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c", "feat_d"],
            "pca_rank": [1, 2, 4, 3],
            "pearson_rank": [4, 3, 1, 2],
            "consensus_rank": [1, 2, 3, 4],
            "n_methods_top_k": [2, 1, 1, 0],
            "importance_label": [
                "strongly important",
                "moderately important",
                "moderately important",
                "weakly important or inconsistent",
            ],
        }
    )


# ---------------------------------------------------------------------------
# interpret_lasso
# ---------------------------------------------------------------------------


class TestInterpretLasso:
    def test_returns_string(self, lasso_result: pd.DataFrame) -> None:
        out = interpret_lasso(lasso_result, _TARGET)
        assert isinstance(out, str)

    def test_contains_caveat(
        self, lasso_result: pd.DataFrame
    ) -> None:
        out = interpret_lasso(lasso_result, _TARGET)
        assert _CAVEAT in out

    def test_mentions_top_feature(
        self, lasso_result: pd.DataFrame
    ) -> None:
        out = interpret_lasso(lasso_result, _TARGET)
        assert "feat_a" in out

    def test_mentions_target_name(
        self, lasso_result: pd.DataFrame
    ) -> None:
        out = interpret_lasso(lasso_result, _TARGET)
        assert _TARGET in out

    def test_zero_weight_rule(
        self, lasso_result: pd.DataFrame
    ) -> None:
        """feat_c has coef=0 → zero-weight message."""
        out = interpret_lasso(lasso_result, _TARGET)
        assert "feat_c" in out
        assert "zero weight" in out
        assert "linear assumptions" in out

    def test_zero_weight_singular_pronoun(
        self, lasso_result: pd.DataFrame
    ) -> None:
        """Single zero-weight feature uses 'it'."""
        out = interpret_lasso(lasso_result, _TARGET)
        assert "suggesting it may be irrelevant" in out

    def test_all_zero_weight(
        self, lasso_all_zero: pd.DataFrame
    ) -> None:
        out = interpret_lasso(lasso_all_zero, _TARGET)
        assert "zero weight to all features" in out
        assert _CAVEAT in out

    def test_nonzero_features_listed(
        self, lasso_result: pd.DataFrame
    ) -> None:
        """feat_b is also nonzero and should be mentioned."""
        out = interpret_lasso(lasso_result, _TARGET)
        assert "feat_b" in out

    def test_coefficient_value_shown(
        self, lasso_result: pd.DataFrame
    ) -> None:
        out = interpret_lasso(lasso_result, _TARGET)
        assert "+1.5000" in out

    def test_all_nonzero_no_zero_message(self) -> None:
        result = pd.DataFrame(
            {
                "feature": ["a", "b"],
                "coefficient": [1.0, 0.5],
                "abs_coefficient": [1.0, 0.5],
                "rank": [1, 2],
            }
        )
        out = interpret_lasso(result, _TARGET)
        assert "zero weight" not in out


# ---------------------------------------------------------------------------
# interpret_random_forest
# ---------------------------------------------------------------------------


class TestInterpretRandomForest:
    def test_returns_string(self, rf_result: pd.DataFrame) -> None:
        out = interpret_random_forest(rf_result, _TARGET)
        assert isinstance(out, str)

    def test_contains_caveat(self, rf_result: pd.DataFrame) -> None:
        out = interpret_random_forest(rf_result, _TARGET)
        assert _CAVEAT in out

    def test_mentions_top_feature(
        self, rf_result: pd.DataFrame
    ) -> None:
        out = interpret_random_forest(rf_result, _TARGET)
        assert "feat_a" in out

    def test_mentions_importance_score(
        self, rf_result: pd.DataFrame
    ) -> None:
        out = interpret_random_forest(rf_result, _TARGET)
        assert "0.7000" in out

    def test_mentions_target_name(
        self, rf_result: pd.DataFrame
    ) -> None:
        out = interpret_random_forest(rf_result, _TARGET)
        assert _TARGET in out

    def test_other_features_listed(
        self, rf_result: pd.DataFrame
    ) -> None:
        out = interpret_random_forest(rf_result, _TARGET)
        assert "feat_b" in out
        assert "feat_c" in out

    def test_single_feature(self) -> None:
        result = pd.DataFrame(
            {"feature": ["a"], "importance": [1.0], "rank": [1]}
        )
        out = interpret_random_forest(result, _TARGET)
        assert isinstance(out, str)
        assert _CAVEAT in out


# ---------------------------------------------------------------------------
# interpret_shap
# ---------------------------------------------------------------------------


class TestInterpretShap:
    def test_returns_string(self, shap_result: dict) -> None:
        out = interpret_shap(shap_result, _TARGET)
        assert isinstance(out, str)

    def test_contains_caveat(self, shap_result: dict) -> None:
        out = interpret_shap(shap_result, _TARGET)
        assert _CAVEAT in out

    def test_mentions_top_feature(self, shap_result: dict) -> None:
        out = interpret_shap(shap_result, _TARGET)
        assert "feat_a" in out

    def test_mentions_shap_value(self, shap_result: dict) -> None:
        out = interpret_shap(shap_result, _TARGET)
        assert "0.5000" in out

    def test_mentions_target_name(self, shap_result: dict) -> None:
        out = interpret_shap(shap_result, _TARGET)
        assert _TARGET in out

    def test_secondary_features_listed(
        self, shap_result: dict
    ) -> None:
        out = interpret_shap(shap_result, _TARGET)
        assert "feat_b" in out

    def test_single_feature(self) -> None:
        fi = pd.DataFrame(
            {
                "feature": ["a"],
                "mean_abs_shap": [0.9],
                "rank": [1],
            }
        )
        out = interpret_shap(
            {"feature_importance": fi}, _TARGET
        )
        assert isinstance(out, str)
        assert _CAVEAT in out


# ---------------------------------------------------------------------------
# interpret_pca
# ---------------------------------------------------------------------------


class TestInterpretPca:
    def test_returns_string(self, pca_result: dict) -> None:
        out = interpret_pca(pca_result)
        assert isinstance(out, str)

    def test_contains_caveat(self, pca_result: dict) -> None:
        out = interpret_pca(pca_result)
        assert _CAVEAT in out

    def test_mentions_n_components(self, pca_result: dict) -> None:
        out = interpret_pca(pca_result)
        assert "2 component" in out

    def test_mentions_variance_percentage(
        self, pca_result: dict
    ) -> None:
        out = interpret_pca(pca_result)
        assert "80.0%" in out

    def test_mentions_pc1_top_feature(
        self, pca_result: dict
    ) -> None:
        """feat_a has highest loading in PC1."""
        out = interpret_pca(pca_result)
        assert "PC1" in out
        assert "feat_a" in out

    def test_mentions_overall_top_feature(
        self, pca_result: dict
    ) -> None:
        out = interpret_pca(pca_result)
        # feat_a has max_loading=0.8 overall
        assert "feat_a" in out

    def test_single_component(self) -> None:
        loadings = pd.DataFrame(
            {"PC1": [0.9, 0.1]},
            index=["x", "y"],
        )
        feat_rank = pd.DataFrame(
            {"feature": ["x", "y"], "max_loading": [0.9, 0.1],
             "rank": [1, 2]}
        )
        result = {
            "n_components": 1,
            "explained_variance_ratio": np.array([0.75]),
            "cumulative_variance": np.array([0.75]),
            "loadings": loadings,
            "feature_rank": feat_rank,
        }
        out = interpret_pca(result)
        assert "1 component" in out
        assert "75.0%" in out


# ---------------------------------------------------------------------------
# interpret_correlations
# ---------------------------------------------------------------------------


class TestInterpretCorrelations:
    def test_returns_string(
        self, corr_result_both: pd.DataFrame
    ) -> None:
        out = interpret_correlations(corr_result_both, _TARGET)
        assert isinstance(out, str)

    def test_contains_caveat(
        self, corr_result_both: pd.DataFrame
    ) -> None:
        out = interpret_correlations(corr_result_both, _TARGET)
        assert _CAVEAT in out

    def test_mentions_significant_pearson_feature(
        self, corr_result_both: pd.DataFrame
    ) -> None:
        out = interpret_correlations(corr_result_both, _TARGET)
        assert "feat_a" in out

    def test_no_linear_correlation_rule(
        self, corr_result_both: pd.DataFrame
    ) -> None:
        """Features with |r|<0.1 and p>0.05 trigger the rule."""
        out = interpret_correlations(corr_result_both, _TARGET)
        assert "no significant linear correlation" in out
        # feat_b and feat_c both qualify
        assert "feat_b" in out
        assert "feat_c" in out

    def test_correlation_strength_strong(
        self, corr_result_both: pd.DataFrame
    ) -> None:
        """r=0.9 → 'strong'."""
        out = interpret_correlations(corr_result_both, _TARGET)
        assert "strong" in out

    def test_pearson_only(
        self, corr_result_pearson_only: pd.DataFrame
    ) -> None:
        out = interpret_correlations(
            corr_result_pearson_only, _TARGET
        )
        assert "Pearson" in out
        assert "Spearman" not in out

    def test_spearman_mentioned_in_both(
        self, corr_result_both: pd.DataFrame
    ) -> None:
        out = interpret_correlations(corr_result_both, _TARGET)
        assert "Spearman" in out

    def test_no_significant_features(
        self, corr_result_none_significant: pd.DataFrame
    ) -> None:
        out = interpret_correlations(
            corr_result_none_significant, _TARGET
        )
        assert "No features" in out or "no significant" in out

    def test_mentions_r_value(
        self, corr_result_both: pd.DataFrame
    ) -> None:
        out = interpret_correlations(corr_result_both, _TARGET)
        assert "r = 0.900" in out

    def test_mentions_target_name(
        self, corr_result_both: pd.DataFrame
    ) -> None:
        out = interpret_correlations(corr_result_both, _TARGET)
        assert _TARGET in out


# ---------------------------------------------------------------------------
# interpret_consensus
# ---------------------------------------------------------------------------


class TestInterpretConsensus:
    def test_returns_string(
        self, consensus_result: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_result, _TARGET, 2)
        assert isinstance(out, str)

    def test_contains_caveat(
        self, consensus_result: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_result, _TARGET, 2)
        assert _CAVEAT in out

    def test_mentions_top_feature(
        self, consensus_result: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_result, _TARGET, 2)
        assert "feat_a" in out

    def test_mentions_target_name(
        self, consensus_result: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_result, _TARGET, 2)
        assert _TARGET in out

    def test_mentions_n_methods(
        self, consensus_result: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_result, _TARGET, 2)
        assert "2 method" in out

    def test_strongly_important_label(
        self, consensus_result: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_result, _TARGET, 2)
        assert "strongly important" in out
        assert "feat_a" in out

    def test_moderately_important_label(
        self, consensus_result: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_result, _TARGET, 2)
        assert "moderately important" in out
        assert "feat_b" in out

    def test_weakly_important_label(
        self, consensus_result: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_result, _TARGET, 2)
        assert "weakly important or inconsistent" in out

    def test_pca_correlation_interaction_flag(
        self, consensus_with_pca_corr: pd.DataFrame
    ) -> None:
        """feat_a: pca_rank=1 (top), pearson_rank=4 (bottom)."""
        out = interpret_consensus(
            consensus_with_pca_corr, _TARGET, 2
        )
        assert "nonlinear/interaction effect" in out
        assert "feat_a" in out

    def test_no_pca_flag_without_columns(
        self, consensus_result: pd.DataFrame
    ) -> None:
        """No pca_rank column → no interaction flag."""
        out = interpret_consensus(consensus_result, _TARGET, 2)
        assert "nonlinear/interaction effect" not in out
