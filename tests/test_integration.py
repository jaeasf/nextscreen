"""End-to-end integration test exercising the full NEXTscreen pipeline.

Runs against examples/example_dataset.csv and verifies every module
in sequence without the Streamlit UI layer.

Steps covered
-------------
1. Data loading & replicate detection
2. Replicate handling (average strategy)
3. Feature selection — LASSO, Random Forest, SHAP, PCA, Correlations
4. Consensus ranking
5. Plain-English interpretations (narrator)
6. NEXTorch Bayesian Optimization
7. HTML report generation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLE_CSV = (
    Path(__file__).parent.parent / "examples" / "example_dataset.csv"
)
FEATURE_COLS = [
    "temperature",
    "pressure",
    "catalyst_loading",
    "solvent_ratio",
    "reaction_time",
]
TARGET_COLS = ["yield", "selectivity"]


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Load the example CSV once for the whole module."""
    from nextscreen.data.loader import load_file

    df = load_file(EXAMPLE_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 12
    assert set(FEATURE_COLS).issubset(df.columns)
    assert set(TARGET_COLS).issubset(df.columns)
    return df


@pytest.fixture(scope="module")
def processed_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Detect and average replicates, returning the processed frame."""
    from nextscreen.data.loader import detect_replicates, handle_replicates

    rep_df = detect_replicates(raw_df, FEATURE_COLS)
    # detect_replicates returns the replicate rows as a DataFrame.
    assert isinstance(rep_df, pd.DataFrame)
    # Example has two replicate groups.
    assert not rep_df.empty
    assert rep_df["replicate_group"].nunique() >= 1

    processed, _summary = handle_replicates(
        raw_df, FEATURE_COLS, TARGET_COLS, strategy="average"
    )
    assert isinstance(processed, pd.DataFrame)
    # Averaging should reduce row count.
    assert len(processed) < len(raw_df)
    # Feature and target columns preserved.
    assert set(FEATURE_COLS + TARGET_COLS).issubset(processed.columns)
    return processed


@pytest.fixture(scope="module")
def X(processed_df: pd.DataFrame) -> pd.DataFrame:
    return processed_df[FEATURE_COLS]


@pytest.fixture(scope="module")
def y_yield(processed_df: pd.DataFrame) -> pd.Series:
    return processed_df["yield"]


@pytest.fixture(scope="module")
def y_sel(processed_df: pd.DataFrame) -> pd.Series:
    return processed_df["selectivity"]


# ---------------------------------------------------------------------------
# Step 3 — Feature selection
# ---------------------------------------------------------------------------


class TestLasso:
    def test_runs_and_returns_dataframe(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.lasso import run_lasso

        result = run_lasso(X, y_yield)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {
            "feature",
            "coefficient",
            "abs_coefficient",
            "rank",
        }
        assert list(result["feature"]) == sorted(
            result["feature"].tolist(),
            key=lambda f: result.loc[
                result["feature"] == f, "abs_coefficient"
            ].iloc[0],
            reverse=True,
        )

    def test_all_features_present(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.lasso import run_lasso

        result = run_lasso(X, y_yield)
        assert set(result["feature"]) == set(FEATURE_COLS)

    def test_both_targets(
        self,
        X: pd.DataFrame,
        y_yield: pd.Series,
        y_sel: pd.Series,
    ) -> None:
        from nextscreen.features.lasso import run_lasso

        for y in (y_yield, y_sel):
            res = run_lasso(X, y)
            assert len(res) == len(FEATURE_COLS)


class TestRandomForest:
    def test_importances_sum_to_one(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.random_forest import run_random_forest

        result = run_random_forest(X, y_yield, n_estimators=20)
        assert isinstance(result, pd.DataFrame)
        assert abs(result["importance"].sum() - 1.0) < 1e-6

    def test_all_features_ranked(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.random_forest import run_random_forest

        result = run_random_forest(X, y_yield, n_estimators=20)
        assert set(result["feature"]) == set(FEATURE_COLS)


class TestShap:
    def test_returns_expected_keys(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.shap_analysis import run_shap

        result = run_shap(X, y_yield, background_samples=5)
        assert isinstance(result, dict)
        assert "shap_values" in result
        assert "feature_importance" in result
        assert "X_background" in result

    def test_shap_values_shape(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.shap_analysis import run_shap

        result = run_shap(X, y_yield, background_samples=5)
        sv = result["shap_values"]
        assert sv.shape == (len(X), len(FEATURE_COLS))

    def test_feature_importance_dataframe(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.shap_analysis import run_shap

        result = run_shap(X, y_yield, background_samples=5)
        fi = result["feature_importance"]
        assert isinstance(fi, pd.DataFrame)
        assert set(fi["feature"]) == set(FEATURE_COLS)
        assert (fi["mean_abs_shap"] >= 0).all()


class TestPca:
    def test_returns_expected_keys(
        self, X: pd.DataFrame
    ) -> None:
        from nextscreen.features.pca import run_pca

        result = run_pca(X)
        assert isinstance(result, dict)
        for key in (
            "explained_variance_ratio",
            "cumulative_variance",
            "n_components",
            "loadings",
            "feature_rank",
        ):
            assert key in result

    def test_cumulative_variance_threshold(
        self, X: pd.DataFrame
    ) -> None:
        from nextscreen.features.pca import run_pca

        result = run_pca(X, variance_threshold=0.90)
        evr = result["explained_variance_ratio"]
        cum = result["cumulative_variance"]
        assert float(cum[-1]) >= 0.90 or len(evr) == result["n_components"]

    def test_loadings_shape(self, X: pd.DataFrame) -> None:
        from nextscreen.features.pca import run_pca

        result = run_pca(X)
        loadings: pd.DataFrame = result["loadings"]
        assert loadings.shape[0] == len(FEATURE_COLS)
        assert loadings.shape[1] == result["n_components"]


class TestCorrelations:
    def test_both_method_returns_all_columns(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.correlations import run_correlations

        result = run_correlations(X, y_yield, method="both")
        assert isinstance(result, pd.DataFrame)
        for col in (
            "pearson_r",
            "pearson_p",
            "pearson_significant",
            "spearman_r",
            "spearman_p",
            "spearman_significant",
            "rank",
        ):
            assert col in result.columns

    def test_index_is_feature_names(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.correlations import run_correlations

        result = run_correlations(X, y_yield)
        assert set(result.index.tolist()) == set(FEATURE_COLS)

    def test_pearson_r_in_range(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.correlations import run_correlations

        result = run_correlations(X, y_yield)
        assert (result["pearson_r"].abs() <= 1.0).all()


# ---------------------------------------------------------------------------
# Step 4 — Consensus ranking
# ---------------------------------------------------------------------------


class TestConsensus:
    def test_consensus_covers_all_features(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.consensus import (
            compute_consensus,
            label_importance,
        )
        from nextscreen.features.lasso import run_lasso
        from nextscreen.features.random_forest import run_random_forest

        lasso_res = run_lasso(X, y_yield)
        rf_res = run_random_forest(X, y_yield, n_estimators=20)
        rankings = {
            "lasso": lasso_res[["feature", "rank"]],
            "random_forest": rf_res[["feature", "rank"]],
        }
        con = compute_consensus(rankings)
        assert set(con["feature"]) == set(FEATURE_COLS)
        assert "consensus_rank" in con.columns
        assert "avg_normalized_rank" in con.columns

        labelled = label_importance(con, n_methods=2)
        assert "importance_label" in labelled.columns
        valid_labels = {
            "strongly important",
            "moderately important",
            "weakly important or inconsistent",
        }
        assert set(labelled["importance_label"]).issubset(valid_labels)

    def test_consensus_rank_is_contiguous(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.features.consensus import compute_consensus
        from nextscreen.features.lasso import run_lasso
        from nextscreen.features.random_forest import run_random_forest

        rankings = {
            "lasso": run_lasso(X, y_yield)[["feature", "rank"]],
            "random_forest": run_random_forest(
                X, y_yield, n_estimators=20
            )[["feature", "rank"]],
        }
        con = compute_consensus(rankings)
        ranks = sorted(con["consensus_rank"].tolist())
        assert ranks == list(range(1, len(FEATURE_COLS) + 1))


# ---------------------------------------------------------------------------
# Step 5 — Narrator interpretations
# ---------------------------------------------------------------------------


class TestNarrator:
    @pytest.fixture(scope="class")
    def lasso_result(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> pd.DataFrame:
        from nextscreen.features.lasso import run_lasso

        return run_lasso(X, y_yield)

    @pytest.fixture(scope="class")
    def rf_result(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> pd.DataFrame:
        from nextscreen.features.random_forest import run_random_forest

        return run_random_forest(X, y_yield, n_estimators=20)

    @pytest.fixture(scope="class")
    def shap_result(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> dict:
        from nextscreen.features.shap_analysis import run_shap

        return run_shap(X, y_yield, background_samples=5)

    @pytest.fixture(scope="class")
    def pca_result(self, X: pd.DataFrame) -> dict:
        from nextscreen.features.pca import run_pca

        return run_pca(X)

    @pytest.fixture(scope="class")
    def corr_result(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> pd.DataFrame:
        from nextscreen.features.correlations import run_correlations

        return run_correlations(X, y_yield, method="both")

    def test_interpret_lasso_returns_string(
        self, lasso_result: pd.DataFrame
    ) -> None:
        from nextscreen.interpretation.narrator import interpret_lasso

        text = interpret_lasso(lasso_result, "yield")
        assert isinstance(text, str)
        assert len(text) > 20
        assert "yield" in text.lower() or "feature" in text.lower()

    def test_interpret_rf_returns_string(
        self, rf_result: pd.DataFrame
    ) -> None:
        from nextscreen.interpretation.narrator import (
            interpret_random_forest,
        )

        text = interpret_random_forest(rf_result, "yield")
        assert isinstance(text, str)
        assert len(text) > 20

    def test_interpret_shap_returns_string(
        self, shap_result: dict
    ) -> None:
        from nextscreen.interpretation.narrator import interpret_shap

        text = interpret_shap(shap_result, "yield")
        assert isinstance(text, str)
        assert len(text) > 20

    def test_interpret_pca_returns_string(
        self, pca_result: dict
    ) -> None:
        from nextscreen.interpretation.narrator import interpret_pca

        text = interpret_pca(pca_result)
        assert isinstance(text, str)
        assert len(text) > 20

    def test_interpret_correlations_returns_string(
        self, corr_result: pd.DataFrame
    ) -> None:
        from nextscreen.interpretation.narrator import (
            interpret_correlations,
        )

        text = interpret_correlations(corr_result, "yield")
        assert isinstance(text, str)
        assert len(text) > 20

    def test_interpret_consensus_returns_string(
        self,
        X: pd.DataFrame,
        y_yield: pd.Series,
    ) -> None:
        from nextscreen.features.consensus import (
            compute_consensus,
            label_importance,
        )
        from nextscreen.features.lasso import run_lasso
        from nextscreen.features.random_forest import run_random_forest
        from nextscreen.interpretation.narrator import (
            interpret_consensus,
        )

        rankings = {
            "lasso": run_lasso(X, y_yield)[["feature", "rank"]],
            "random_forest": run_random_forest(
                X, y_yield, n_estimators=20
            )[["feature", "rank"]],
        }
        con = label_importance(
            compute_consensus(rankings), n_methods=2
        )
        text = interpret_consensus(con, "yield", n_methods=2)
        assert isinstance(text, str)
        assert len(text) > 20
        assert "Domain expertise" in text


# ---------------------------------------------------------------------------
# Step 6 — Bayesian Optimization
# ---------------------------------------------------------------------------


class TestBayesianOptimization:
    def test_single_suggestion(
        self, processed_df: pd.DataFrame, y_yield: pd.Series
    ) -> None:
        from nextscreen.nextorch_integration.handoff import (
            build_parameter_space,
            run_optimization,
        )

        bounds = {
            "temperature": {
                "lower": 100.0,
                "upper": 300.0,
                "type": "continuous",
            },
            "pressure": {
                "lower": 1.0,
                "upper": 3.0,
                "type": "continuous",
            },
        }
        params = build_parameter_space(
            ["temperature", "pressure"], bounds
        )
        assert len(params) == 2

        result = run_optimization(
            params,
            processed_df,
            target_col="yield",
            n_suggestions=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 4)  # 2 features + pred + uncertainty
        assert "predicted_yield" in result.columns
        assert "uncertainty" in result.columns
        assert (result["uncertainty"] >= 0).all()

    def test_batch_suggestions(
        self, processed_df: pd.DataFrame
    ) -> None:
        from nextscreen.nextorch_integration.handoff import (
            build_parameter_space,
            run_optimization,
        )

        bounds = {
            f: {"lower": 0.0, "upper": 1.0, "type": "continuous"}
            for f in ["temperature", "pressure", "catalyst_loading"]
        }
        bounds["temperature"] = {
            "lower": 100.0,
            "upper": 300.0,
            "type": "continuous",
        }
        bounds["pressure"] = {
            "lower": 1.0,
            "upper": 3.0,
            "type": "continuous",
        }
        bounds["catalyst_loading"] = {
            "lower": 0.05,
            "upper": 0.20,
            "type": "continuous",
        }
        params = build_parameter_space(
            ["temperature", "pressure", "catalyst_loading"], bounds
        )
        result = run_optimization(
            params,
            processed_df,
            target_col="selectivity",
            n_suggestions=3,
        )
        assert result.shape[0] == 3
        assert "predicted_selectivity" in result.columns

    def test_integer_parameter(
        self, processed_df: pd.DataFrame
    ) -> None:
        from nextscreen.nextorch_integration.handoff import (
            build_parameter_space,
            run_optimization,
        )

        bounds = {
            "temperature": {
                "lower": 100.0,
                "upper": 300.0,
                "type": "continuous",
            },
            "reaction_time": {
                "lower": 60.0,
                "upper": 120.0,
                "type": "integer",
            },
        }
        params = build_parameter_space(
            ["temperature", "reaction_time"], bounds
        )
        result = run_optimization(
            params,
            processed_df,
            target_col="yield",
            n_suggestions=2,
        )
        assert result.shape[0] == 2

    def test_out_of_range_n_suggestions_raises(
        self, processed_df: pd.DataFrame
    ) -> None:
        from nextscreen.nextorch_integration.handoff import (
            build_parameter_space,
            run_optimization,
        )

        bounds = {
            "temperature": {
                "lower": 100.0,
                "upper": 300.0,
                "type": "continuous",
            }
        }
        params = build_parameter_space(["temperature"], bounds)
        with pytest.raises(ValueError, match="n_suggestions"):
            run_optimization(
                params,
                processed_df,
                target_col="yield",
                n_suggestions=0,
            )


# ---------------------------------------------------------------------------
# Step 7 — HTML report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    @pytest.fixture(scope="class")
    def feature_results(
        self, X: pd.DataFrame, y_yield: pd.Series
    ) -> dict[str, dict[str, object]]:
        from nextscreen.features.correlations import run_correlations
        from nextscreen.features.lasso import run_lasso
        from nextscreen.features.pca import run_pca
        from nextscreen.features.random_forest import run_random_forest
        from nextscreen.features.shap_analysis import run_shap

        return {
            "yield": {
                "lasso": run_lasso(X, y_yield),
                "random_forest": run_random_forest(
                    X, y_yield, n_estimators=20
                ),
                "shap": run_shap(
                    X, y_yield, background_samples=5
                ),
                "pca": run_pca(X),
                "correlations": run_correlations(
                    X, y_yield, method="both"
                ),
            }
        }

    @pytest.fixture(scope="class")
    def consensus_results(
        self,
        feature_results: dict[str, dict[str, object]],
    ) -> dict[str, pd.DataFrame]:
        from nextscreen.features.consensus import (
            compute_consensus,
            label_importance,
        )

        out: dict[str, pd.DataFrame] = {}
        for target, results in feature_results.items():
            rankings = {}
            for method in ("lasso", "random_forest"):
                res = results.get(method)
                if isinstance(res, pd.DataFrame):
                    rankings[method] = res[["feature", "rank"]]
            con = compute_consensus(rankings)
            out[target] = label_importance(
                con, n_methods=len(rankings)
            )
        return out

    def test_html_report_created(
        self,
        feature_results: dict[str, dict[str, object]],
        consensus_results: dict[str, pd.DataFrame],
    ) -> None:
        from nextscreen.reporting.report import build_html_report

        with tempfile.TemporaryDirectory() as tmp:
            path = build_html_report(
                dataset_summary={
                    "file_name": "example_dataset.csv",
                    "n_rows": 10,
                    "n_features": 5,
                    "target_cols": ["yield"],
                    "replicate_strategy": "average",
                },
                feature_results=feature_results,
                consensus_results=consensus_results,
                selected_features=["temperature", "pressure"],
                bounds={
                    "temperature": {
                        "lower": 100.0,
                        "upper": 300.0,
                        "type": "continuous",
                    },
                    "pressure": {
                        "lower": 1.0,
                        "upper": 3.0,
                        "type": "continuous",
                    },
                },
                suggested_experiments=None,
                output_dir=Path(tmp),
            )
            assert path.exists()
            assert path.suffix == ".html"
            content = path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert "NEXTscreen" in content

    def test_report_contains_all_sections(
        self,
        feature_results: dict[str, dict[str, object]],
        consensus_results: dict[str, pd.DataFrame],
    ) -> None:
        from nextscreen.reporting.report import build_html_report

        with tempfile.TemporaryDirectory() as tmp:
            path = build_html_report(
                dataset_summary={
                    "file_name": "example_dataset.csv",
                    "n_rows": 10,
                    "n_features": 5,
                    "target_cols": ["yield"],
                    "replicate_strategy": "average",
                },
                feature_results=feature_results,
                consensus_results=consensus_results,
                selected_features=["temperature", "pressure"],
                bounds={
                    "temperature": {
                        "lower": 100.0,
                        "upper": 300.0,
                        "type": "continuous",
                    },
                    "pressure": {
                        "lower": 1.0,
                        "upper": 3.0,
                        "type": "continuous",
                    },
                },
                suggested_experiments=None,
                output_dir=Path(tmp),
            )
            content = path.read_text(encoding="utf-8")
            for section in (
                "Dataset Summary",
                "Feature Selection Results",
                "Selected Features",
                "LASSO",
                "Random Forest",
                "SHAP",
                "PCA",
                "Correlations",
                "Consensus",
            ):
                assert section in content, (
                    f"Section '{section}' missing from report"
                )

    def test_report_embeds_plots_as_png(
        self,
        feature_results: dict[str, dict[str, object]],
        consensus_results: dict[str, pd.DataFrame],
    ) -> None:
        from nextscreen.reporting.report import build_html_report

        with tempfile.TemporaryDirectory() as tmp:
            path = build_html_report(
                dataset_summary={
                    "file_name": "example_dataset.csv",
                    "n_rows": 10,
                    "n_features": 5,
                    "target_cols": ["yield"],
                    "replicate_strategy": "average",
                },
                feature_results=feature_results,
                consensus_results=consensus_results,
                selected_features=["temperature"],
                bounds={
                    "temperature": {
                        "lower": 100.0,
                        "upper": 300.0,
                        "type": "continuous",
                    }
                },
                suggested_experiments=None,
                output_dir=Path(tmp),
            )
            content = path.read_text(encoding="utf-8")
            # Kaleido-generated PNGs are embedded as data URIs.
            assert "data:image/png;base64," in content

    def test_report_includes_suggested_experiments(
        self,
        feature_results: dict[str, dict[str, object]],
        consensus_results: dict[str, pd.DataFrame],
        processed_df: pd.DataFrame,
    ) -> None:
        from nextscreen.nextorch_integration.handoff import (
            build_parameter_space,
            run_optimization,
        )
        from nextscreen.reporting.report import build_html_report

        bounds = {
            "temperature": {
                "lower": 100.0,
                "upper": 300.0,
                "type": "continuous",
            },
            "pressure": {
                "lower": 1.0,
                "upper": 3.0,
                "type": "continuous",
            },
        }
        params = build_parameter_space(
            ["temperature", "pressure"], bounds
        )
        suggestions = run_optimization(
            params,
            processed_df,
            target_col="yield",
            n_suggestions=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = build_html_report(
                dataset_summary={
                    "file_name": "example_dataset.csv",
                    "n_rows": 10,
                    "n_features": 5,
                    "target_cols": ["yield"],
                    "replicate_strategy": "average",
                },
                feature_results=feature_results,
                consensus_results=consensus_results,
                selected_features=["temperature", "pressure"],
                bounds=bounds,
                suggested_experiments=suggestions,
                output_dir=Path(tmp),
            )
            content = path.read_text(encoding="utf-8")
            assert "Suggested Experiments" in content
            assert "predicted_yield" in content


# ---------------------------------------------------------------------------
# Full pipeline smoke test
# ---------------------------------------------------------------------------


def test_full_pipeline_smoke(tmp_path: Path) -> None:
    """Single end-to-end smoke test: load → process → analyse → report."""
    from nextscreen.data.loader import (
        detect_replicates,
        handle_replicates,
        load_file,
    )
    from nextscreen.features.consensus import (
        compute_consensus,
        label_importance,
    )
    from nextscreen.features.correlations import run_correlations
    from nextscreen.features.lasso import run_lasso
    from nextscreen.features.pca import run_pca
    from nextscreen.features.random_forest import run_random_forest
    from nextscreen.features.shap_analysis import run_shap
    from nextscreen.nextorch_integration.handoff import (
        build_parameter_space,
        run_optimization,
    )
    from nextscreen.reporting.report import build_html_report

    # 1. Load
    df = load_file(EXAMPLE_CSV)
    assert len(df) >= 12

    # 2. Detect & handle replicates
    rep_df = detect_replicates(df, FEATURE_COLS)
    assert not rep_df.empty
    assert rep_df["replicate_group"].nunique() >= 1
    processed, _summary = handle_replicates(
        df, FEATURE_COLS, TARGET_COLS, strategy="average"
    )
    assert len(processed) < len(df)

    X = processed[FEATURE_COLS]
    y = processed["yield"]

    # 3. Feature selection (all methods)
    lasso_res = run_lasso(X, y)
    rf_res = run_random_forest(X, y, n_estimators=20)
    shap_res = run_shap(X, y, background_samples=5)
    pca_res = run_pca(X)
    corr_res = run_correlations(X, y, method="both")

    results: dict[str, object] = {
        "lasso": lasso_res,
        "random_forest": rf_res,
        "shap": shap_res,
        "pca": pca_res,
        "correlations": corr_res,
    }

    # 4. Consensus — mirror the app's _prep_rankings logic.
    rankings: dict[str, pd.DataFrame] = {}
    for mname, res in results.items():
        if mname == "correlations":
            # Index is feature name; reset gives a 'feature' column.
            rankings[mname] = (  # type: ignore[index]
                res[["rank"]].reset_index()
            )
        elif mname == "shap":
            rankings[mname] = res["feature_importance"][  # type: ignore[index]
                ["feature", "rank"]
            ]
        elif mname == "pca":
            rankings[mname] = res["feature_rank"][  # type: ignore[index]
                ["feature", "rank"]
            ]
        else:
            rankings[mname] = res[["feature", "rank"]]  # type: ignore[index]
    con = compute_consensus(rankings)
    con = label_importance(con, n_methods=len(rankings))
    assert len(con) == len(FEATURE_COLS)

    # 5. BO (2 suggestions for top-2 features)
    top2 = con.head(2)["feature"].tolist()
    bounds = {
        "temperature": {
            "lower": 100.0,
            "upper": 300.0,
            "type": "continuous",
        },
        "pressure": {
            "lower": 1.0,
            "upper": 3.0,
            "type": "continuous",
        },
    }
    feat_for_bo = [f for f in top2 if f in bounds]
    if not feat_for_bo:
        feat_for_bo = list(bounds.keys())
    params = build_parameter_space(feat_for_bo, bounds)
    suggestions = run_optimization(
        params, processed, target_col="yield", n_suggestions=2
    )
    assert len(suggestions) == 2

    # 6. Report
    html_path = build_html_report(
        dataset_summary={
            "file_name": EXAMPLE_CSV.name,
            "n_rows": len(processed),
            "n_features": len(FEATURE_COLS),
            "target_cols": ["yield"],
            "replicate_strategy": "average",
        },
        feature_results={"yield": results},
        consensus_results={"yield": con},
        selected_features=feat_for_bo,
        bounds={f: bounds[f] for f in feat_for_bo},
        suggested_experiments=suggestions,
        output_dir=tmp_path,
    )
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in content
    assert "predicted_yield" in content
    assert html_path.stat().st_size > 50_000
