"""Pruebas unitarias para el modelo de supervivencia y análisis de riesgos proporcionales."""

from __future__ import annotations

import pandas as pd
import pytest

from gridbreak_cl.config import get_project_root
from gridbreak_cl.models.survival_analysis import SurvivalModel


@pytest.fixture
def benchmark_survival_df() -> pd.DataFrame:
    """Carga el dataset de supervivencia generado a partir del benchmark real."""
    surv_path = get_project_root() / "data" / "processed" / "survival_dataset.parquet"
    if not surv_path.exists():
        from gridbreak_cl.features.build_features import (
            process_and_save_feature_pipeline,
        )

        process_and_save_feature_pipeline()
    return pd.read_parquet(surv_path)


def test_survival_kaplan_meier_fitting(benchmark_survival_df: pd.DataFrame) -> None:
    model = SurvivalModel(penalizer=0.05)
    km_dict = model.fit_kaplan_meier(benchmark_survival_df, strata_col="tercil_nse")

    assert "Global" in km_dict
    assert "Vulnerable" in km_dict
    assert "Alto" in km_dict
    assert km_dict["Global"].survival_function_ is not None


def test_survival_logrank_significance(benchmark_survival_df: pd.DataFrame) -> None:
    model = SurvivalModel(penalizer=0.05)
    lr = model.compute_logrank_test(benchmark_survival_df, strata_col="tercil_nse")

    assert lr["strata"] == "tercil_nse"
    assert lr["p_value"] < 0.001
    assert lr["is_significant_99"] is True


def test_survival_cox_model_and_hazard_ratios(
    benchmark_survival_df: pd.DataFrame,
) -> None:
    model = SurvivalModel(penalizer=0.05)
    cph = model.fit_cox_model(benchmark_survival_df)
    assert cph is not None

    hr_df = model.get_hazard_ratios()
    assert "Variable" in hr_df.columns
    assert "Hazard Ratio (HR)" in hr_df.columns

    # Validar que NSE reduce el riesgo instantáneo (Beta negativo, HR < 1.0)
    nse_row = hr_df[hr_df["Variable"] == "nse_score"].iloc[0]
    assert float(nse_row["Coeficiente (Beta)"]) < 0.0
    assert float(nse_row["Hazard Ratio (HR)"]) < 1.0

    # Validar que red aérea incrementa el riesgo instantáneo (Beta positivo, HR > 1.0)
    aerial_row = hr_df[hr_df["Variable"] == "red_aerea_km_ratio"].iloc[0]
    assert float(aerial_row["Coeficiente (Beta)"]) > 0.0
    assert float(aerial_row["Hazard Ratio (HR)"]) > 1.0

    # Concordance index superior a 0.80
    metrics = model.get_metrics()
    assert metrics["concordance_index"] >= 0.80


def test_predict_survival_curves_and_median(
    benchmark_survival_df: pd.DataFrame,
) -> None:
    model = SurvivalModel(penalizer=0.05)
    model.fit_cox_model(benchmark_survival_df)

    sample = benchmark_survival_df.head(2)
    curves = model.predict_survival_curves(sample)
    assert len(curves) > 0

    medians = model.predict_median_survival_time(sample)
    assert len(medians) == 2
    assert all(m > 0 for m in medians)
