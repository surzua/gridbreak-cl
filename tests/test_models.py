"""Validación de convergencia y coherencia física del modelo de fragilidad."""

import numpy as np
import pandas as pd
import pytest

from gridbreak_cl.models.fragility_curves import FragilityModel


@pytest.fixture
def synthetic_storm_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 400
    precip = np.random.uniform(5, 70, n)
    rafaga = np.random.uniform(20, 110, n)
    rafaga_cuad = (rafaga**2) / 100.0
    nse = np.random.choice([-1.5, 0.0, 1.5], n)  # Vulnerable, Medio, Acomodado
    red_aerea = np.where(nse < 0, 0.90, np.where(nse > 0, 0.35, 0.70))
    es_cge = np.random.choice([0.0, 1.0], n)
    arbolado = np.random.uniform(2, 12, n)

    # Probabilidad con mayor susceptibilidad en NSE bajo ante lluvia y viento
    logit = (
        -4.0
        + 0.08 * precip
        + 0.04 * rafaga
        + 0.02 * rafaga_cuad
        - 0.80 * nse
        - 0.02 * (precip * nse)
        - 0.015 * (rafaga * nse)
        + 1.50 * (red_aerea - 0.70)
        + 0.30 * es_cge
        + 0.04 * arbolado
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    corte = (np.random.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame(
        {
            "precip_acumulada_mm": precip,
            "rafaga_max_kmh": rafaga,
            "rafaga_cuadratica": rafaga_cuad,
            "nse_score": nse,
            "red_aerea_km_ratio": red_aerea,
            "empresa": np.where(es_cge == 1.0, "CGE", "ENEL"),
            "arbolado_m2_hab": arbolado,
            "es_corte_critico": corte,
        }
    )


def test_model_fit_and_vulnerability_gap(synthetic_storm_data: pd.DataFrame) -> None:
    model = FragilityModel()
    results = model.fit(synthetic_storm_data)
    assert results is not None
    assert "precip_x_nse" in results.params
    assert "red_aerea_km_ratio" in results.params
    assert "rafaga_cuadratica" in results.params

    # A mismo temporal (30 mm de agua y 60 km/h de viento):
    prob_vulnerable = model.predict_probability(
        precip_mm=30.0, rafaga_kmh=60.0, nse_score=-1.5, red_aerea_ratio=0.90
    )
    prob_acomodado = model.predict_probability(
        precip_mm=30.0, rafaga_kmh=60.0, nse_score=1.5, red_aerea_ratio=0.35
    )

    # La comuna vulnerable debe tener mayor probabilidad de colapso
    assert prob_vulnerable > prob_acomodado


def test_critical_threshold_analytical_solution(
    synthetic_storm_data: pd.DataFrame,
) -> None:
    model = FragilityModel()
    model.fit(synthetic_storm_data)

    # Umbral de lluvia para 50% de corte con ráfaga base de 50 km/h
    r50_vulnerable = model.compute_critical_rain_threshold(
        rafaga_kmh=50.0, nse_score=-1.5, red_aerea_ratio=0.90
    )
    r50_acomodado = model.compute_critical_rain_threshold(
        rafaga_kmh=50.0, nse_score=1.5, red_aerea_ratio=0.35
    )

    # La comuna vulnerable debe requerir menos lluvia para colapsar
    assert r50_vulnerable < r50_acomodado

    # Umbral de viento analítico
    w50_vulnerable = model.compute_critical_wind_threshold(
        precip_mm=25.0, nse_score=-1.5, red_aerea_ratio=0.90
    )
    w50_acomodado = model.compute_critical_wind_threshold(
        precip_mm=25.0, nse_score=1.5, red_aerea_ratio=0.35
    )
    assert w50_vulnerable < w50_acomodado


def test_fragility_surface_generation(synthetic_storm_data: pd.DataFrame) -> None:
    model = FragilityModel()
    model.fit(synthetic_storm_data)

    r_grid, w_grid, prob_grid = model.generate_fragility_surface(
        nse_score=0.0, resolution=15
    )
    assert r_grid.shape == (15, 15)
    assert w_grid.shape == (15, 15)
    assert prob_grid.shape == (15, 15)
    assert np.all((prob_grid >= 0.0) & (prob_grid <= 1.0))


def test_model_diagnostics_and_real_benchmark_fit() -> None:
    from gridbreak_cl.config import get_project_root

    parquet_path = (
        get_project_root() / "data" / "processed" / "benchmark_temporales_2024.parquet"
    )
    assert parquet_path.exists()

    df_real = pd.read_parquet(parquet_path)
    model = FragilityModel()
    results = model.fit(df_real)
    assert results is not None

    metrics = model.get_metrics()
    assert "aic" in metrics
    assert "pseudo_r2_mcfadden" in metrics
    assert metrics["nobs"] == len(df_real)

    summary_df = model.get_model_summary_df()
    assert isinstance(summary_df, pd.DataFrame)
    assert "Variable" in summary_df.columns
    assert "Odds Ratio" in summary_df.columns
    assert "p-value" in summary_df.columns

    # Validar coherencia causal sobre datos de benchmark
    params = results.params
    assert params["nse_score"] < 0  # Mayor ingreso reduce fragilidad
    assert params["red_aerea_km_ratio"] > 0  # Más cable aéreo incrementa fragilidad
    assert params["precip_acumulada_mm"] > 0  # Más lluvia incrementa fragilidad
    assert params["rafaga_cuadratica"] > 0  # Viento cuadrático incrementa fragilidad
