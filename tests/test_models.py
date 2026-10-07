"""Validación de convergencia y coherencia física del modelo de fragilidad."""

import numpy as np
import pandas as pd
import pytest

from gridbreak_cl.models.fragility_curves import FragilityModel


@pytest.fixture
def synthetic_storm_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 350
    precip = np.random.uniform(5, 70, n)
    rafaga = np.random.uniform(20, 110, n)
    nse = np.random.choice([-1.5, 0.0, 1.5], n)  # Vulnerable, Medio, Acomodado
    es_cge = np.random.choice([0.0, 1.0], n)
    arbolado = np.random.uniform(2, 12, n)

    # Probabilidad con mayor susceptibilidad en NSE bajo ante lluvia y viento
    logit = (
        -3.2
        + 0.08 * precip
        + 0.05 * rafaga
        - 0.80 * nse
        - 0.02 * (precip * nse)
        - 0.015 * (rafaga * nse)
        + 0.30 * es_cge
        + 0.04 * arbolado
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    corte = (np.random.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame(
        {
            "precip_acumulada_mm": precip,
            "rafaga_max_kmh": rafaga,
            "nse_score": nse,
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

    # A mismo temporal (30 mm de agua y 60 km/h de viento):
    prob_vulnerable = model.predict_probability(
        precip_mm=30.0, rafaga_kmh=60.0, nse_score=-1.5
    )
    prob_acomodado = model.predict_probability(
        precip_mm=30.0, rafaga_kmh=60.0, nse_score=1.5
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
        rafaga_kmh=50.0, nse_score=-1.5
    )
    r50_acomodado = model.compute_critical_rain_threshold(
        rafaga_kmh=50.0, nse_score=1.5
    )

    # La comuna vulnerable debe requerir menos lluvia para colapsar
    assert r50_vulnerable < r50_acomodado


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
