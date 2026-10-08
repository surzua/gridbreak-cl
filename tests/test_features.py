"""Pruebas unitarias para el módulo de ingeniería de características."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridbreak_cl.features.build_features import (
    FeatureEngineer,
    build_survival_dataset,
    process_and_save_feature_pipeline,
)


@pytest.fixture
def sample_storm_panel() -> pd.DataFrame:
    """Fixture con mini-panel de prueba de 2 comunas por 5 horas."""
    rows = []
    times = pd.date_range("2024-06-20 00:00", periods=5, freq="h")
    for t_idx, t in enumerate(times):
        # Comuna 1: Vulnerable, colapsa en t_idx = 2
        rows.append(
            {
                "evento": "TestStorm",
                "timestamp": t,
                "comuna_id": "13103",
                "comuna_nombre": "Cerro Navia",
                "empresa": "ENEL",
                "nse_score": -1.8,
                "ingreso_autonomo_promedio": 490000,
                "pobreza_multidimensional": 0.28,
                "arbolado_m2_hab": 2.1,
                "red_aerea_km_ratio": 0.91,
                "es_rural": 0,
                "clientes_totales": 48000,
                "clientes_sin_suministro": 3000 if t_idx >= 2 else 500,
                "tasa_afectacion": 0.0625 if t_idx >= 2 else 0.0104,
                "es_corte_critico": 1 if t_idx >= 2 else 0,
                "precip_acumulada_mm": 5.0 * (t_idx + 1),
                "rafaga_max_kmh": 20.0 + 10.0 * t_idx,
            }
        )
        # Comuna 2: Acomodada, nunca colapsa
        rows.append(
            {
                "evento": "TestStorm",
                "timestamp": t,
                "comuna_id": "13132",
                "comuna_nombre": "Vitacura",
                "empresa": "ENEL",
                "nse_score": 2.5,
                "ingreso_autonomo_promedio": 2600000,
                "pobreza_multidimensional": 0.01,
                "arbolado_m2_hab": 16.5,
                "red_aerea_km_ratio": 0.28,
                "es_rural": 0,
                "clientes_totales": 46000,
                "clientes_sin_suministro": 200,
                "tasa_afectacion": 0.0043,
                "es_corte_critico": 0,
                "precip_acumulada_mm": 4.5 * (t_idx + 1),
                "rafaga_max_kmh": 18.0 + 8.0 * t_idx,
            }
        )
    return pd.DataFrame(rows)


def test_rolling_weather_features(sample_storm_panel: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    res = engineer.compute_rolling_weather_features(sample_storm_panel)

    assert "precip_roll_3h" in res.columns
    assert "precip_roll_6h" in res.columns
    assert "rafaga_max_roll_3h" in res.columns
    assert "delta_rafaga_1h" in res.columns
    assert "energia_cinetica_viento" in res.columns
    assert "tercil_nse" in res.columns

    # Verificar cálculo de energía cinética física E = 0.5 * (v/3.6)^2
    v = float(res.loc[0, "rafaga_max_kmh"])
    expected_e = round(0.5 * ((v / 3.6) ** 2), 2)
    assert abs(res.loc[0, "energia_cinetica_viento"] - expected_e) < 0.05


def test_survival_dataset_construction(sample_storm_panel: pd.DataFrame) -> None:
    surv_df = build_survival_dataset(sample_storm_panel, critical_rate_threshold=0.05)

    assert len(surv_df) == 2
    assert "duration" in surv_df.columns
    assert "event" in surv_df.columns

    # Cerro Navia colapsa en t_idx = 2 (hora 3)
    c_navia = surv_df[surv_df["comuna_id"] == "13103"].iloc[0]
    assert c_navia["event"] == 1
    assert c_navia["duration"] == 3.0

    # Vitacura nunca colapsa (censurada por la derecha)
    vitacura = surv_df[surv_df["comuna_id"] == "13132"].iloc[0]
    assert vitacura["event"] == 0
    assert vitacura["duration"] == 5.0


def test_feature_pipeline_io(tmp_path: Path, sample_storm_panel: pd.DataFrame) -> None:
    input_file = tmp_path / "raw.parquet"
    out_enriched = tmp_path / "enriched.parquet"
    out_surv = tmp_path / "survival.parquet"

    sample_storm_panel.to_parquet(input_file, index=False)

    p1, p2 = process_and_save_feature_pipeline(
        input_parquet=input_file,
        output_enriched_path=out_enriched,
        output_survival_path=out_surv,
    )

    assert p1.exists()
    assert p2.exists()
    df_e = pd.read_parquet(p1)
    df_s = pd.read_parquet(p2)
    assert len(df_e) == len(sample_storm_panel)
    assert len(df_s) == 2
