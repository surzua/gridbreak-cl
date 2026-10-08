"""Validación de esquemas Pydantic y generación de datos benchmark."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridbreak_cl.config import SECCutRecord
from gridbreak_cl.etl.historical_seed import (
    generate_benchmark_storm_dataset,
    get_rm_comunas_metadata,
)


def test_sec_record_validation() -> None:
    rec = SECCutRecord(
        timestamp=datetime.now(),
        comuna_id="13101",
        comuna_nombre="Santiago",
        empresa="ENEL",
        clientes_sin_suministro=12000,
        clientes_totales=240000,
    )
    assert rec.tasa_afectacion == pytest.approx(0.05)
    assert rec.es_corte_critico is True


def test_rm_comunas_metadata() -> None:
    df_meta = get_rm_comunas_metadata()
    assert len(df_meta) >= 40
    assert "comuna_nombre" in df_meta.columns
    assert "nse_score" in df_meta.columns
    assert "empresa" in df_meta.columns
    assert "red_aerea_km_ratio" in df_meta.columns
    assert "es_rural" in df_meta.columns

    # Validar que los ratios de red aérea estén en rango físico [0, 1]
    assert df_meta["red_aerea_km_ratio"].between(0.0, 1.0).all()

    # Validar que existan comunas de ENEL y CGE
    empresas = set(df_meta["empresa"].unique())
    assert "ENEL" in empresas
    assert "CGE" in empresas


def test_generate_benchmark_storm_dataset() -> None:
    df_storm = generate_benchmark_storm_dataset()
    assert isinstance(df_storm, pd.DataFrame)
    assert len(df_storm) > 1000
    assert "evento" in df_storm.columns
    assert "precip_acumulada_mm" in df_storm.columns
    assert "rafaga_max_kmh" in df_storm.columns
    assert "rafaga_cuadratica" in df_storm.columns
    assert "red_aerea_km_ratio" in df_storm.columns
    assert "es_corte_critico" in df_storm.columns


def test_normalize_comuna_name() -> None:
    from gridbreak_cl.etl.sec_collector import normalize_comuna_name

    assert normalize_comuna_name("alhue") == "Alhué"
    assert normalize_comuna_name("Conchali") == "Conchalí"
    assert normalize_comuna_name("ESTACION CENTRAL") == "Estación Central"
    assert normalize_comuna_name("Maipu") == "Maipú"
    assert normalize_comuna_name("San Jose de Maipo") == "San José de Maipo"
    assert normalize_comuna_name("Tiltil") == "Til Til"
    assert normalize_comuna_name("Til Til") == "Til Til"
    assert normalize_comuna_name("Ñuñoa") == "Ñuñoa"
    assert normalize_comuna_name("peñalolen") == "Peñalolén"


def test_sec_collector_process_rm_snapshot() -> None:
    from gridbreak_cl.etl.sec_collector import SECCollector

    collector = SECCollector()
    ts = datetime(2026, 10, 7, 22, 0)
    mock_raw = [
        {
            "NOMBRE_REGION": "Metropolitana",
            "NOMBRE_COMUNA": "Cerro Navia",
            "CLIENTES_AFECTADOS": 5000,
        },
        {
            "NOMBRE_REGION": "Metropolitana",
            "NOMBRE_COMUNA": "Las Condes",
            "CLIENTES_AFECTADOS": 10,
        },
        {
            "NOMBRE_REGION": "Valparaíso",
            "NOMBRE_COMUNA": "Valparaíso",
            "CLIENTES_AFECTADOS": 800,
        },
    ]

    df_rm = collector.process_rm_snapshot(mock_raw, ts)
    assert len(df_rm) == 52  # Todas las 52 comunas de la RM presentes
    assert "clientes_sin_suministro" in df_rm.columns
    assert "tasa_afectacion" in df_rm.columns
    assert "es_corte_critico" in df_rm.columns

    # Cerro Navia tiene 48.000 clientes, 5.000 afectados -> > 10% -> es_corte_critico = True
    c_navia = df_rm[df_rm["comuna_nombre"] == "Cerro Navia"].iloc[0]
    assert c_navia["clientes_sin_suministro"] == 5000
    assert bool(c_navia["es_corte_critico"]) is True

    # Las Condes tiene 155.000 clientes, 10 afectados -> no crítico
    l_condes = df_rm[df_rm["comuna_nombre"] == "Las Condes"].iloc[0]
    assert l_condes["clientes_sin_suministro"] == 10
    assert bool(l_condes["es_corte_critico"]) is False

    # Comuna no reportada en el mock debe tener 0 clientes afectados
    prov = df_rm[df_rm["comuna_nombre"] == "Providencia"].iloc[0]
    assert prov["clientes_sin_suministro"] == 0
    assert bool(prov["es_corte_critico"]) is False


def test_dmc_client_offline_mock() -> None:
    from gridbreak_cl.etl.dmc_client import DMCWeatherClient

    client = DMCWeatherClient()
    ts, df_obs = client.collect_latest_weather(persist_raw=False, offline_mock=True)
    assert len(df_obs) == 5
    assert (df_obs["precipitacion_mm"] == 5.0).all()
    assert (df_obs["viento_kmh"] == 25.0).all()
    assert (df_obs["rafaga_max_kmh"] == 45.0).all()


def test_spatial_interpolator_properties() -> None:
    from gridbreak_cl.etl.spatial_join import (
        SpatialInterpolator,
        haversine_distance_km,
    )

    # 1. Distancia de Haversine
    dist = haversine_distance_km(-33.4442, -70.6828, -33.4442, -70.6828)
    assert dist == pytest.approx(0.0)

    dist_qn_pud = haversine_distance_km(-33.4442, -70.6828, -33.3931, -70.7858)
    assert 10.0 < dist_qn_pud < 15.0  # ~11 km

    # 2. Conservación de campo homogéneo (si todas las estaciones marcan 20 mm, toda comuna recibe 20 mm)
    df_stations = pd.DataFrame(
        [
            {
                "estacion_codigo": "ST1",
                "estacion_nombre": "Estacion 1",
                "lat": -33.4,
                "lon": -70.6,
                "precipitacion_mm": 20.0,
                "viento_kmh": 30.0,
                "rafaga_max_kmh": 50.0,
            },
            {
                "estacion_codigo": "ST2",
                "estacion_nombre": "Estacion 2",
                "lat": -33.6,
                "lon": -70.8,
                "precipitacion_mm": 20.0,
                "viento_kmh": 30.0,
                "rafaga_max_kmh": 50.0,
            },
        ]
    )

    interpolator = SpatialInterpolator(power=2.0)
    df_res = interpolator.interpolate_stations_to_comunas(df_stations)
    assert len(df_res) == 52
    assert np.allclose(df_res["precip_acumulada_mm"].to_numpy(), 20.0)
    assert np.allclose(df_res["viento_kmh"].to_numpy(), 30.0)
    assert np.allclose(df_res["rafaga_max_kmh"].to_numpy(), 50.0)
    assert "estacion_mas_cercana" in df_res.columns
    assert "distancia_estacion_km" in df_res.columns


def test_live_pipeline_end_to_end_offline(tmp_path: Path) -> None:
    from gridbreak_cl.etl.live_pipeline import LivePipeline

    pipeline = LivePipeline()
    out_file = tmp_path / "test_live_latest.parquet"

    df_live = pipeline.run_ingestion(
        persist_raw=False,
        output_parquet_path=out_file,
        append_to_history=False,
        offline_mock=True,
    )

    assert out_file.exists()
    assert len(df_live) == 52
    assert "tasa_afectacion" in df_live.columns
    assert "precip_acumulada_mm" in df_live.columns
    assert "rafaga_max_kmh" in df_live.columns
    assert "rafaga_cuadratica" in df_live.columns
    assert "precip_x_nse" in df_live.columns
    assert "es_cge" in df_live.columns


def test_sec_collector_retry_failure() -> None:
    from unittest.mock import patch

    import requests

    from gridbreak_cl.etl.sec_collector import SECCollector

    collector = SECCollector(max_retries=2, timeout_seconds=1)
    with patch("requests.post", side_effect=requests.RequestException("Timeout")):
        with pytest.raises(ConnectionError, match="Fallo en conexión con endpoint SEC"):
            collector.fetch_raw_snapshot(datetime(2026, 10, 7, 22, 0))
