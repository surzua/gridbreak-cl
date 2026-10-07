"""Validación de esquemas Pydantic y generación de datos benchmark."""

from datetime import datetime

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
    assert "es_corte_critico" in df_storm.columns
