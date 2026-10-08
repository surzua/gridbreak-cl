"""Pruebas unitarias para los módulos de visualización y generación de mapas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridbreak_cl.etl.historical_seed import get_rm_comunas_metadata
from gridbreak_cl.models.fragility_curves import FragilityModel
from gridbreak_cl.visualization.map_generator import (
    generate_folium_live_map,
    generate_folium_vulnerability_map,
    get_risk_color,
)
from gridbreak_cl.visualization.static_charts import (
    generate_all_portfolio_figures,
    plot_critical_thresholds_gap,
    plot_fragility_curves_comparison,
)


@pytest.fixture
def sample_comunas_df() -> pd.DataFrame:
    df = get_rm_comunas_metadata().head(5).copy()
    df["prob_colapso"] = [0.75, 0.40, 0.10, 0.60, 0.20]
    df["clientes_sin_suministro"] = [1000, 200, 50, 400, 30]
    df["tasa_afectacion"] = [0.06, 0.015, 0.003, 0.025, 0.002]
    df["rafaga_max_kmh"] = [65.0, 50.0, 35.0, 55.0, 40.0]
    df["estacion_mas_cercana"] = ["Quinta Normal"] * 5
    return df


def test_get_risk_color() -> None:
    assert get_risk_color(0.60) == "#e74c3c"
    assert get_risk_color(0.35) == "#f39c12"
    assert get_risk_color(0.15) == "#2ecc71"


def test_folium_maps_generation(tmp_path: Path, sample_comunas_df: pd.DataFrame) -> None:
    map_vuln = tmp_path / "vuln.html"
    map_live = tmp_path / "live.html"

    out1 = generate_folium_vulnerability_map(sample_comunas_df, output_html_path=map_vuln)
    out2 = generate_folium_live_map(sample_comunas_df, output_html_path=map_live)

    assert out1.exists()
    assert out2.exists()
    assert out1.stat().st_size > 500
    assert out2.stat().st_size > 500


def test_static_charts_generation(tmp_path: Path, sample_comunas_df: pd.DataFrame) -> None:
    # Generar modelos sintéticos para probar plots
    frag_model = FragilityModel()
    # Generar mini panel sintético
    from gridbreak_cl.etl.historical_seed import generate_benchmark_storm_dataset

    benchmark_df = generate_benchmark_storm_dataset()
    frag_model.fit(benchmark_df)

    p_frag = plot_fragility_curves_comparison(
        frag_model, output_path=tmp_path / "frag.png"
    )
    assert p_frag.exists()
    assert p_frag.stat().st_size > 1000

    p_thresh = plot_critical_thresholds_gap(
        frag_model, sample_comunas_df, output_path=tmp_path / "thresh.png"
    )
    assert p_thresh.exists()
    assert p_thresh.stat().st_size > 1000


def test_generate_all_figures(tmp_path: Path) -> None:
    figs = generate_all_portfolio_figures(output_dir=tmp_path)
    assert len(figs) == 4
    for f in figs:
        assert f.exists()
        assert f.stat().st_size > 1000
