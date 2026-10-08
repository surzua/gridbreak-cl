"""Generación de mapas interactivos geoespaciales con Folium."""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd

from gridbreak_cl.config import get_project_root


def get_risk_color(prob: float) -> str:
    """Retorna un código de color hexadecimal según la probabilidad de corte crítico."""
    if prob >= 0.50:
        return "#e74c3c"  # Rojo alto riesgo
    elif prob >= 0.25:
        return "#f39c12"  # Naranja riesgo moderado
    else:
        return "#2ecc71"  # Verde baja vulnerabilidad


def generate_folium_vulnerability_map(
    df_sim: pd.DataFrame,
    output_html_path: Path | None = None,
) -> Path:
    """Genera un mapa interactivo Leaflet/Folium con las 52 comunas de la RM y su riesgo estimado."""
    # Coordenadas centrales de Santiago
    m = folium.Map(
        location=[-33.4500, -70.6600],
        zoom_start=10,
        tiles="OpenStreetMap",
    )

    for _, row in df_sim.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        prob = float(row.get("prob_colapso", 0.0))
        nombre = str(row["comuna_nombre"])
        empresa = str(row.get("empresa", "ENEL"))
        clientes = int(row.get("clientes_totales", 0))
        red_aerea = float(row.get("red_aerea_km_ratio", 0.70))

        color = get_risk_color(prob)
        radius = 8 + (prob * 14)

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
            <h4 style="margin: 0 0 5px 0; color: #2c3e50;">{nombre}</h4>
            <hr style="margin: 2px 0 6px 0;">
            <b>Distribuidora:</b> {empresa}<br>
            <b>Prob. Colapso:</b> <span style="color: {color}; font-weight: bold;">{prob:.1%}</span><br>
            <b>Red Aérea Expuesta:</b> {red_aerea:.0%}<br>
            <b>Clientes Totales:</b> {clientes:,}<br>
            <b>NSE Score:</b> {float(row.get('nse_score', 0.0)):+.2f}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            tooltip=f"{nombre}: {prob:.1%} riesgo",
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)

    if output_html_path is None:
        output_html_path = (
            get_project_root()
            / "reports"
            / "maps"
            / "vulnerabilidad_rm_simulada.html"
        )

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html_path))
    return output_html_path


def generate_folium_live_map(
    df_live: pd.DataFrame,
    output_html_path: Path | None = None,
) -> Path:
    """Genera un mapa interactivo con el estado de clientes sin suministro en tiempo real."""
    m = folium.Map(
        location=[-33.4500, -70.6600],
        zoom_start=10,
        tiles="OpenStreetMap",
    )

    for _, row in df_live.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        nombre = str(row["comuna_nombre"])
        sin_luz = int(row["clientes_sin_suministro"])
        tasa = float(row["tasa_afectacion"])
        estacion = str(row.get("estacion_mas_cercana", "DMC"))
        rafaga = float(row.get("rafaga_max_kmh", 0.0))

        color = "#e74c3c" if tasa >= 0.05 else "#f39c12" if tasa >= 0.01 else "#27ae60"
        radius = 6 + min(20, int(sin_luz / 200))

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
            <h4 style="margin: 0 0 5px 0;">{nombre}</h4>
            <hr style="margin: 2px 0 6px 0;">
            <b>Sin Suministro:</b> {sin_luz:,} ({tasa:.2%})<br>
            <b>Estación:</b> {estacion}<br>
            <b>Ráfaga IDW:</b> {rafaga:.1f} km/h<br>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            tooltip=f"{nombre}: {sin_luz:,} sin luz ({tasa:.2%})",
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)

    if output_html_path is None:
        output_html_path = (
            get_project_root() / "reports" / "maps" / "monitoreo_en_vivo_rm.html"
        )

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html_path))
    return output_html_path
