"""Cruce geoespacial e interpolación climática comunal mediante IDW (Inverse Distance Weighting).

Pondera mediciones meteorológicas puntuales de estaciones de referencia hacia los centroides
urbanos de las 52 comunas de la Región Metropolitana, garantizando representatividad física.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from gridbreak_cl.config import load_settings
from gridbreak_cl.etl.historical_seed import get_rm_comunas_metadata

logger = logging.getLogger(__name__)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia geodésica en kilómetros entre dos coordenadas (fórmula de Haversine)."""
    r_tierra = 6371.0  # Radio medio de la Tierra en km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_tierra * c


class SpatialInterpolator:
    """Interpolador geoespacial inverso a la distancia (IDW) para métricas meteorológicas."""

    def __init__(self, power: float | None = None) -> None:
        settings = load_settings()
        self.power = power or float(settings.get("modeling", {}).get("idw_power", 2.0))

    def interpolate_stations_to_comunas(
        self,
        df_stations: pd.DataFrame,
        df_comunas: pd.DataFrame | None = None,
        weather_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Aplica IDW desde las estaciones meteorológicas hacia las comunas de la RM.

        Args:
            df_stations: DataFrame con observaciones de estaciones (debe incluir 'lat', 'lon'
                         y las columnas climáticas a interpolar).
            df_comunas: DataFrame con metadatos comunales (debe incluir 'comuna_id',
                        'comuna_nombre', 'lat', 'lon'). Si es None, usa get_rm_comunas_metadata().
            weather_cols: Columnas a interpolar (por defecto: precipitacion_mm, viento_kmh, rafaga_max_kmh).

        Returns:
            DataFrame comunal con las métricas climáticas interpoladas y la estación más cercana.
        """
        if df_comunas is None:
            df_comunas = get_rm_comunas_metadata()

        if weather_cols is None:
            weather_cols = ["precipitacion_mm", "viento_kmh", "rafaga_max_kmh"]

        # Validar columnas requeridas
        for col in ["lat", "lon"]:
            if col not in df_stations.columns or col not in df_comunas.columns:
                raise ValueError(
                    f"Columna '{col}' requerida tanto en estaciones como en comunas."
                )

        present_weather_cols = [c for c in weather_cols if c in df_stations.columns]
        if not present_weather_cols:
            raise ValueError(
                f"Ninguna de las columnas {weather_cols} se encuentra en df_stations."
            )

        n_comunas = len(df_comunas)
        n_estaciones = len(df_stations)

        # Matriz de distancias (n_comunas x n_estaciones)
        comuna_coords = df_comunas[["lat", "lon"]].to_numpy()
        station_coords = df_stations[["lat", "lon"]].to_numpy()

        dist_matrix = np.zeros((n_comunas, n_estaciones), dtype=float)
        for i in range(n_comunas):
            c_lat, c_lon = comuna_coords[i]
            for j in range(n_estaciones):
                s_lat, s_lon = station_coords[j]
                dist_matrix[i, j] = haversine_distance_km(c_lat, c_lon, s_lat, s_lon)

        # Cálculo de pesos IDW: w_ij = 1 / (d_ij + eps)^p
        eps = 1e-5
        weights = 1.0 / np.power(dist_matrix + eps, self.power)
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        norm_weights = weights / sum_weights

        result_df = df_comunas.copy()

        # Interpolar cada variable meteorológica
        for col in present_weather_cols:
            station_vals = df_stations[col].to_numpy(dtype=float)
            # Producto punto entre pesos normalizados y valores de estaciones
            interpolated = np.dot(norm_weights, station_vals)

            # Mapeo a nombres estándar del pipeline
            out_col = "precip_acumulada_mm" if col == "precipitacion_mm" else col
            result_df[out_col] = np.round(interpolated, 2)

        # Asignar estación de referencia más cercana (Voronoi nearest)
        nearest_idx = np.argmin(dist_matrix, axis=1)
        nearest_dists = np.min(dist_matrix, axis=1)

        station_names = df_stations["estacion_nombre"].to_numpy()
        result_df["estacion_mas_cercana"] = station_names[nearest_idx]
        result_df["distancia_estacion_km"] = np.round(nearest_dists, 2)

        return result_df
