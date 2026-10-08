"""Cliente de telemetría meteorológica para la Región Metropolitana (DMC / Open-Meteo).

Obtiene mediciones horarias de precipitación acumulada, viento medio y ráfagas máximas
para la red de estaciones meteorológicas de referencia en la RM, con soporte dual
(DMC Oficial autenticado + Open-Meteo público resiliente sin credenciales).
Valida las observaciones con el esquema WeatherRecord de Pydantic.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from gridbreak_cl.config import WeatherRecord, get_project_root, load_settings

logger = logging.getLogger(__name__)

# Estaciones meteorológicas de referencia calibradas para la RM
DEFAULT_REFERENCE_STATIONS: list[dict[str, Any]] = [
    {
        "codigo": "330020",
        "nombre": "Quinta Normal",
        "lat": -33.4442,
        "lon": -70.6828,
    },
    {
        "codigo": "330019",
        "nombre": "Tobalaba",
        "lat": -33.4561,
        "lon": -70.5503,
    },
    {
        "codigo": "330030",
        "nombre": "Pudahuel",
        "lat": -33.3931,
        "lon": -70.7858,
    },
    {
        "codigo": "330056",
        "nombre": "La Florida",
        "lat": -33.5228,
        "lon": -70.5847,
    },
    {
        "codigo": "330067",
        "nombre": "Talagante",
        "lat": -33.6653,
        "lon": -70.9328,
    },
]


class DMCWeatherClient:
    """Cliente meteorológico para la RM con arquitectura de fallback y reintentos.

    Soporta:
    1. DMC Oficial (si existen DMC_USUARIO y DMC_TOKEN en entorno).
    2. Open-Meteo API pública en vivo (sin credenciales, alta precisión horaria para Chile).
    3. Modo Mock/Offline para pruebas deterministas y testing CI/CD.
    """

    def __init__(
        self,
        stations: list[dict[str, Any]] | None = None,
        timeout_seconds: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        settings = load_settings()
        sources = settings.get("sources", {})

        self.stations = (
            stations
            or sources.get("dmc_reference_stations")
            or DEFAULT_REFERENCE_STATIONS
        )
        self.open_meteo_url = sources.get(
            "open_meteo_url", "https://api.open-meteo.com/v1/forecast"
        )
        self.dmc_api_url = sources.get(
            "dmc_api_url",
            "https://climatologia.meteochile.gob.cl/application/servicios/getDatosRecientesRedEma",
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.tz = ZoneInfo("America/Santiago")

        # Credenciales opcionales de DMC
        self.dmc_user = os.getenv("DMC_USUARIO") or sources.get("dmc_usuario")
        self.dmc_token = os.getenv("DMC_TOKEN") or sources.get("dmc_token")

    def _fetch_from_open_meteo(
        self, lat: float, lon: float
    ) -> tuple[float, float, float]:
        """Consulta variables meteorológicas en vivo para una coordenada dada en Chile."""
        params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "current": "precipitation,wind_speed_10m,wind_gusts_10m",
            "timezone": "America/Santiago",
        }
        delay = 1.0
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    self.open_meteo_url,
                    params=params,
                    timeout=self.timeout_seconds,
                    headers={"User-Agent": "GridbreakCL/0.1.0"},
                )
                resp.raise_for_status()
                data = resp.json()
                current = data.get("current", {})

                precip = float(current.get("precipitation", 0.0) or 0.0)
                wind = float(current.get("wind_speed_10m", 0.0) or 0.0)
                gust = float(current.get("wind_gusts_10m", 0.0) or wind * 1.3)

                return max(0.0, precip), max(0.0, wind), max(0.0, gust)
            except Exception as e:
                last_err = e
                logger.warning(
                    "Intento %d/%d fallido para estación en (%.4f, %.4f): %s",
                    attempt,
                    self.max_retries,
                    lat,
                    lon,
                    e,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= self.backoff_factor

        raise ConnectionError(
            f"Fallo al consultar Open-Meteo en ({lat}, {lon}): {last_err}"
        )

    def _fetch_from_dmc_official(self) -> dict[str, dict[str, float]]:
        """Consulta la API oficial de DMC si existen credenciales válidas."""
        if not self.dmc_user or not self.dmc_token:
            return {}

        params = {"usuario": self.dmc_user, "token": self.dmc_token}
        resp = requests.get(
            self.dmc_api_url, params=params, timeout=self.timeout_seconds
        )
        resp.raise_for_status()
        data = resp.json()

        # Parsear estaciones según formato de datos de la DMC
        parsed: dict[str, dict[str, float]] = {}
        for item in data.get("datosEstaciones", []):
            est_meta = item.get("estacion", {})
            codigo = str(est_meta.get("codigoNacional", ""))
            # Extraer mediciones de precipitación y viento si existen
            mediciones = item.get("mediciones", {})
            parsed[codigo] = {
                "precip": float(mediciones.get("aguaCaida", 0.0) or 0.0),
                "wind": float(mediciones.get("viento", 0.0) or 0.0),
                "gust": float(mediciones.get("rafaga", 0.0) or 0.0),
            }
        return parsed

    def fetch_station_observations(
        self,
        target_timestamp: datetime | None = None,
        offline_mock: bool = False,
    ) -> tuple[datetime, pd.DataFrame]:
        """Obtiene las observaciones de las estaciones meteorológicas de referencia de la RM.

        Retorna un DataFrame validado con precipitación, viento y ráfagas por estación.
        """
        now_dt = target_timestamp or datetime.now(self.tz)
        rows: list[dict[str, Any]] = []

        # 1. Intentar DMC oficial si hay credenciales
        dmc_data: dict[str, dict[str, float]] = {}
        if not offline_mock and self.dmc_user and self.dmc_token:
            try:
                dmc_data = self._fetch_from_dmc_official()
                logger.info("Obtenidas %d estaciones de DMC oficial", len(dmc_data))
            except Exception as e:
                logger.warning(
                    "Fallo al consultar DMC oficial (%s). Recurriendo a fallback Open-Meteo.",
                    e,
                )

        # 2. Iterar sobre las estaciones de referencia
        for st in self.stations:
            codigo = str(st["codigo"])
            nombre = str(st["nombre"])
            lat = float(st["lat"])
            lon = float(st["lon"])

            if offline_mock:
                # Mediciones sintéticas base para testing offline
                precip = 5.0
                wind = 25.0
                gust = 45.0
            elif codigo in dmc_data:
                precip = dmc_data[codigo]["precip"]
                wind = dmc_data[codigo]["wind"]
                gust = dmc_data[codigo]["gust"]
            else:
                # Fallback en vivo con Open-Meteo
                precip, wind, gust = self._fetch_from_open_meteo(lat, lon)

            # Validar con Pydantic WeatherRecord
            validated = WeatherRecord(
                timestamp=now_dt,
                estacion_codigo=codigo,
                estacion_nombre=nombre,
                precipitacion_mm=precip,
                viento_kmh=wind,
                rafaga_max_kmh=gust,
            )

            rows.append(
                {
                    "timestamp": validated.timestamp,
                    "estacion_codigo": validated.estacion_codigo,
                    "estacion_nombre": validated.estacion_nombre,
                    "lat": lat,
                    "lon": lon,
                    "precipitacion_mm": validated.precipitacion_mm,
                    "viento_kmh": validated.viento_kmh,
                    "rafaga_max_kmh": validated.rafaga_max_kmh,
                }
            )

        df_obs = pd.DataFrame(rows)
        return now_dt, df_obs

    def save_raw_snapshot(
        self,
        df_obs: pd.DataFrame,
        timestamp: datetime,
        output_dir: Path | None = None,
    ) -> Path:
        """Persiste el snapshot meteorológico en data/raw/weather/."""
        if output_dir is None:
            output_dir = get_project_root() / "data" / "raw" / "weather"

        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"weather_snapshot_{timestamp.strftime('%Y%m%d_%H%M')}.json"
        target_file = output_dir / filename

        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": timestamp.isoformat(),
                    "stations_count": len(df_obs),
                    "observations": df_obs.to_dict(orient="records"),
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        logger.info("Snapshot meteorológico guardado en: %s", target_file)
        return target_file

    def collect_latest_weather(
        self, persist_raw: bool = True, offline_mock: bool = False
    ) -> tuple[datetime, pd.DataFrame]:
        """Pipeline completo de telemetría meteorológica."""
        timestamp, df_obs = self.fetch_station_observations(offline_mock=offline_mock)
        if persist_raw:
            self.save_raw_snapshot(df_obs, timestamp)
        return timestamp, df_obs
