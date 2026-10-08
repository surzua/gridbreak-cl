"""Pipeline orquestador de ingesta dual en vivo para Gridbreak Chile.

Coordina la extracción de telemetría de la SEC, la adquisición meteorológica (DMC / Open-Meteo),
la interpolación geoespacial IDW y la persistencia de datasets consolidados listos para
consumo por modelos y visualizadores.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from gridbreak_cl.config import get_project_root, load_settings
from gridbreak_cl.etl.dmc_client import DMCWeatherClient
from gridbreak_cl.etl.sec_collector import SECCollector
from gridbreak_cl.etl.spatial_join import SpatialInterpolator

logger = logging.getLogger(__name__)


class LivePipeline:
    """Orquestador integral de ingesta en vivo para la Región Metropolitana."""

    def __init__(
        self,
        sec_collector: SECCollector | None = None,
        weather_client: DMCWeatherClient | None = None,
        interpolator: SpatialInterpolator | None = None,
    ) -> None:
        self.sec_collector = sec_collector or SECCollector()
        self.weather_client = weather_client or DMCWeatherClient()
        self.interpolator = interpolator or SpatialInterpolator()
        self.settings = load_settings()

    def run_ingestion(
        self,
        persist_raw: bool = True,
        output_parquet_path: Path | None = None,
        append_to_history: bool = True,
        offline_mock: bool = False,
    ) -> pd.DataFrame:
        """Ejecuta el ciclo completo de ingesta, cruce geoespacial y consolidación.

        Args:
            persist_raw: Si guarda snapshots JSON en data/raw/sec y data/raw/weather.
            output_parquet_path: Ruta destino para el snapshot consolidado.
            append_to_history: Si guarda también en live_telemetry_history.parquet.
            offline_mock: Modo de prueba offline con datos sintéticos calibrados.

        Returns:
            DataFrame con el panel consolidado de las 52 comunas de la RM.
        """
        logger.info("Iniciando pipeline de telemetría en vivo Gridbreak...")

        # 1. Ingesta SEC
        if offline_mock:
            from gridbreak_cl.etl.historical_seed import get_rm_comunas_metadata

            sec_ts = datetime.now(self.sec_collector.tz)
            meta = get_rm_comunas_metadata()
            df_sec = pd.DataFrame(
                {
                    "timestamp": sec_ts,
                    "comuna_id": meta["comuna_id"],
                    "comuna_nombre": meta["comuna_nombre"],
                    "empresa": meta["empresa"],
                    "clientes_sin_suministro": [120] * len(meta),
                    "clientes_totales": meta["clientes_totales"],
                    "tasa_afectacion": [120 / t for t in meta["clientes_totales"]],
                    "es_corte_critico": [False] * len(meta),
                }
            )
        else:
            sec_ts, df_sec = self.sec_collector.collect_latest_rm(
                persist_raw=persist_raw
            )

        logger.info(
            "Telemetría SEC adquirida: %d comunas, %d clientes sin suministro en RM",
            len(df_sec),
            df_sec["clientes_sin_suministro"].sum(),
        )

        # 2. Ingesta Meteorológica
        weather_ts, df_stations = self.weather_client.collect_latest_weather(
            persist_raw=persist_raw, offline_mock=offline_mock
        )
        logger.info(
            "Telemetría meteorológica adquirida: %d estaciones",
            len(df_stations),
        )

        # 3. Interpolación Geoespacial IDW hacia centroides comunales
        df_comunas_clim = self.interpolator.interpolate_stations_to_comunas(
            df_stations=df_stations
        )

        # 4. Cruce (Merge) entre SEC y Clima + Covariables Estructurales
        merged = pd.merge(
            df_sec,
            df_comunas_clim,
            on=["comuna_id", "comuna_nombre", "empresa", "clientes_totales"],
            how="inner",
        )

        # 5. Generación de features analíticas requeridas por los modelos
        merged["rafaga_cuadratica"] = (merged["rafaga_max_kmh"] ** 2) / 100.0
        merged["precip_x_nse"] = (
            merged["precip_acumulada_mm"] * merged["nse_score"]
        )
        merged["rafaga_x_nse"] = (
            merged["rafaga_max_kmh"] * merged["nse_score"]
        )
        merged["es_cge"] = (merged["empresa"] == "CGE").astype(float)

        # 6. Persistencia del snapshot en Parquet
        root = get_project_root()
        if output_parquet_path is None:
            live_path_str = self.settings.get("data_paths", {}).get(
                "live_latest_file", "data/processed/live_latest.parquet"
            )
            output_parquet_path = root / live_path_str

        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_parquet_path, index=False)
        logger.info("Snapshot procesado guardado en: %s", output_parquet_path)

        # 7. Opcional: Historial incremental de telemetría
        if append_to_history:
            history_path = (
                root / "data" / "processed" / "live_telemetry_history.parquet"
            )
            if history_path.exists():
                try:
                    df_existing = pd.read_parquet(history_path)
                    combined = pd.concat(
                        [df_existing, merged], ignore_index=True
                    )
                    # Deduplicar por comuna y timestamp
                    combined = combined.drop_duplicates(
                        subset=["comuna_id", "timestamp"]
                    )
                    combined.to_parquet(history_path, index=False)
                except Exception as e:
                    logger.warning("No se pudo actualizar historial: %s", e)
            else:
                merged.to_parquet(history_path, index=False)

        return merged


def run_live_pipeline() -> pd.DataFrame:
    """Función de conveniencia para ejecutar la ingesta en vivo."""
    pipeline = LivePipeline()
    return pipeline.run_ingestion()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = run_live_pipeline()
    print(f"Comunas consolidadas en vivo: {len(df)}")
    print(df.head())
