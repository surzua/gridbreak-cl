"""Cliente colector de telemetría e interrupciones de suministro eléctrico de la SEC.

Consume la plataforma de la Superintendencia de Electricidad y Combustibles (SEC),
normaliza identificadores comunales, realiza el cruce con el padrón base regulado de la RM
y valida los esquemas mediante Pydantic (SECCutRecord).
"""

from __future__ import annotations

import json
import logging
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from gridbreak_cl.config import SECCutRecord, get_project_root, load_settings
from gridbreak_cl.etl.historical_seed import get_rm_comunas_metadata

logger = logging.getLogger(__name__)

# Diccionario de equivalencias para normalizar cadenas emitidas por la SEC
COMUNA_NORM_MAP: dict[str, str] = {
    "alhue": "Alhué",
    "buin": "Buin",
    "calera de tango": "Calera de Tango",
    "cerrillos": "Cerrillos",
    "cerro navia": "Cerro Navia",
    "colina": "Colina",
    "conchali": "Conchalí",
    "curacavi": "Curacaví",
    "el bosque": "El Bosque",
    "el monte": "El Monte",
    "estacion central": "Estación Central",
    "huechuraba": "Huechuraba",
    "independencia": "Independencia",
    "isla de maipo": "Isla de Maipo",
    "la cisterna": "La Cisterna",
    "la florida": "La Florida",
    "la granja": "La Granja",
    "la pintana": "La Pintana",
    "la reina": "La Reina",
    "lampa": "Lampa",
    "las condes": "Las Condes",
    "lo barnechea": "Lo Barnechea",
    "lo espejo": "Lo Espejo",
    "lo prado": "Lo Prado",
    "macul": "Macul",
    "maipu": "Maipú",
    "maria pinto": "María Pinto",
    "melipilla": "Melipilla",
    "nunoa": "Ñuñoa",
    "ñuñoa": "Ñuñoa",
    "padre hurtado": "Padre Hurtado",
    "paine": "Paine",
    "pedro aguirre cerda": "Pedro Aguirre Cerda",
    "penaflor": "Peñaflor",
    "peñaflor": "Peñaflor",
    "penalolen": "Peñalolén",
    "peñalolen": "Peñalolén",
    "peñalolén": "Peñalolén",
    "pirque": "Pirque",
    "providencia": "Providencia",
    "pudahuel": "Pudahuel",
    "puente alto": "Puente Alto",
    "quilicura": "Quilicura",
    "quinta normal": "Quinta Normal",
    "recoleta": "Recoleta",
    "renca": "Renca",
    "san bernardo": "San Bernardo",
    "san joaquin": "San Joaquín",
    "san jose de maipo": "San José de Maipo",
    "san miguel": "San Miguel",
    "san pedro": "San Pedro",
    "san ramon": "San Ramón",
    "santiago": "Santiago",
    "talagante": "Talagante",
    "tiltil": "Til Til",
    "til til": "Til Til",
    "vitacura": "Vitacura",
}


def _strip_accents(text: str) -> str:
    """Remueve tildes y caracteres especiales para matching fonético/lexical."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


def normalize_comuna_name(raw_name: str) -> str:
    """Convierte cualquier variante de nombre de comuna de la RM al nombre canónico."""
    cleaned = _strip_accents(raw_name)
    if cleaned in COMUNA_NORM_MAP:
        return COMUNA_NORM_MAP[cleaned]

    raw_lower = raw_name.strip().lower()
    if raw_lower in COMUNA_NORM_MAP:
        return COMUNA_NORM_MAP[raw_lower]

    return raw_name.strip()


class SECCollector:
    """Colector de datos de interrupción de suministro eléctrico de la SEC.

    Consume la API REST/AJAX de la SEC en apps.sec.cl, aplica backoff exponencial,
    persiste snapshots crudos en data/raw/sec/ y cruza con el padrón comunal de la RM.
    """

    def __init__(
        self,
        base_url: str | None = None,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        timeout_seconds: int = 10,
    ) -> None:
        settings = load_settings()
        sources = settings.get("sources", {})
        self.endpoint_por_fecha = sources.get(
            "sec_endpoint_por_fecha",
            "https://apps.sec.cl/INTONLINEv1/ClientesAfectados/GetPorFecha",
        )
        self.endpoint_hora_server = sources.get(
            "sec_endpoint_hora_server",
            "https://apps.sec.cl/INTONLINEv1/ClientesAfectados/GetHoraServer",
        )
        self.endpoint_regional = sources.get(
            "sec_endpoint_clientes_regional",
            "https://apps.sec.cl/INTONLINEv1/ClientesAfectados/GetClientesRegional",
        )
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout_seconds = timeout_seconds
        self.tz = ZoneInfo("America/Santiago")

    def _post_with_retry(
        self, url: str, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Realiza una petición POST con política de reintentos y backoff exponencial."""
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "GridbreakCL/0.1.0 (Research Pipeline RM)",
        }
        delay = 1.0
        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url, json=payload, headers=headers, timeout=self.timeout_seconds
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return [data]
                return []
            except (requests.RequestException, ValueError) as ex:
                last_exception = ex
                logger.warning(
                    "Intento %d/%d fallido para %s: %s. Reintentando en %.1fs...",
                    attempt,
                    self.max_retries,
                    url,
                    ex,
                    delay,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= self.backoff_factor

        logger.error(
            "Agotados los %d reintentos para %s. Error: %s",
            self.max_retries,
            url,
            last_exception,
        )
        raise ConnectionError(
            f"Fallo en conexión con endpoint SEC ({url}) tras {self.max_retries} intentos: {last_exception}"
        )

    def fetch_server_time(self) -> datetime:
        """Obtiene la hora actual registrada en el servidor de la SEC con zona horaria de Chile."""
        try:
            data = self._post_with_retry(self.endpoint_hora_server, {})
            if data and "FECHA" in data[0]:
                fecha_str = data[0]["FECHA"]  # Ej: "07/10/2026 23:05"
                naive_dt = datetime.strptime(fecha_str, "%d/%m/%Y %H:%M")
                return naive_dt.replace(tzinfo=self.tz)
        except Exception as e:
            logger.warning(
                "No se pudo consultar hora del servidor SEC: %s. Usando hora local.", e
            )

        return datetime.now(self.tz)

    def fetch_raw_snapshot(
        self, dt: datetime | None = None
    ) -> tuple[datetime, list[dict[str, Any]]]:
        """Consulta el snapshot de clientes afectados para una hora específica.

        Si dt es None, consulta la hora del servidor actual.
        """
        target_dt = dt or self.fetch_server_time()
        payload = {
            "anho": target_dt.year,
            "mes": target_dt.month,
            "dia": target_dt.day,
            "hora": target_dt.hour,
        }
        data = self._post_with_retry(self.endpoint_por_fecha, payload)
        return target_dt, data

    def save_raw_snapshot(
        self,
        raw_data: list[dict[str, Any]],
        timestamp: datetime,
        output_dir: Path | None = None,
    ) -> Path:
        """Persiste el snapshot crudo en formato JSON incremental en data/raw/sec/."""
        if output_dir is None:
            output_dir = get_project_root() / "data" / "raw" / "sec"

        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"sec_snapshot_{timestamp.strftime('%Y%m%d_%H%M')}.json"
        target_file = output_dir / filename

        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": timestamp.isoformat(),
                    "source": self.endpoint_por_fecha,
                    "records_count": len(raw_data),
                    "data": raw_data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("Snapshot SEC crudo guardado en: %s", target_file)
        return target_file

    def process_rm_snapshot(
        self,
        raw_data: list[dict[str, Any]],
        timestamp: datetime,
    ) -> pd.DataFrame:
        """Procesa y valida los datos de la RM, cruzando con el padrón de 52 comunas."""
        df_meta = get_rm_comunas_metadata()
        rm_data = [
            d
            for d in raw_data
            if "Metropolitana" in str(d.get("NOMBRE_REGION", ""))
            or "RM" in str(d.get("NOMBRE_REGION", ""))
        ]

        # Mapear afectaciones observadas por nombre canónico de comuna
        afectados_por_comuna: dict[str, int] = {}
        for item in rm_data:
            raw_comuna = str(item.get("NOMBRE_COMUNA", ""))
            canonical = normalize_comuna_name(raw_comuna)
            clientes_afectados = int(item.get("CLIENTES_AFECTADOS", 0))
            # Si una comuna aparece más de una vez (ej. por distribuidora), acumular
            afectados_por_comuna[canonical] = (
                afectados_por_comuna.get(canonical, 0) + clientes_afectados
            )

        records: list[SECCutRecord] = []
        rows: list[dict[str, Any]] = []

        for _, meta in df_meta.iterrows():
            c_nombre = str(meta["comuna_nombre"])
            c_id = str(meta["comuna_id"])
            empresa = str(meta["empresa"])
            clientes_totales = int(meta["clientes_totales"])

            # Clientes sin suministro reportados o 0 si no hubo reporte de corte
            clientes_sin_luz = afectados_por_comuna.get(c_nombre, 0)
            # Acotar clientes sin luz al total comunal por consistencia
            clientes_sin_luz = min(clientes_sin_luz, clientes_totales)

            validated_rec = SECCutRecord(
                timestamp=timestamp,
                comuna_id=c_id,
                comuna_nombre=c_nombre,
                empresa=empresa,
                clientes_sin_suministro=clientes_sin_luz,
                clientes_totales=clientes_totales,
            )
            records.append(validated_rec)

            rows.append(
                {
                    "timestamp": validated_rec.timestamp,
                    "comuna_id": validated_rec.comuna_id,
                    "comuna_nombre": validated_rec.comuna_nombre,
                    "empresa": validated_rec.empresa,
                    "clientes_sin_suministro": validated_rec.clientes_sin_suministro,
                    "clientes_totales": validated_rec.clientes_totales,
                    "tasa_afectacion": validated_rec.tasa_afectacion,
                    "es_corte_critico": validated_rec.es_corte_critico,
                }
            )

        df_result = pd.DataFrame(rows)
        return df_result

    def collect_latest_rm(
        self, persist_raw: bool = True
    ) -> tuple[datetime, pd.DataFrame]:
        """Pipeline completo: consulta última hora, guarda crudo y devuelve DataFrame validado."""
        timestamp, raw_data = self.fetch_raw_snapshot()
        if persist_raw:
            self.save_raw_snapshot(raw_data, timestamp)

        df_rm = self.process_rm_snapshot(raw_data, timestamp)
        return timestamp, df_rm
