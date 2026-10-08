"""Módulo de configuración y esquemas de validación con Pydantic."""

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SECCutRecord(BaseModel):
    """Registro de afectación de suministro eléctrico por comuna."""

    timestamp: datetime
    comuna_id: str
    comuna_nombre: str
    empresa: str
    clientes_sin_suministro: int = Field(ge=0)
    clientes_totales: int = Field(gt=0)

    @property
    def tasa_afectacion(self) -> float:
        """Proporción de clientes desconectados."""
        return self.clientes_sin_suministro / self.clientes_totales

    @property
    def es_corte_critico(self) -> bool:
        """Determina si la afectación supera el 5% comunal."""
        return self.tasa_afectacion >= 0.05


class WeatherRecord(BaseModel):
    """Registro meteorológico de estación de referencia."""

    timestamp: datetime
    estacion_codigo: str
    estacion_nombre: str
    precipitacion_mm: float = Field(ge=0.0)
    viento_kmh: float = Field(ge=0.0)
    rafaga_max_kmh: float = Field(ge=0.0)


class ComunaFeatures(BaseModel):
    """Covariables estructurales y sociodemográficas a nivel comunal."""

    comuna_id: str
    comuna_nombre: str
    empresa_principal: str
    nse_score: float  # Estandarizado (media 0, std 1; mayor = mayor ingreso)
    ingreso_autonomo_promedio: float = Field(ge=0.0)
    pobreza_multidimensional: float = Field(ge=0.0, le=1.0)
    arbolado_m2_hab: float = Field(ge=0.0)
    red_aerea_km_ratio: float = Field(ge=0.0, le=1.0)
    densidad_clientes_km: float = Field(ge=0.0)
    es_rural: bool = False


def get_project_root() -> Path:
    """Retorna la ruta raíz del proyecto."""
    return Path(__file__).resolve().parent.parent.parent


def load_settings(config_path: Path | None = None) -> dict[str, Any]:
    """Carga configuración desde settings.yaml."""
    if config_path is None:
        config_path = get_project_root() / "config" / "settings.yaml"

    if not config_path.exists():
        return {}

    with open(config_path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
        return data
