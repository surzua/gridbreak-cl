"""Módulo de generación y persistencia de datasets benchmark para la RM (Temporales 2024)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from gridbreak_cl.config import get_project_root


def get_rm_comunas_metadata() -> pd.DataFrame:
    """Retorna metadatos base y estructurales de comunas representativas de la RM."""
    comunas_data = [
        # Comuna, CUT, Empresa, Lat, Lon, Clientes, NSE_score, Ingreso, Pobreza, Arbolado_m2
        (
            "Santiago",
            "13101",
            "ENEL",
            -33.4489,
            -70.6693,
            240000,
            0.7,
            1250000,
            0.08,
            4.2,
        ),
        (
            "Cerrillos",
            "13102",
            "ENEL",
            -33.5000,
            -70.7167,
            45000,
            -0.4,
            680000,
            0.17,
            3.8,
        ),
        (
            "Cerro Navia",
            "13103",
            "ENEL",
            -33.4242,
            -70.7381,
            48000,
            -1.8,
            490000,
            0.28,
            2.1,
        ),
        (
            "Conchalí",
            "13104",
            "ENEL",
            -33.3833,
            -70.6833,
            52000,
            -0.7,
            610000,
            0.21,
            3.1,
        ),
        (
            "El Bosque",
            "13105",
            "ENEL",
            -33.5667,
            -70.6833,
            62000,
            -1.1,
            560000,
            0.23,
            2.6,
        ),
        (
            "Estación Central",
            "13106",
            "ENEL",
            -33.4619,
            -70.7022,
            75000,
            -0.3,
            710000,
            0.16,
            2.9,
        ),
        (
            "Huechuraba",
            "13107",
            "ENEL",
            -33.3667,
            -70.6500,
            39000,
            0.2,
            890000,
            0.14,
            5.8,
        ),
        (
            "Independencia",
            "13108",
            "ENEL",
            -33.4167,
            -70.6667,
            51000,
            -0.2,
            730000,
            0.15,
            3.3,
        ),
        (
            "La Cisterna",
            "13109",
            "ENEL",
            -33.5333,
            -70.6667,
            38000,
            -0.3,
            720000,
            0.16,
            3.5,
        ),
        (
            "La Florida",
            "13110",
            "ENEL",
            -33.5333,
            -70.5833,
            145000,
            0.1,
            820000,
            0.13,
            5.2,
        ),
        (
            "La Granja",
            "13111",
            "ENEL",
            -33.5333,
            -70.6167,
            46000,
            -1.3,
            530000,
            0.25,
            2.4,
        ),
        (
            "La Pintana",
            "13112",
            "ENEL",
            -33.5833,
            -70.6333,
            61000,
            -2.1,
            440000,
            0.31,
            1.8,
        ),
        (
            "La Reina",
            "13113",
            "ENEL",
            -33.4500,
            -70.5333,
            41000,
            1.7,
            1850000,
            0.04,
            11.4,
        ),
        (
            "Las Condes",
            "13114",
            "ENEL",
            -33.4000,
            -70.5500,
            155000,
            2.3,
            2300000,
            0.02,
            14.8,
        ),
        (
            "Lo Barnechea",
            "13115",
            "ENEL",
            -33.3500,
            -70.5000,
            43000,
            2.1,
            2150000,
            0.05,
            13.2,
        ),
        (
            "Lo Espejo",
            "13116",
            "ENEL",
            -33.5167,
            -70.7000,
            37000,
            -1.9,
            470000,
            0.29,
            1.9,
        ),
        (
            "Lo Prado",
            "13117",
            "ENEL",
            -33.4442,
            -70.7253,
            40000,
            -1.2,
            550000,
            0.24,
            2.7,
        ),
        ("Macul", "13118", "ENEL", -33.4833, -70.6000, 49000, 0.3, 870000, 0.12, 4.9),
        ("Maipú", "13119", "ENEL", -33.5167, -70.7667, 185000, 0.0, 790000, 0.14, 4.5),
        ("Ñuñoa", "13120", "ENEL", -33.4500, -70.6000, 110000, 1.5, 1650000, 0.05, 8.6),
        (
            "Pedro Aguirre Cerda",
            "13121",
            "ENEL",
            -33.4833,
            -70.6667,
            41000,
            -1.1,
            570000,
            0.22,
            2.8,
        ),
        (
            "Peñalolén",
            "13122",
            "ENEL",
            -33.4833,
            -70.5333,
            82000,
            -0.1,
            750000,
            0.16,
            6.1,
        ),
        (
            "Providencia",
            "13123",
            "ENEL",
            -33.4333,
            -70.6167,
            98000,
            2.0,
            2050000,
            0.03,
            12.3,
        ),
        (
            "Pudahuel",
            "13124",
            "ENEL",
            -33.4333,
            -70.7833,
            82000,
            -0.8,
            620000,
            0.20,
            3.4,
        ),
        (
            "Quilicura",
            "13125",
            "ENEL",
            -33.3667,
            -70.7333,
            76000,
            -0.4,
            690000,
            0.17,
            3.9,
        ),
        (
            "Quinta Normal",
            "13126",
            "ENEL",
            -33.4333,
            -70.7000,
            47000,
            -0.9,
            590000,
            0.21,
            3.7,
        ),
        (
            "Recoleta",
            "13127",
            "ENEL",
            -33.4000,
            -70.6333,
            63000,
            -0.6,
            640000,
            0.19,
            4.0,
        ),
        ("Renca", "13128", "ENEL", -33.4000, -70.7333, 53000, -1.3, 540000, 0.24, 2.5),
        (
            "San Joaquín",
            "13129",
            "ENEL",
            -33.5000,
            -70.6333,
            40000,
            -0.6,
            650000,
            0.18,
            3.6,
        ),
        (
            "San Miguel",
            "13130",
            "ENEL",
            -33.5000,
            -70.6500,
            58000,
            0.4,
            910000,
            0.11,
            4.8,
        ),
        (
            "San Ramón",
            "13131",
            "ENEL",
            -33.5333,
            -70.6500,
            34000,
            -1.7,
            500000,
            0.27,
            2.0,
        ),
        (
            "Vitacura",
            "13132",
            "ENEL",
            -33.3833,
            -70.5833,
            46000,
            2.5,
            2600000,
            0.01,
            16.5,
        ),
        (
            "Puente Alto",
            "13201",
            "CGE",
            -33.6167,
            -70.5833,
            210000,
            -0.5,
            670000,
            0.18,
            3.7,
        ),
        ("Pirque", "13202", "CGE", -33.6667, -70.5667, 12000, -0.2, 740000, 0.15, 7.5),
        (
            "San José de Maipo",
            "13203",
            "CGE",
            -33.6333,
            -70.3667,
            9000,
            -0.4,
            700000,
            0.16,
            9.8,
        ),
        (
            "San Bernardo",
            "13401",
            "CGE",
            -33.6000,
            -70.7000,
            115000,
            -0.8,
            630000,
            0.20,
            3.2,
        ),
        ("Buin", "13402", "CGE", -33.7333, -70.7333, 38000, -0.6, 650000, 0.19, 4.1),
        ("Paine", "13404", "CGE", -33.8167, -70.7500, 32000, -0.9, 600000, 0.22, 4.3),
        ("Colina", "13301", "ENEL", -33.2000, -70.6833, 49000, 0.3, 930000, 0.13, 6.2),
        ("Lampa", "13302", "ENEL", -33.2833, -70.8833, 36000, -0.7, 620000, 0.21, 3.5),
        ("Til Til", "13303", "ENEL", -33.0833, -70.9333, 8500, -1.0, 580000, 0.23, 3.8),
        (
            "Talagante",
            "13601",
            "CGE",
            -33.6667,
            -70.9333,
            31000,
            -0.5,
            660000,
            0.18,
            4.2,
        ),
        (
            "Peñaflor",
            "13605",
            "CGE",
            -33.6000,
            -70.8833,
            35000,
            -0.4,
            680000,
            0.17,
            3.9,
        ),
        (
            "Melipilla",
            "13501",
            "CGE",
            -33.7000,
            -71.2167,
            45000,
            -1.0,
            570000,
            0.23,
            3.6,
        ),
    ]

    cols = [
        "comuna_nombre",
        "comuna_id",
        "empresa",
        "lat",
        "lon",
        "clientes_totales",
        "nse_score",
        "ingreso_autonomo_promedio",
        "pobreza_multidimensional",
        "arbolado_m2_hab",
    ]
    return pd.DataFrame(comunas_data, columns=cols)


def generate_benchmark_storm_dataset(seed: int = 42) -> pd.DataFrame:
    """Genera dataset panel horario de los dos temporales emblemáticos de 2024 en la RM.

    Evento 1: Junio 2024 (lluvias intensas de 40 a 75 mm, vientos de 30 a 50 km/h)
    Evento 2: Agosto 2024 (viento extremo con ráfagas >100-120 km/h, lluvia de 15 a 35 mm)
    """
    rng = np.random.default_rng(seed)
    comunas = get_rm_comunas_metadata()
    records: list[dict[str, Any]] = []

    # Configuración de los dos temporales
    events: list[dict[str, Any]] = [
        {
            "nombre": "Temporal_Junio_2024",
            "start": datetime(2024, 6, 20, 0, 0),
            "hours": 36,
            "rain_peak_mm": 65.0,
            "wind_peak_kmh": 45.0,
            "type": "rain_dominant",
        },
        {
            "nombre": "Temporal_Agosto_2024",
            "start": datetime(2024, 8, 1, 18, 0),
            "hours": 30,
            "rain_peak_mm": 28.0,
            "wind_peak_kmh": 115.0,
            "type": "wind_dominant",
        },
    ]

    for event in events:
        start_time = cast(datetime, event["start"])
        total_hours = int(event["hours"])

        for h in range(total_hours):
            current_time = start_time + timedelta(hours=h)
            progress = np.sin((h / total_hours) * np.pi) ** 1.5

            for _, com in comunas.iterrows():
                # Variabilidad meteorológica local según posición geográfica
                dist_factor = 1.0 + 0.15 * (com["lat"] - (-33.45))
                precip_acum = max(
                    0.0,
                    event["rain_peak_mm"] * progress * dist_factor + rng.normal(0, 2),
                )
                rafaga_max = max(
                    10.0,
                    event["wind_peak_kmh"] * progress * (1.0 + rng.normal(0, 0.08)),
                )

                # Fragilidad y respuesta de red según vulnerabilidad comunal
                # Menor NSE -> mayor fragilidad ante lluvia y ráfagas
                es_cge = 1.0 if com["empresa"] == "CGE" else 0.0
                logit = (
                    -3.8
                    + 0.075 * precip_acum
                    + 0.052 * rafaga_max
                    - 0.85 * com["nse_score"]
                    - 0.022 * (precip_acum * com["nse_score"])
                    - 0.016 * (rafaga_max * com["nse_score"])
                    + 0.35 * es_cge
                    + 0.03 * com["arbolado_m2_hab"]
                    + rng.normal(0, 0.3)
                )

                prob_corte = 1.0 / (1.0 + np.exp(-logit))

                # Estimación de clientes sin suministro
                tasa_afectacion = float(
                    np.clip(prob_corte * rng.uniform(0.08, 0.28), 0.0, 0.95)
                )
                clientes_sin_suministro = int(tasa_afectacion * com["clientes_totales"])
                es_corte_critico = int(tasa_afectacion >= 0.05)

                records.append(
                    {
                        "evento": event["nombre"],
                        "timestamp": current_time,
                        "comuna_id": com["comuna_id"],
                        "comuna_nombre": com["comuna_nombre"],
                        "empresa": com["empresa"],
                        "lat": com["lat"],
                        "lon": com["lon"],
                        "nse_score": com["nse_score"],
                        "ingreso_autonomo_promedio": com["ingreso_autonomo_promedio"],
                        "pobreza_multidimensional": com["pobreza_multidimensional"],
                        "arbolado_m2_hab": com["arbolado_m2_hab"],
                        "clientes_totales": com["clientes_totales"],
                        "clientes_sin_suministro": clientes_sin_suministro,
                        "tasa_afectacion": tasa_afectacion,
                        "es_corte_critico": es_corte_critico,
                        "precip_acumulada_mm": round(precip_acum, 1),
                        "rafaga_max_kmh": round(rafaga_max, 1),
                    }
                )

    df = pd.DataFrame(records)
    return df


def save_benchmark_dataset(target_path: Path | None = None) -> Path:
    """Genera y guarda el dataset de referencia en formato Parquet."""
    if target_path is None:
        target_path = (
            get_project_root()
            / "data"
            / "processed"
            / "benchmark_temporales_2024.parquet"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_benchmark_storm_dataset()
    df.to_parquet(target_path, index=False)
    return target_path
