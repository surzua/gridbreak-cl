"""Pipeline de ingeniería de características temporales, físicas y socioeconómicas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gridbreak_cl.config import get_project_root


class FeatureEngineer:
    """Transformador de características temporales y de estrés físico para series de tiempo comunales."""

    def __init__(self) -> None:
        pass

    def compute_rolling_weather_features(
        self,
        df: pd.DataFrame,
        group_cols: list[str] | None = None,
        time_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Calcula ventanas móviles de lluvia y ráfagas agrupadas por evento y comuna.

        Genera:
        - precip_roll_3h: Lluvia acumulada en ventana móvil de 3 horas.
        - precip_roll_6h: Lluvia acumulada en ventana móvil de 6 horas.
        - precip_roll_12h: Lluvia acumulada en ventana móvil de 12 horas.
        - rafaga_max_roll_3h: Máxima ráfaga en ventana móvil de 3 horas.
        - rafaga_max_roll_6h: Máxima ráfaga en ventana móvil de 6 horas.
        - delta_rafaga_1h: Variación horaria de ráfagas (aceleración del viento).
        - energia_cinetica_viento: Presión dinámica proporcional a la energía cinética (J/kg).
        """
        data = df.copy()
        if group_cols is None:
            if "evento" in data.columns and "comuna_id" in data.columns:
                group_cols = ["evento", "comuna_id"]
            elif "comuna_id" in data.columns:
                group_cols = ["comuna_id"]
            else:
                group_cols = []

        data[time_col] = pd.to_datetime(data[time_col])
        if group_cols:
            data = data.sort_values(group_cols + [time_col]).reset_index(drop=True)
        else:
            data = data.sort_values(time_col).reset_index(drop=True)

        if group_cols:
            grouped = data.groupby(group_cols)
            data["precip_roll_3h"] = grouped["precip_acumulada_mm"].transform(
                lambda s: s.rolling(3, min_periods=1).sum()
            )
            data["precip_roll_6h"] = grouped["precip_acumulada_mm"].transform(
                lambda s: s.rolling(6, min_periods=1).sum()
            )
            data["precip_roll_12h"] = grouped["precip_acumulada_mm"].transform(
                lambda s: s.rolling(12, min_periods=1).sum()
            )
            data["rafaga_max_roll_3h"] = grouped["rafaga_max_kmh"].transform(
                lambda s: s.rolling(3, min_periods=1).max()
            )
            data["rafaga_max_roll_6h"] = grouped["rafaga_max_kmh"].transform(
                lambda s: s.rolling(6, min_periods=1).max()
            )
            data["delta_rafaga_1h"] = grouped["rafaga_max_kmh"].transform(
                lambda s: s.diff().fillna(0.0)
            )
        else:
            data["precip_roll_3h"] = data["precip_acumulada_mm"].rolling(3, min_periods=1).sum()
            data["precip_roll_6h"] = data["precip_acumulada_mm"].rolling(6, min_periods=1).sum()
            data["precip_roll_12h"] = data["precip_acumulada_mm"].rolling(12, min_periods=1).sum()
            data["rafaga_max_roll_3h"] = data["rafaga_max_kmh"].rolling(3, min_periods=1).max()
            data["rafaga_max_roll_6h"] = data["rafaga_max_kmh"].rolling(6, min_periods=1).max()
            data["delta_rafaga_1h"] = data["rafaga_max_kmh"].diff().fillna(0.0)

        # Energía cinética específica del viento (v en m/s, E = 0.5 * v^2)
        v_ms = data["rafaga_max_kmh"] / 3.6
        data["energia_cinetica_viento"] = (0.5 * (v_ms**2)).round(2)

        # Categorización de terciles socioeconómicos
        if "nse_score" in data.columns:
            if data["nse_score"].nunique() >= 3:
                try:
                    data["tercil_nse"] = pd.qcut(
                        data["nse_score"],
                        q=3,
                        labels=["Vulnerable", "Medio", "Alto"],
                    )
                except ValueError:
                    data["tercil_nse"] = pd.cut(
                        data["nse_score"],
                        bins=[-float("inf"), -0.5, 0.5, float("inf")],
                        labels=["Vulnerable", "Medio", "Alto"],
                    )
            else:
                data["tercil_nse"] = pd.cut(
                    data["nse_score"],
                    bins=[-float("inf"), -0.5, 0.5, float("inf")],
                    labels=["Vulnerable", "Medio", "Alto"],
                )

        return data


def build_survival_dataset(
    df: pd.DataFrame,
    critical_rate_threshold: float = 0.05,
    event_col_name: str = "evento",
    comuna_id_col: str = "comuna_id",
) -> pd.DataFrame:
    """Construye un dataset longitudinal de supervivencia (Time-to-Event).

    Para cada unidad (evento, comuna):
    - duration: Horas transcurridas desde el inicio del temporal hasta la primera
      desconexión que supera critical_rate_threshold (corte crítico), o la duración
      total observada del evento si no colapsa (censura por la derecha).
    - event: 1 si ocurrió el corte crítico dentro de la ventana de observación, 0 si fue censurado.
    - Covariables: nse_score, tercil_nse, red_aerea_km_ratio, arbolado_m2_hab, es_cge, es_rural,
      rafaga_max_evento, precip_total_evento.
    """
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    group_keys = (
        [event_col_name, comuna_id_col]
        if event_col_name in data.columns
        else [comuna_id_col]
    )

    records: list[dict[str, Any]] = []

    for _group_val, group in data.groupby(group_keys):
        sorted_group = group.sort_values("timestamp").reset_index(drop=True)
        total_hours = len(sorted_group)

        # Búsqueda del primer instante de corte crítico
        critical_mask = sorted_group["tasa_afectacion"] >= critical_rate_threshold
        critical_indices = sorted_group.index[critical_mask].tolist()

        if len(critical_indices) > 0:
            # Duración medida en horas (1-indexed para evitar duración 0 en lifelines)
            duration = float(critical_indices[0] + 1)
            event_occurred = 1
        else:
            duration = float(max(1, total_hours))
            event_occurred = 0

        first_row = sorted_group.iloc[0]

        record: dict[str, Any] = {
            "duration": duration,
            "event": event_occurred,
            "comuna_id": str(first_row.get("comuna_id", "")),
            "comuna_nombre": str(first_row.get("comuna_nombre", "")),
            "nse_score": float(first_row.get("nse_score", 0.0)),
            "ingreso_autonomo_promedio": float(
                first_row.get("ingreso_autonomo_promedio", 0.0)
            ),
            "pobreza_multidimensional": float(
                first_row.get("pobreza_multidimensional", 0.0)
            ),
            "arbolado_m2_hab": float(first_row.get("arbolado_m2_hab", 5.0)),
            "red_aerea_km_ratio": float(first_row.get("red_aerea_km_ratio", 0.70)),
            "es_cge": float(
                1.0
                if str(first_row.get("empresa", "")).upper() == "CGE"
                else 0.0
            ),
            "es_rural": float(first_row.get("es_rural", 0)),
            "clientes_totales": int(first_row.get("clientes_totales", 10000)),
            "rafaga_max_evento": float(sorted_group["rafaga_max_kmh"].max()),
            "precip_total_evento": float(
                sorted_group["precip_acumulada_mm"].max()
            ),
        }

        if event_col_name in sorted_group.columns:
            record["evento"] = str(first_row[event_col_name])

        records.append(record)

    surv_df = pd.DataFrame(records)

    # Tercil socioeconómico
    if len(surv_df) >= 3 and surv_df["nse_score"].nunique() >= 3:
        try:
            surv_df["tercil_nse"] = pd.qcut(
                surv_df["nse_score"],
                q=3,
                labels=["Vulnerable", "Medio", "Alto"],
            ).astype(str)
        except ValueError:
            surv_df["tercil_nse"] = pd.cut(
                surv_df["nse_score"],
                bins=[-float("inf"), -0.5, 0.5, float("inf")],
                labels=["Vulnerable", "Medio", "Alto"],
            ).astype(str)
    else:
        surv_df["tercil_nse"] = pd.cut(
            surv_df["nse_score"],
            bins=[-float("inf"), -0.5, 0.5, float("inf")],
            labels=["Vulnerable", "Medio", "Alto"],
        ).astype(str)

    return surv_df


def process_and_save_feature_pipeline(
    input_parquet: Path | None = None,
    output_enriched_path: Path | None = None,
    output_survival_path: Path | None = None,
) -> tuple[Path, Path]:
    """Ejecuta el pipeline completo de ingeniería de features y persiste los Parquets procesados."""
    root = get_project_root()
    if input_parquet is None:
        input_parquet = root / "data" / "processed" / "benchmark_temporales_2024.parquet"
    if output_enriched_path is None:
        output_enriched_path = (
            root / "data" / "processed" / "panel_features_enriched.parquet"
        )
    if output_survival_path is None:
        output_survival_path = root / "data" / "processed" / "survival_dataset.parquet"

    if not input_parquet.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_parquet}")

    df_raw = pd.read_parquet(input_parquet)

    engineer = FeatureEngineer()
    df_enriched = engineer.compute_rolling_weather_features(df_raw)
    df_survival = build_survival_dataset(df_raw)

    output_enriched_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(output_enriched_path, index=False)

    output_survival_path.parent.mkdir(parents=True, exist_ok=True)
    df_survival.to_parquet(output_survival_path, index=False)

    return output_enriched_path, output_survival_path
