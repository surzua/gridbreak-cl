"""Generación de gráficos estáticos de grado publicación y portafolio técnico."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridbreak_cl.config import get_project_root
from gridbreak_cl.models.fragility_curves import FragilityModel
from gridbreak_cl.models.survival_analysis import SurvivalModel

# Paleta editorial de alto contraste
PALETTE = {
    "vulnerable": "#e74c3c",  # Rojo señal
    "medio": "#f39c12",  # Naranja alerta
    "alto": "#2ecc71",  # Verde resiliente
    "referencia": "#3498db",  # Azul institucional
    "neutral_dark": "#2c3e50",  # Gris oscuro
    "grid": "#ecf0f1",  # Gris claro
}


def set_portfolio_style() -> None:
    """Configura el estilo base de matplotlib para gráficos editoriales de alta fidelidad."""
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#bdc3c7"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = PALETTE["grid"]
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.6


def plot_fragility_curves_comparison(
    model: FragilityModel,
    wind_kmh: float = 60.0,
    rain_range: tuple[float, float] = (0.0, 80.0),
    output_path: Path | None = None,
) -> Path:
    """Genera curvas sigmoides de fragilidad comparando los 3 terciles socioeconómicos."""
    set_portfolio_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    r_vals = np.linspace(rain_range[0], rain_range[1], 120)
    scenarios: list[dict[str, Any]] = [
        {
            "name": "Tercil Vulnerable (NSE = -1.5)",
            "nse": -1.5,
            "color": PALETTE["vulnerable"],
            "aerial": 0.88,
        },
        {
            "name": "Tercil Medio (NSE = 0.0)",
            "nse": 0.0,
            "color": PALETTE["medio"],
            "aerial": 0.70,
        },
        {
            "name": "Tercil Acomodado (NSE = +1.8)",
            "nse": 1.8,
            "color": PALETTE["alto"],
            "aerial": 0.38,
        },
    ]

    for sc in scenarios:
        probs = [
            model.predict_probability(
                precip_mm=r,
                rafaga_kmh=wind_kmh,
                nse_score=float(sc["nse"]),
                red_aerea_ratio=float(sc["aerial"]),
                es_cge=0.0,
                arbolado_m2_hab=5.0,
            )
            for r in r_vals
        ]
        ax.plot(r_vals, probs, label=sc["name"], color=sc["color"], lw=2.8)

        # Cálculo de umbral R50
        r50 = model.compute_critical_rain_threshold(
            rafaga_kmh=wind_kmh,
            nse_score=float(sc["nse"]),
            red_aerea_ratio=float(sc["aerial"]),
        )
        if 0.0 < r50 <= rain_range[1]:
            ax.scatter([r50], [0.5], color=sc["color"], s=70, zorder=5)
            ax.annotate(
                f"$R_{{50}}={r50:.1f}$ mm",
                (r50, 0.5),
                textcoords="offset points",
                xytext=(10, -12 if sc["nse"] > 0 else 8),
                fontsize=9,
                fontweight="bold",
                color=sc["color"],
            )

    # Línea umbral del 50%
    ax.axhline(
        0.5,
        color="#7f8c8d",
        linestyle=":",
        lw=1.2,
        label="Umbral Crítico de Falla ($P=0.50$)",
    )

    ax.set_title(
        f"Curvas de Fragilidad Eléctrica RM: Brecha Socioeconómica\n(Ráfaga constante = {wind_kmh:.0f} km/h)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Precipitación Acumulada (mm)", fontsize=11)
    ax.set_ylabel("Probabilidad de Corte Crítico (>5% comunal)", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(rain_range[0], rain_range[1])
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    plt.tight_layout()

    if output_path is None:
        output_path = (
            get_project_root()
            / "reports"
            / "figures"
            / "curvas_fragilidad_terciles.png"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_kaplan_meier_survival(
    surv_model: SurvivalModel,
    df_surv: pd.DataFrame,
    strata_col: str = "tercil_nse",
    output_path: Path | None = None,
) -> Path:
    """Genera curvas empíricas de supervivencia Kaplan-Meier con test de Log-Rank."""
    set_portfolio_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    km_dict = surv_model.fit_kaplan_meier(df_surv, strata_col=strata_col)
    logrank = surv_model.compute_logrank_test(df_surv, strata_col=strata_col)

    color_map = {
        "Vulnerable": PALETTE["vulnerable"],
        "Medio": PALETTE["medio"],
        "Alto": PALETTE["alto"],
        "Global": PALETTE["neutral_dark"],
    }

    for label, kmf in km_dict.items():
        if label == "Global":
            continue
        timeline = kmf.timeline
        sf = kmf.survival_function_[kmf.survival_function_.columns[0]]
        color = color_map.get(str(label), PALETTE["referencia"])

        ax.step(
            timeline, sf, where="post", label=f"Tercil {label}", color=color, lw=2.5
        )

    ax.axhline(
        0.5,
        color="#7f8c8d",
        linestyle=":",
        lw=1.2,
        label="Mediana de Supervivencia ($S=0.50$)",
    )

    ax.set_title(
        f"Dinámica Temporal de Colapso Eléctrico: Curvas Kaplan-Meier\nLog-Rank Test $p = {logrank['p_value']:.2e}$",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Horas transcurridas desde el inicio del temporal", fontsize=11)
    ax.set_ylabel("Probabilidad de Supervivencia de la Red $S(t)$", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    plt.tight_layout()

    if output_path is None:
        output_path = (
            get_project_root()
            / "reports"
            / "figures"
            / "supervivencia_kaplan_meier.png"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_cox_hazard_ratios(
    surv_model: SurvivalModel,
    output_path: Path | None = None,
) -> Path:
    """Genera Forest Plot de Hazard Ratios del modelo de Riesgos Proporcionales de Cox."""
    set_portfolio_style()
    hr_df = surv_model.get_hazard_ratios().copy()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    y_pos = np.arange(len(hr_df))
    hrs = hr_df["Hazard Ratio (HR)"].values
    ci_inf = hr_df["HR IC 95% Inf"].values
    ci_sup = hr_df["HR IC 95% Sup"].values
    labels = hr_df["Variable"].values

    # Eje horizontal y línea de nulidad HR = 1.0
    ax.axvline(
        1.0, color="#7f8c8d", linestyle="--", lw=1.2, label="Sin Efecto (HR = 1.0)"
    )

    # Limitar visualmente el eje para presentación limpia
    display_hrs = np.clip(hrs, 0.05, 15.0)
    display_err_lower = display_hrs - np.clip(ci_inf, 0.05, 15.0)
    display_err_upper = np.clip(ci_sup, 0.05, 15.0) - display_hrs

    # Colorear según factor protector o de riesgo
    colors = [PALETTE["alto"] if h < 1.0 else PALETTE["vulnerable"] for h in hrs]

    ax.errorbar(
        display_hrs,
        y_pos,
        xerr=[display_err_lower, display_err_upper],
        fmt="o",
        color=PALETTE["neutral_dark"],
        ecolor="#95a5a6",
        elinewidth=1.8,
        capsize=4,
        markersize=7,
    )
    ax.scatter(display_hrs, y_pos, color=colors, s=70, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Hazard Ratio $\\exp(\\beta)$ [Escala Acotada]", fontsize=11)
    ax.set_title(
        "Modelo de Cox: Factores Determinantes del Colapso Eléctrico\n(Hazard Ratios con Intervalo de Confianza 95%)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", loc="lower right")

    plt.tight_layout()

    if output_path is None:
        output_path = (
            get_project_root() / "reports" / "figures" / "hazard_ratios_forest_plot.png"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_critical_thresholds_gap(
    model: FragilityModel,
    df_comunas: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Genera gráfico de barras comparativo de umbrales críticos de viento W50 para comunas clave."""
    set_portfolio_style()

    comunas_sel = [
        "Cerro Navia",
        "La Pintana",
        "Lo Espejo",
        "San Ramón",
        "Renca",
        "Santiago",
        "Ñuñoa",
        "Providencia",
        "Las Condes",
        "Vitacura",
    ]
    sub = df_comunas[df_comunas["comuna_nombre"].isin(comunas_sel)].copy()

    w50_vals: list[float] = []
    for _, row in sub.iterrows():
        w = model.compute_critical_wind_threshold(
            precip_mm=30.0,
            nse_score=float(row["nse_score"]),
            red_aerea_ratio=float(row["red_aerea_km_ratio"]),
            es_cge=1.0 if row["empresa"] == "CGE" else 0.0,
            arbolado_m2_hab=float(row["arbolado_m2_hab"]),
        )
        w50_vals.append(round(w, 1))

    sub["w50"] = w50_vals
    sub = sub.sort_values("w50").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    colors = [
        PALETTE["vulnerable"]
        if n < -0.5
        else PALETTE["alto"]
        if n > 0.5
        else PALETTE["medio"]
        for n in sub["nse_score"]
    ]

    bars = ax.barh(sub["comuna_nombre"], sub["w50"], color=colors, height=0.65)

    for bar, val in zip(bars, sub["w50"], strict=False):
        ax.text(
            val + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} km/h",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        "Umbral Crítico de Ráfaga ($W_{50}$) ante 30 mm de Lluvia\nVelocidad de viento que detona 50% de probabilidad de colapso",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Ráfaga de Viento Crítica ($W_{50}$ en km/h)", fontsize=11)
    ax.set_xlim(0, max(sub["w50"]) + 15)
    ax.grid(axis="x")

    plt.tight_layout()

    if output_path is None:
        output_path = (
            get_project_root()
            / "reports"
            / "figures"
            / "brecha_umbrales_viento_comunal.png"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_all_portfolio_figures(output_dir: Path | None = None) -> list[Path]:
    """Genera y guarda el conjunto completo de figuras estáticas de alta resolución para el portafolio."""
    root = get_project_root()
    if output_dir is None:
        output_dir = root / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar modelos y datos
    df_raw = pd.read_parquet(
        root / "data" / "processed" / "benchmark_temporales_2024.parquet"
    )
    frag_model = FragilityModel()
    frag_model.fit(df_raw)

    surv_path = root / "data" / "processed" / "survival_dataset.parquet"
    if not surv_path.exists():
        from gridbreak_cl.features.build_features import (
            process_and_save_feature_pipeline,
        )

        process_and_save_feature_pipeline()

    df_surv = pd.read_parquet(surv_path)
    surv_model = SurvivalModel(penalizer=0.05)
    surv_model.fit_kaplan_meier(df_surv, strata_col="tercil_nse")
    surv_model.fit_cox_model(df_surv)

    from gridbreak_cl.etl.historical_seed import get_rm_comunas_metadata

    df_comunas = get_rm_comunas_metadata()

    paths = [
        plot_fragility_curves_comparison(
            frag_model, output_path=output_dir / "curvas_fragilidad_terciles.png"
        ),
        plot_kaplan_meier_survival(
            surv_model,
            df_surv,
            output_path=output_dir / "supervivencia_kaplan_meier.png",
        ),
        plot_cox_hazard_ratios(
            surv_model, output_path=output_dir / "hazard_ratios_forest_plot.png"
        ),
        plot_critical_thresholds_gap(
            frag_model,
            df_comunas,
            output_path=output_dir / "brecha_umbrales_viento_comunal.png",
        ),
    ]

    return paths
