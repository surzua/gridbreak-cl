"""Ajuste de modelos logísticos e inferencia de curvas y superficies de fragilidad comunal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults


class FragilityModel:
    """Modelo de Fragilidad Eléctrica ante estresores climáticos (Lluvia y Viento).

    Ajusta un Modelo Lineal Generalizado (GLM) binomial logit con términos de interacción
    para estimar la probabilidad de que una comuna supere el umbral de desconexión crítica.
    """

    FEATURE_COLS = [
        "precip_acumulada_mm",
        "rafaga_max_kmh",
        "nse_score",
        "precip_x_nse",
        "rafaga_x_nse",
        "es_cge",
        "arbolado_m2_hab",
    ]

    def __init__(self) -> None:
        self.model_results: GLMResults | None = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera términos de interacción y transformaciones requeridas."""
        data = df.copy()
        data["precip_x_nse"] = data["precip_acumulada_mm"] * data["nse_score"]
        data["rafaga_x_nse"] = data["rafaga_max_kmh"] * data["nse_score"]

        if "es_cge" not in data.columns:
            if "empresa" in data.columns:
                data["es_cge"] = (
                    data["empresa"].astype(str).str.upper() == "CGE"
                ).astype(float)
            else:
                data["es_cge"] = 0.0

        if "arbolado_m2_hab" not in data.columns:
            data["arbolado_m2_hab"] = 5.0

        return data

    def fit(self, df: pd.DataFrame) -> GLMResults:
        """Ajusta el modelo GLM binomial sobre el dataset preparado."""
        data = self._prepare_features(df)
        X = data[self.FEATURE_COLS]
        X = sm.add_constant(X, has_constant="add")
        y = data["es_corte_critico"].astype(int)

        glm = sm.GLM(y, X, family=sm.families.Binomial())
        self.model_results = glm.fit()
        return self.model_results

    def predict_probability(
        self,
        precip_mm: float,
        rafaga_kmh: float,
        nse_score: float,
        es_cge: float = 0.0,
        arbolado_m2_hab: float = 5.0,
    ) -> float:
        """Calcula la probabilidad predicha de corte crítico (0.0 a 1.0)."""
        if self.model_results is None:
            raise ValueError("El modelo debe ser ajustado previamente con fit().")

        p = self.model_results.params
        logit = (
            p["const"]
            + p["precip_acumulada_mm"] * precip_mm
            + p["rafaga_max_kmh"] * rafaga_kmh
            + p["nse_score"] * nse_score
            + p["precip_x_nse"] * (precip_mm * nse_score)
            + p["rafaga_x_nse"] * (rafaga_kmh * nse_score)
            + p["es_cge"] * es_cge
            + p["arbolado_m2_hab"] * arbolado_m2_hab
        )
        return float(1.0 / (1.0 + np.exp(-logit)))

    def compute_critical_rain_threshold(
        self,
        rafaga_kmh: float,
        nse_score: float,
        es_cge: float = 0.0,
        arbolado_m2_hab: float = 5.0,
        target_prob: float = 0.5,
    ) -> float:
        """Calcula analíticamente los mm de lluvia necesarios para alcanzar target_prob."""
        if self.model_results is None:
            raise ValueError("El modelo debe ser ajustado previamente con fit().")

        p = self.model_results.params
        target_logit = float(np.log(target_prob / (1.0 - target_prob)))

        numerator = target_logit - (
            p["const"]
            + p["rafaga_max_kmh"] * rafaga_kmh
            + p["nse_score"] * nse_score
            + p["rafaga_x_nse"] * (rafaga_kmh * nse_score)
            + p["es_cge"] * es_cge
            + p["arbolado_m2_hab"] * arbolado_m2_hab
        )
        denominator = p["precip_acumulada_mm"] + p["precip_x_nse"] * nse_score

        if denominator == 0:
            return float("nan")

        return float(numerator / denominator)

    def compute_critical_wind_threshold(
        self,
        precip_mm: float,
        nse_score: float,
        es_cge: float = 0.0,
        arbolado_m2_hab: float = 5.0,
        target_prob: float = 0.5,
    ) -> float:
        """Calcula analíticamente la ráfaga de viento (km/h) para alcanzar target_prob."""
        if self.model_results is None:
            raise ValueError("El modelo debe ser ajustado previamente con fit().")

        p = self.model_results.params
        target_logit = float(np.log(target_prob / (1.0 - target_prob)))

        numerator = target_logit - (
            p["const"]
            + p["precip_acumulada_mm"] * precip_mm
            + p["nse_score"] * nse_score
            + p["precip_x_nse"] * (precip_mm * nse_score)
            + p["es_cge"] * es_cge
            + p["arbolado_m2_hab"] * arbolado_m2_hab
        )
        denominator = p["rafaga_max_kmh"] + p["rafaga_x_nse"] * nse_score

        if denominator == 0:
            return float("nan")

        return float(numerator / denominator)

    def generate_fragility_surface(
        self,
        nse_score: float,
        es_cge: float = 0.0,
        arbolado_m2_hab: float = 5.0,
        rain_range: tuple[float, float] = (0.0, 80.0),
        wind_range: tuple[float, float] = (0.0, 120.0),
        resolution: int = 40,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Genera una malla (Precip, Viento, Probabilidad) para gráficos 3D."""
        r_vals = np.linspace(rain_range[0], rain_range[1], resolution)
        w_vals = np.linspace(wind_range[0], wind_range[1], resolution)
        R_grid, W_grid = np.meshgrid(r_vals, w_vals)

        prob_grid = np.zeros_like(R_grid)
        for i in range(resolution):
            for j in range(resolution):
                prob_grid[i, j] = self.predict_probability(
                    precip_mm=float(R_grid[i, j]),
                    rafaga_kmh=float(W_grid[i, j]),
                    nse_score=nse_score,
                    es_cge=es_cge,
                    arbolado_m2_hab=arbolado_m2_hab,
                )

        return R_grid, W_grid, prob_grid
