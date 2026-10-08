"""Ajuste de modelos logísticos e inferencia de curvas y superficies de fragilidad comunal."""

from __future__ import annotations

from typing import Any

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
        "rafaga_cuadratica",
        "nse_score",
        "precip_x_nse",
        "rafaga_x_nse",
        "red_aerea_km_ratio",
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

        if "rafaga_cuadratica" not in data.columns:
            data["rafaga_cuadratica"] = (data["rafaga_max_kmh"] ** 2) / 100.0

        if "red_aerea_km_ratio" not in data.columns:
            data["red_aerea_km_ratio"] = 0.70

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
        red_aerea_ratio: float = 0.70,
        es_cge: float = 0.0,
        arbolado_m2_hab: float = 5.0,
    ) -> float:
        """Calcula la probabilidad predicha de corte crítico (0.0 a 1.0)."""
        if self.model_results is None:
            raise ValueError("El modelo debe ser ajustado previamente con fit().")

        p = self.model_results.params
        rafaga_cuad = (rafaga_kmh**2) / 100.0
        logit = (
            p["const"]
            + p["precip_acumulada_mm"] * precip_mm
            + p["rafaga_max_kmh"] * rafaga_kmh
            + p.get("rafaga_cuadratica", 0.0) * rafaga_cuad
            + p["nse_score"] * nse_score
            + p["precip_x_nse"] * (precip_mm * nse_score)
            + p["rafaga_x_nse"] * (rafaga_kmh * nse_score)
            + p.get("red_aerea_km_ratio", 0.0) * red_aerea_ratio
            + p["es_cge"] * es_cge
            + p["arbolado_m2_hab"] * arbolado_m2_hab
        )
        return float(1.0 / (1.0 + np.exp(-logit)))

    def compute_critical_rain_threshold(
        self,
        rafaga_kmh: float,
        nse_score: float,
        red_aerea_ratio: float = 0.70,
        es_cge: float = 0.0,
        arbolado_m2_hab: float = 5.0,
        target_prob: float = 0.5,
    ) -> float:
        """Calcula analíticamente los mm de lluvia necesarios para alcanzar target_prob."""
        if self.model_results is None:
            raise ValueError("El modelo debe ser ajustado previamente con fit().")

        # Si a 0 mm de lluvia ya supera target_prob, el umbral es 0.0 mm
        base_prob = self.predict_probability(
            precip_mm=0.0,
            rafaga_kmh=rafaga_kmh,
            nse_score=nse_score,
            red_aerea_ratio=red_aerea_ratio,
            es_cge=es_cge,
            arbolado_m2_hab=arbolado_m2_hab,
        )
        if base_prob >= target_prob:
            return 0.0

        p = self.model_results.params
        target_logit = float(np.log(target_prob / (1.0 - target_prob)))
        rafaga_cuad = (rafaga_kmh**2) / 100.0

        numerator = target_logit - (
            p["const"]
            + p["rafaga_max_kmh"] * rafaga_kmh
            + p.get("rafaga_cuadratica", 0.0) * rafaga_cuad
            + p["nse_score"] * nse_score
            + p["rafaga_x_nse"] * (rafaga_kmh * nse_score)
            + p.get("red_aerea_km_ratio", 0.0) * red_aerea_ratio
            + p["es_cge"] * es_cge
            + p["arbolado_m2_hab"] * arbolado_m2_hab
        )
        denominator = p["precip_acumulada_mm"] + p["precip_x_nse"] * nse_score

        if denominator == 0:
            return float("nan")

        val = float(numerator / denominator)
        return max(0.0, val)

    def compute_critical_wind_threshold(
        self,
        precip_mm: float,
        nse_score: float,
        red_aerea_ratio: float = 0.70,
        es_cge: float = 0.0,
        arbolado_m2_hab: float = 5.0,
        target_prob: float = 0.5,
    ) -> float:
        """Calcula analíticamente la ráfaga de viento (km/h) para alcanzar target_prob."""
        if self.model_results is None:
            raise ValueError("El modelo debe ser ajustado previamente con fit().")

        # Si a 0 km/h de viento ya supera target_prob, el umbral es 0.0 km/h
        base_prob = self.predict_probability(
            precip_mm=precip_mm,
            rafaga_kmh=0.0,
            nse_score=nse_score,
            red_aerea_ratio=red_aerea_ratio,
            es_cge=es_cge,
            arbolado_m2_hab=arbolado_m2_hab,
        )
        if base_prob >= target_prob:
            return 0.0

        p = self.model_results.params
        target_logit = float(np.log(target_prob / (1.0 - target_prob)))

        # a * W^2 + b * W + c = 0
        a = float(p.get("rafaga_cuadratica", 0.0) / 100.0)
        b = float(p["rafaga_max_kmh"] + p["rafaga_x_nse"] * nse_score)
        c = float(
            p["const"]
            + p["precip_acumulada_mm"] * precip_mm
            + p["nse_score"] * nse_score
            + p["precip_x_nse"] * (precip_mm * nse_score)
            + p.get("red_aerea_km_ratio", 0.0) * red_aerea_ratio
            + p["es_cge"] * es_cge
            + p["arbolado_m2_hab"] * arbolado_m2_hab
            - target_logit
        )

        if abs(a) < 1e-7:
            # Caso puramente lineal
            if b == 0:
                return float("nan")
            return max(0.0, float(-c / b))

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return float("nan")

        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)

        valid_roots = [r for r in [root1, root2] if r >= 0]
        if not valid_roots:
            return 0.0

        return float(min(valid_roots))

    def generate_fragility_surface(
        self,
        nse_score: float,
        red_aerea_ratio: float = 0.70,
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
                    red_aerea_ratio=red_aerea_ratio,
                    es_cge=es_cge,
                    arbolado_m2_hab=arbolado_m2_hab,
                )

        return R_grid, W_grid, prob_grid

    def get_metrics(self) -> dict[str, float]:
        """Retorna métricas de bondad de ajuste del modelo GLM."""
        if self.model_results is None:
            raise ValueError("El modelo debe ser ajustado previamente con fit().")
        res = self.model_results
        llf = float(res.llf)
        llnull = float(res.llnull)
        pseudo_r2 = 1.0 - (llf / llnull) if llnull != 0 else 0.0
        return {
            "aic": float(res.aic),
            "bic": float(getattr(res, "bic_deviance", res.bic)),
            "log_likelihood": llf,
            "null_log_likelihood": llnull,
            "pseudo_r2_mcfadden": float(pseudo_r2),
            "nobs": float(res.nobs),
        }

    def get_model_summary_df(self) -> pd.DataFrame:
        """Retorna un DataFrame con coeficientes, errores estándar, odds ratios y significancia."""
        if self.model_results is None:
            raise ValueError("El modelo debe ser ajustado previamente con fit().")
        res = self.model_results
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        odds_ratios = np.exp(params)

        records: list[dict[str, Any]] = []
        for var in params.index:
            p_val = float(pvalues[var])
            stars = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else ""
            )
            records.append(
                {
                    "Variable": str(var),
                    "Coeficiente": round(float(params[var]), 4),
                    "Error Estándar": round(float(bse[var]), 4),
                    "Odds Ratio": round(float(odds_ratios[var]), 4),
                    "p-value": f"{p_val:.4f} {stars}".strip(),
                }
            )
        return pd.DataFrame(records)
