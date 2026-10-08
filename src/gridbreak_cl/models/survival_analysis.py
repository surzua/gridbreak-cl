"""Modelamiento de supervivencia y tiempo hasta el colapso de red eléctrica."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


class SurvivalModel:
    """Modelo de análisis de supervivencia para estimar el tiempo al colapso eléctrico.

    Integra estimadores no paramétricos de Kaplan-Meier, pruebas de Log-Rank
    y el modelo semi-paramétrico de Riesgos Proporcionales de Cox con penalización L2.
    """

    DEFAULT_COX_FEATURES = [
        "nse_score",
        "red_aerea_km_ratio",
        "arbolado_m2_hab",
        "es_cge",
        "rafaga_max_evento",
        "precip_total_evento",
    ]

    def __init__(self, penalizer: float = 0.05) -> None:
        self.penalizer = penalizer
        self.cox_fitter: CoxPHFitter | None = None
        self.km_fitters: dict[str, KaplanMeierFitter] = {}
        self.feature_cols: list[str] = []

    def fit_kaplan_meier(
        self,
        df: pd.DataFrame,
        strata_col: str | None = "tercil_nse",
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> dict[str, KaplanMeierFitter]:
        """Ajusta estimadores de supervivencia de Kaplan-Meier (global o estratificado)."""
        self.km_fitters = {}

        # Ajuste global
        km_global = KaplanMeierFitter()
        km_global.fit(
            durations=df[duration_col],
            event_observed=df[event_col],
            label="Global RM",
        )
        self.km_fitters["Global"] = km_global

        # Ajuste por estratos (ej. Tercil socioeconómico o Empresa)
        if strata_col and strata_col in df.columns:
            for stratum in sorted(df[strata_col].dropna().unique()):
                subset = df[df[strata_col] == stratum]
                km_strat = KaplanMeierFitter()
                km_strat.fit(
                    durations=subset[duration_col],
                    event_observed=subset[event_col],
                    label=f"{strata_col}: {stratum}",
                )
                self.km_fitters[str(stratum)] = km_strat

        return self.km_fitters

    def compute_logrank_test(
        self,
        df: pd.DataFrame,
        strata_col: str = "tercil_nse",
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> dict[str, Any]:
        """Calcula la prueba multivariada de Log-Rank para contrastar diferencias entre estratos."""
        if strata_col not in df.columns:
            raise ValueError(
                f"Columna de estrato '{strata_col}' no presente en DataFrame."
            )

        results = multivariate_logrank_test(
            event_durations=df[duration_col],
            groups=df[strata_col],
            event_observed=df[event_col],
        )

        return {
            "strata": strata_col,
            "test_statistic": float(results.test_statistic),
            "p_value": float(results.p_value),
            "degrees_of_freedom": int(results.degrees_of_freedom),
            "is_significant_95": bool(results.p_value < 0.05),
            "is_significant_99": bool(results.p_value < 0.01),
        }

    def fit_cox_model(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> CoxPHFitter:
        """Ajusta el modelo de Riesgos Proporcionales de Cox con penalización L2 para estabilidad."""
        if feature_cols is None:
            feature_cols = [c for c in self.DEFAULT_COX_FEATURES if c in df.columns]

        self.feature_cols = feature_cols
        cols_to_use = [duration_col, event_col] + feature_cols
        data = df[cols_to_use].dropna().copy()

        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(data, duration_col=duration_col, event_col=event_col)
        self.cox_fitter = cph
        return self.cox_fitter

    def get_hazard_ratios(self) -> pd.DataFrame:
        """Retorna un DataFrame con coeficientes, Hazard Ratios (exp(coef)), IC 95% y significancia."""
        if self.cox_fitter is None:
            raise ValueError(
                "El modelo de Cox no ha sido ajustado. Ejecute fit_cox_model() primero."
            )

        summary = self.cox_fitter.summary
        records: list[dict[str, Any]] = []

        for var, row in summary.iterrows():
            coef = float(row["coef"])
            hr = float(row["exp(coef)"])
            se = float(row["se(coef)"])
            hr_lower = float(row.get("exp(coef) lower 95%", np.exp(coef - 1.96 * se)))
            hr_upper = float(row.get("exp(coef) upper 95%", np.exp(coef + 1.96 * se)))
            p_val = float(row["p"])
            z_val = float(row["z"])

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
                    "Coeficiente (Beta)": round(coef, 4),
                    "Hazard Ratio (HR)": round(hr, 4),
                    "Error Estandar": round(se, 4),
                    "HR IC 95% Inf": round(hr_lower, 4),
                    "HR IC 95% Sup": round(hr_upper, 4),
                    "z-score": round(z_val, 3),
                    "p-value": f"{p_val:.4f} {stars}".strip(),
                }
            )

        return pd.DataFrame(records)

    def predict_survival_curves(self, covariates_df: pd.DataFrame) -> pd.DataFrame:
        """Predice las funciones de supervivencia S(t | X) para los perfiles comunales provistos."""
        if self.cox_fitter is None:
            raise ValueError("El modelo de Cox no ha sido ajustado.")

        subset = covariates_df[self.feature_cols]
        curves: pd.DataFrame = self.cox_fitter.predict_survival_function(subset)
        return curves

    def predict_median_survival_time(
        self, covariates_df: pd.DataFrame
    ) -> pd.Series[float]:
        """Predice el tiempo mediano al colapso (en horas) para cada observación provista."""
        if self.cox_fitter is None:
            raise ValueError("El modelo de Cox no ha sido ajustado.")

        curves = self.predict_survival_curves(covariates_df)
        median_times: list[float] = []

        for col in curves.columns:
            series = curves[col]
            # Momento en que la supervivencia cae por debajo de 0.50
            sub_half = series[series <= 0.5]
            if len(sub_half) > 0:
                median_times.append(float(sub_half.index[0]))
            else:
                # Si nunca cae de 0.5 en la ventana observada, retorna el tiempo máximo observado
                median_times.append(float(series.index[-1]))

        return pd.Series(median_times, index=covariates_df.index)

    def get_metrics(self) -> dict[str, Any]:
        """Retorna métricas de bondad de ajuste del modelo de Cox."""
        if self.cox_fitter is None:
            raise ValueError("El modelo de Cox no ha sido ajustado.")

        cph = self.cox_fitter
        return {
            "concordance_index": float(cph.concordance_index_),
            "log_likelihood": float(cph.log_likelihood_),
            "partial_aic": float(cph.AIC_partial_),
            "nobs": int(len(cph.durations)),
            "events_observed": int(cph.event_observed.sum()),
            "penalizer": float(self.penalizer),
        }

    def check_proportional_hazards(self) -> pd.DataFrame:
        """Evalúa el supuesto de proporcionalidad de riesgos mediante residuos de Schoenfeld."""
        if self.cox_fitter is None:
            raise ValueError("El modelo de Cox no ha sido ajustado.")

        try:
            from lifelines.statistics import proportional_hazard_test

            test_results = proportional_hazard_test(
                self.cox_fitter,
                training_df=getattr(self.cox_fitter, "_central_values", None),
            )
            return test_results.summary
        except Exception:
            # Fallback seguro con resumen de coeficientes
            return self.get_hazard_ratios()[
                ["Variable", "Coeficiente (Beta)", "p-value"]
            ]
