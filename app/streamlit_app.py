"""Gridbreak Chile: El Umbral del Apagón.

Aplicación interactiva y Simulador de Estrés Climático para la Región Metropolitana,
con soporte dual: Simulador Econométrico GLM y Telemetría en Vivo (SEC & DMC/Open-Meteo).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gridbreak_cl.config import get_project_root
from gridbreak_cl.etl.historical_seed import (
    get_rm_comunas_metadata,
    save_benchmark_dataset,
)
from gridbreak_cl.etl.live_pipeline import run_live_pipeline
from gridbreak_cl.models.fragility_curves import FragilityModel

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Gridbreak Chile | El Umbral del Apagón",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data_and_model() -> tuple[pd.DataFrame, pd.DataFrame, FragilityModel]:
    """Carga metadatos comunales, dataset benchmark y entrena el modelo de fragilidad."""
    root = get_project_root()
    parquet_path = root / "data" / "processed" / "benchmark_temporales_2024.parquet"

    if not parquet_path.exists():
        save_benchmark_dataset(parquet_path)

    df_storm = pd.read_parquet(parquet_path)
    df_comunas = get_rm_comunas_metadata()

    model = FragilityModel()
    model.fit(df_storm)

    return df_comunas, df_storm, model


def load_live_telemetry(force_refresh: bool = False) -> pd.DataFrame:
    """Carga el snapshot más reciente de telemetría en vivo de la RM."""
    root = get_project_root()
    live_path = root / "data" / "processed" / "live_latest.parquet"

    if force_refresh or not live_path.exists():
        return run_live_pipeline()

    return pd.read_parquet(live_path)


# Cargar datos base y modelo entrenado
df_comunas, df_storm, model = load_data_and_model()

# --- SIDEBAR: SELECTOR DE MODO Y CONTROLES ---
st.sidebar.image(
    "https://img.icons8.com/fluency/96/lightning-bolt.png",
    width=64,
)
st.sidebar.title("⚡ Gridbreak Chile")
st.sidebar.caption("Fragilidad Eléctrica y Riesgo Climático RM")

app_mode = st.sidebar.radio(
    "Modo de Operación",
    [
        "🎮 Simulador Predictivo (Curvas GLM)",
        "📡 Telemetría en Vivo (SEC & DMC)",
    ],
    index=0,
)

st.sidebar.markdown("---")

if app_mode == "🎮 Simulador Predictivo (Curvas GLM)":
    st.sidebar.subheader("🎮 Escenario Meteorológico")

    preset = st.sidebar.selectbox(
        "Preconfiguración de Temporal",
        [
            "Personalizado",
            "Temporal Agosto 2024 (Vendaval Extremo)",
            "Temporal Junio 2024 (Río Atmosférico)",
            "Temporal Moderado Típico",
        ],
        index=0,
    )

    if preset == "Temporal Agosto 2024 (Vendaval Extremo)":
        default_rain = 25.0
        default_wind = 110.0
    elif preset == "Temporal Junio 2024 (Río Atmosférico)":
        default_rain = 65.0
        default_wind = 45.0
    elif preset == "Temporal Moderado Típico":
        default_rain = 20.0
        default_wind = 40.0
    else:
        default_rain = 30.0
        default_wind = 65.0

    sim_rain = st.sidebar.slider(
        "🌧️ Lluvia Acumulada (mm)",
        min_value=0.0,
        max_value=80.0,
        value=float(default_rain),
        step=2.0,
    )

    sim_wind = st.sidebar.slider(
        "💨 Ráfaga Máxima de Viento (km/h)",
        min_value=10.0,
        max_value=125.0,
        value=float(default_wind),
        step=5.0,
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Tesis Contrarian:** El colapso de la red ante temporales no es mera fuerza mayor; "
        "es el detonante de una asimetría estructural preexistente entre comunas de la RM."
    )

    # --- CABECERA PRINCIPAL SIMULADOR ---
    st.title("⚡ El Umbral del Apagón: Fragilidad de la Red Eléctrica en Santiago")
    st.markdown(
        "**Análisis Causal y Modelamiento de Fragilidad ante Eventos Climáticos** | "
        "*Región Metropolitana de Santiago*"
    )

    # Métricas resumen del escenario simulado
    prob_comunas: list[float] = []
    for _, com in df_comunas.iterrows():
        p = model.predict_probability(
            precip_mm=sim_rain,
            rafaga_kmh=sim_wind,
            nse_score=float(com["nse_score"]),
            red_aerea_ratio=float(com["red_aerea_km_ratio"]),
            es_cge=1.0 if com["empresa"] == "CGE" else 0.0,
            arbolado_m2_hab=float(com["arbolado_m2_hab"]),
        )
        prob_comunas.append(p)

    df_sim = df_comunas.copy()
    df_sim["prob_colapso"] = prob_comunas
    df_sim["colapso_critico"] = df_sim["prob_colapso"] >= 0.50

    comunas_en_riesgo = int(df_sim["colapso_critico"].sum())
    prob_media_rm = float(df_sim["prob_colapso"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌧️ Lluvia Simulada", f"{sim_rain:.1f} mm")
    c2.metric("💨 Ráfaga Simulada", f"{sim_wind:.1f} km/h")
    c3.metric(
        "🚨 Comunas en Colapso (>50%)",
        f"{comunas_en_riesgo} / {len(df_sim)}",
        delta=f"{round(comunas_en_riesgo / len(df_sim) * 100)}% de la RM",
        delta_color="inverse",
    )
    c4.metric(
        "📊 Probabilidad Media RM",
        f"{prob_media_rm * 100:.1f}%",
    )

    st.markdown("---")

    # --- SECCIÓN 1: MAPA Y COMPARADOR ---
    col_map, col_comp = st.columns([1.1, 0.9])

    with col_map:
        st.subheader("🗺️ Mapa de Vulnerabilidad Eléctrica Comunal")
        st.caption(
            "Probabilidad estimada de corte masivo (>5% desconectado) bajo el temporal actual"
        )

        fig_map = px.scatter_geo(
            df_sim,
            lat="lat",
            lon="lon",
            color="prob_colapso",
            size=df_sim["prob_colapso"].clip(0.1, 1.0) * 35,
            hover_name="comuna_nombre",
            hover_data={
                "lat": False,
                "lon": False,
                "empresa": True,
                "prob_colapso": ":.1%",
                "ingreso_autonomo_promedio": ":$,.0f",
                "red_aerea_km_ratio": ":.0%",
            },
            color_continuous_scale="Reds",
            range_color=[0, 1],
            labels={"prob_colapso": "Prob. Colapso"},
        )
        fig_map.update_geos(
            fitbounds="locations",
            visible=False,
        )
        fig_map.update_layout(
            margin={"l": 0, "r": 0, "t": 10, "b": 10},
            height=450,
            coloraxis_colorbar={"title": "Riesgo"},
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_comp:
        st.subheader("⚖️ Comparador Frente a Frente")
        st.caption("Contrasta la asimetría de respuesta entre dos comunas")

        comp_c1, comp_c2 = st.columns(2)
        comuna_a_name = comp_c1.selectbox(
            "Comuna A (Vulnerable)", df_sim["comuna_nombre"], index=2
        )  # Cerro Navia
        comuna_b_name = comp_c2.selectbox(
            "Comuna B (Acomodada)", df_sim["comuna_nombre"], index=13
        )  # Las Condes

        row_a = df_sim[df_sim["comuna_nombre"] == comuna_a_name].iloc[0]
        row_b = df_sim[df_sim["comuna_nombre"] == comuna_b_name].iloc[0]

        p_a = float(row_a["prob_colapso"])
        p_b = float(row_b["prob_colapso"])

        r50_a = model.compute_critical_rain_threshold(
            rafaga_kmh=sim_wind,
            nse_score=float(row_a["nse_score"]),
            red_aerea_ratio=float(row_a["red_aerea_km_ratio"]),
            es_cge=1.0 if row_a["empresa"] == "CGE" else 0.0,
            arbolado_m2_hab=float(row_a["arbolado_m2_hab"]),
        )
        r50_b = model.compute_critical_rain_threshold(
            rafaga_kmh=sim_wind,
            nse_score=float(row_b["nse_score"]),
            red_aerea_ratio=float(row_b["red_aerea_km_ratio"]),
            es_cge=1.0 if row_b["empresa"] == "CGE" else 0.0,
            arbolado_m2_hab=float(row_b["arbolado_m2_hab"]),
        )

        w50_a = model.compute_critical_wind_threshold(
            precip_mm=sim_rain,
            nse_score=float(row_a["nse_score"]),
            red_aerea_ratio=float(row_a["red_aerea_km_ratio"]),
            es_cge=1.0 if row_a["empresa"] == "CGE" else 0.0,
            arbolado_m2_hab=float(row_a["arbolado_m2_hab"]),
        )
        w50_b = model.compute_critical_wind_threshold(
            precip_mm=sim_rain,
            nse_score=float(row_b["nse_score"]),
            red_aerea_ratio=float(row_b["red_aerea_km_ratio"]),
            es_cge=1.0 if row_b["empresa"] == "CGE" else 0.0,
            arbolado_m2_hab=float(row_b["arbolado_m2_hab"]),
        )

        comp_c1.metric(f"Prob. {comuna_a_name}", f"{p_a * 100:.1f}%")
        comp_c2.metric(f"Prob. {comuna_b_name}", f"{p_b * 100:.1f}%")

        st.markdown(f"""
        | Métrica Clave | {comuna_a_name} | {comuna_b_name} | Brecha |
        | :--- | :--- | :--- | :--- |
        | **Distribuidora** | {row_a["empresa"]} | {row_b["empresa"]} | - |
        | **Ingreso Autónomo Promedio** | ${row_a["ingreso_autonomo_promedio"]:,.0f} | ${row_b["ingreso_autonomo_promedio"]:,.0f} | {row_b["ingreso_autonomo_promedio"] / row_a["ingreso_autonomo_promedio"]:.1f}x |
        | **Exposición Red Aérea** | {row_a["red_aerea_km_ratio"]:.0%} | {row_b["red_aerea_km_ratio"]:.0%} | {row_a["red_aerea_km_ratio"] - row_b["red_aerea_km_ratio"]:+.0%} |
        | **Umbral Crítico de Lluvia ($R_{{50}}$)** | **{r50_a:.1f} mm** | **{r50_b:.1f} mm** | **{r50_b - r50_a:+.1f} mm** |
        | **Umbral Crítico de Viento ($W_{{50}}$)** | **{w50_a:.1f} km/h** | **{w50_b:.1f} km/h** | **{w50_b - w50_a:+.1f} km/h** |
        """)

    # --- SECCIÓN 2: CURVAS DE FRAGILIDAD INTERACTIVAS ---
    st.markdown("---")
    st.subheader("📈 Curvas Sigmoides de Fragilidad Superpuestas")
    st.caption(
        "Observa cómo la curva de la comuna vulnerable asciende prematuramente ante el incremento de la lluvia."
    )

    rain_range = np.linspace(0, 80, 100)
    curve_a = [
        model.predict_probability(
            precip_mm=r,
            rafaga_kmh=sim_wind,
            nse_score=float(row_a["nse_score"]),
            red_aerea_ratio=float(row_a["red_aerea_km_ratio"]),
            es_cge=1.0 if row_a["empresa"] == "CGE" else 0.0,
            arbolado_m2_hab=float(row_a["arbolado_m2_hab"]),
        )
        for r in rain_range
    ]
    curve_b = [
        model.predict_probability(
            precip_mm=r,
            rafaga_kmh=sim_wind,
            nse_score=float(row_b["nse_score"]),
            red_aerea_ratio=float(row_b["red_aerea_km_ratio"]),
            es_cge=1.0 if row_b["empresa"] == "CGE" else 0.0,
            arbolado_m2_hab=float(row_b["arbolado_m2_hab"]),
        )
        for r in rain_range
    ]

    fig_curves = go.Figure()
    fig_curves.add_trace(
        go.Scatter(
            x=rain_range,
            y=curve_a,
            mode="lines",
            name=f"{comuna_a_name} (Vulnerable)",
            line={"color": "#e63946", "width": 3},
        )
    )
    fig_curves.add_trace(
        go.Scatter(
            x=rain_range,
            y=curve_b,
            mode="lines",
            name=f"{comuna_b_name} (Acomodada)",
            line={"color": "#457b9d", "width": 3},
        )
    )
    fig_curves.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="black",
        annotation_text="Umbral Crítico de Falla (50%)",
    )
    fig_curves.add_vline(
        x=sim_rain,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Temporal Actual: {sim_rain:.0f} mm",
    )
    fig_curves.update_layout(
        xaxis_title="Precipitación Acumulada (mm)",
        yaxis_title="Probabilidad de Colapso Masivo",
        yaxis_tickformat=".0%",
        height=400,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    # --- SECCIÓN 3: SUPERFICIE 3D DE FRAGILIDAD ---
    st.markdown("---")
    st.subheader("🧊 Superficie Bivariada 3D de Fragilidad")
    st.caption(
        f"Modelamiento conjunto: Lluvia $\\times$ Viento $\\rightarrow$ Probabilidad de Colapso para {comuna_a_name}"
    )

    r_grid, w_grid, p_grid = model.generate_fragility_surface(
        nse_score=float(row_a["nse_score"]),
        red_aerea_ratio=float(row_a["red_aerea_km_ratio"]),
        es_cge=1.0 if row_a["empresa"] == "CGE" else 0.0,
        arbolado_m2_hab=float(row_a["arbolado_m2_hab"]),
        resolution=35,
    )

    fig_3d = go.Figure(
        data=[
            go.Surface(
                x=r_grid,
                y=w_grid,
                z=p_grid,
                colorscale="Viridis",
                colorbar={"title": "Probabilidad"},
            )
        ]
    )
    fig_3d.update_layout(
        scene={
            "xaxis_title": "Lluvia (mm)",
            "yaxis_title": "Ráfaga (km/h)",
            "zaxis_title": "Prob. Colapso",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        height=500,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # --- SECCIÓN 4: EVIDENCIA ECONOMÉTRICA Y DIAGNÓSTICO CIENTÍFICO ---
    st.markdown("---")
    with st.expander(
        "🔬 Evidencia Econométrica y Diagnóstico del Modelo GLM", expanded=False
    ):
        st.markdown("""
        El modelo ajusta una regresión logística binomial con **términos de interacción cruzada**,
        control por **empresa concesionaria** y control por **exposición física de cableado aéreo** vs. subterráneo:
        """)

        metrics = model.get_metrics()
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric(
            "Pseudo R² (McFadden)", f"{metrics['pseudo_r2_mcfadden']:.3f}"
        )
        m_col2.metric("Akaike Info Criterion (AIC)", f"{metrics['aic']:.1f}")
        m_col3.metric("Log-Likelihood", f"{metrics['log_likelihood']:.1f}")
        m_col4.metric("Observaciones Panel", f"{int(metrics['nobs']):,}")

        st.subheader("Tabla de Coeficientes y Odds Ratios")
        df_summary = model.get_model_summary_df()
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.markdown("""
        > **Interpretación Clave:**
        > * **Interacción Lluvia × NSE y Viento × NSE ($p < 0.01$):** Coeficientes negativos confirman que a igualdad de temporal, una comuna con mayor NSE reduce significativamente el incremento marginal de riesgo.
        > * **Exposición de Red Aérea ($p < 0.001$):** El ratio de cableado aéreo muestra un efecto positivo contundente. Aun aislando este factor, la brecha de NSE permanece estadísticamente significativa.
        > * **Viento Cuadrático ($W^2 / 100$):** Refleja la aceleración del daño mecánico por presión aerodinámica no lineal.
        """)

        # Exportar datos
        csv_bytes = df_sim.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar Escenario Simulado Actual (CSV)",
            data=csv_bytes,
            file_name="simulacion_gridbreak_rm.csv",
            mime="text/csv",
        )

else:
    # =========================================================================
    # MODO 2: TELEMETRÍA EN VIVO (SEC + DMC/OPEN-METEO)
    # =========================================================================
    st.sidebar.subheader("📡 Acciones de Telemetría")
    refresh_clicked = st.sidebar.button(
        "🔄 Actualizar Telemetría Ahora", type="primary"
    )

    with st.spinner("Adquiriendo telemetría en vivo (SEC + Meteorología)..."):
        df_live = load_live_telemetry(force_refresh=refresh_clicked)

    ts_str = str(df_live["timestamp"].iloc[0])[:19]

    st.title("📡 Telemetría en Vivo: Estado Eléctrico y Clima en Tiempo Real")
    st.markdown(
        f"**Monitoreo de Interrupciones SEC e Interpolación Climática (IDW)** | "
        f"*Última sincronización: {ts_str} (Hora Chile)*"
    )

    total_afectados = int(df_live["clientes_sin_suministro"].sum())
    total_clientes_rm = int(df_live["clientes_totales"].sum())
    tasa_global = total_afectados / max(1, total_clientes_rm)
    comunas_criticas = int(df_live["es_corte_critico"].sum())
    lluvia_promedio = float(df_live["precip_acumulada_mm"].mean())
    rafaga_max_rm = float(df_live["rafaga_max_kmh"].max())

    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric(
        "⚡ Clientes Sin Luz RM",
        f"{total_afectados:,}",
        delta=f"{tasa_global:.2%} de la red",
        delta_color="inverse",
    )
    lc2.metric(
        "🚨 Comunas en Corte Crítico (>5%)",
        f"{comunas_criticas} / {len(df_live)}",
        delta="Estable" if comunas_criticas == 0 else "Alerta Masiva",
        delta_color="normal" if comunas_criticas == 0 else "inverse",
    )
    lc3.metric("🌧️ Lluvia Media RM", f"{lluvia_promedio:.1f} mm")
    lc4.metric("💨 Ráfaga Máxima RM", f"{rafaga_max_rm:.1f} km/h")

    st.markdown("---")

    # Calcular probabilidad predicha por el modelo bajo el clima actual
    pred_prob: list[float] = []
    for _, row in df_live.iterrows():
        p = model.predict_probability(
            precip_mm=float(row["precip_acumulada_mm"]),
            rafaga_kmh=float(row["rafaga_max_kmh"]),
            nse_score=float(row["nse_score"]),
            red_aerea_ratio=float(row["red_aerea_km_ratio"]),
            es_cge=float(row["es_cge"]),
            arbolado_m2_hab=float(row["arbolado_m2_hab"]),
        )
        pred_prob.append(p)
    df_live["prob_predicha_modelo"] = pred_prob

    tab_mapa, tab_tabla, tab_diagnostico = st.tabs(
        [
            "🗺️ Mapa en Vivo",
            "📋 Tabla de Monitoreo Comunal",
            "🔬 Observado vs. Modelo GLM",
        ]
    )

    with tab_mapa:
        col_live_map, col_live_detail = st.columns([1.1, 0.9])

        with col_live_map:
            st.subheader("Mapa de Clientes sin Suministro (SEC)")
            st.caption(
                "Burbujas proporcionales a clientes sin suministro y color por tasa de afectación"
            )

            fig_live = px.scatter_geo(
                df_live,
                lat="lat",
                lon="lon",
                color="tasa_afectacion",
                size=df_live["clientes_sin_suministro"].clip(10, 10000),
                hover_name="comuna_nombre",
                hover_data={
                    "lat": False,
                    "lon": False,
                    "empresa": True,
                    "clientes_sin_suministro": True,
                    "tasa_afectacion": ":.2%",
                    "estacion_mas_cercana": True,
                    "rafaga_max_kmh": ":.1f km/h",
                },
                color_continuous_scale="Reds",
                range_color=[0, 0.05],
                labels={"tasa_afectacion": "Tasa Corte"},
            )
            fig_live.update_geos(fitbounds="locations", visible=False)
            fig_live.update_layout(
                margin={"l": 0, "r": 0, "t": 10, "b": 10}, height=460
            )
            st.plotly_chart(fig_live, use_container_width=True)

        with col_live_detail:
            st.subheader("🔍 Detalle por Comuna")
            selected_comuna = st.selectbox(
                "Selecciona una comuna",
                df_live["comuna_nombre"].sort_values(),
                index=0,
            )
            com_row = df_live[
                df_live["comuna_nombre"] == selected_comuna
            ].iloc[0]

            st.markdown(f"""
            * **Distribuidora:** `{com_row["empresa"]}`
            * **Clientes Sin Luz:** `{int(com_row["clientes_sin_suministro"]):,}` / `{int(com_row["clientes_totales"]):,}` ({float(com_row["tasa_afectacion"]):.2%})
            * **Estado:** {"🚨 CORTE CRÍTICO" if bool(com_row["es_corte_critico"]) else "✅ Suministro Operativo"}
            * **Estación Climática Más Cercana:** `{com_row["estacion_mas_cercana"]}` (a `{com_row["distancia_estacion_km"]} km`)
            * **Viento Estimado (IDW):** `{com_row["viento_kmh"]} km/h` (Ráfaga: `{com_row["rafaga_max_kmh"]} km/h`)
            * **Lluvia Estimada (IDW):** `{com_row["precip_acumulada_mm"]} mm`
            * **Riesgo Teórico Predicho (Modelo):** `{com_row["prob_predicha_modelo"]:.1%}`
            """)

    with tab_tabla:
        st.subheader("Monitor de las 52 Comunas de la RM")
        cols_show = [
            "comuna_nombre",
            "empresa",
            "clientes_sin_suministro",
            "clientes_totales",
            "tasa_afectacion",
            "es_corte_critico",
            "estacion_mas_cercana",
            "rafaga_max_kmh",
            "precip_acumulada_mm",
        ]
        df_display = (
            df_live[cols_show]
            .sort_values("clientes_sin_suministro", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            df_display.style.format(
                {
                    "clientes_sin_suministro": "{:,}",
                    "clientes_totales": "{:,}",
                    "tasa_afectacion": "{:.2%}",
                    "rafaga_max_kmh": "{:.1f}",
                    "precip_acumulada_mm": "{:.1f}",
                }
            ),
            use_container_width=True,
            height=400,
        )

    with tab_diagnostico:
        st.subheader("Comparación: Telemetría Real vs. Riesgo GLM Estimado")
        st.caption(
            "Verifica la calibración del modelo ante las condiciones climáticas del momento."
        )

        fig_diag = px.scatter(
            df_live,
            x="prob_predicha_modelo",
            y="tasa_afectacion",
            text="comuna_nombre",
            color="empresa",
            labels={
                "prob_predicha_modelo": "Probabilidad Predicha por Modelo GLM",
                "tasa_afectacion": "Tasa Real de Clientes Sin Luz (SEC)",
            },
        )
        fig_diag.update_traces(textposition="top center")
        fig_diag.update_layout(height=400)
        st.plotly_chart(fig_diag, use_container_width=True)

# --- FOOTER Y REPRODUCIBILIDAD ---
st.markdown("---")
st.caption(
    "⚡ **Gridbreak Chile** | Desarrollado con Python 3.12, uv, Polars, Statsmodels, GeoPandas y Streamlit. "
    "Código 100% reproducible bajo licencia MIT."
)
