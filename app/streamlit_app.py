"""Gridbreak Chile: El Umbral del Apagón.

Aplicación interactiva y Simulador de Estrés Climático para la Región Metropolitana.
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


# Cargar datos en caché
df_comunas, df_storm, model = load_data_and_model()

# --- SIDEBAR: CONTROLES DEL SIMULADOR ---
st.sidebar.image(
    "https://img.icons8.com/fluency/96/lightning-bolt.png",
    width=64,
)
st.sidebar.title("⚡ Gridbreak Chile")
st.sidebar.caption("Simulador de Fragilidad Eléctrica y Riesgo Climático RM")

st.sidebar.markdown("---")
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

# --- CABECERA PRINCIPAL ---
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
with st.expander("🔬 Evidencia Econométrica y Diagnóstico del Modelo GLM", expanded=False):
    st.markdown("""
    El modelo ajusta una regresión logística binomial con **términos de interacción cruzada**,
    control por **empresa concesionaria** y control por **exposición física de cableado aéreo** vs. subterráneo:
    """)

    metrics = model.get_metrics()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Pseudo R² (McFadden)", f"{metrics['pseudo_r2_mcfadden']:.3f}")
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

# --- FOOTER Y REPRODUCIBILIDAD ---
st.markdown("---")
st.caption(
    "⚡ **Gridbreak Chile** | Desarrollado con Python 3.12, uv, Polars, Statsmodels, GeoPandas y Streamlit. "
    "Código 100% reproducible bajo licencia MIT."
)
