# Especificación Técnica y Blueprint de Proyecto: El Umbral del Apagón

## Análisis Causal y Modelamiento de Fragilidad Eléctrica ante Eventos Climáticos en la Región Metropolitana

- **Identificador de Candidato:** 2026-10-07-008
- **Nombre del Repositorio:** gridbreak-cl / el-umbral-del-apagon
- **Objetivo:** Portafolio Técnico Senior (GitHub) y Storytelling de Alto Impacto (LinkedIn)
- **Entorno de Ejecución:** Modern Python Stack administrado vía `uv` (Python 3.12+)

---

## 1. Resumen Ejecutivo y Tesis Analítica

### 1.1 Narrativa Convencional vs. Hipótesis Contrarian
* **Narrativa Corporativa / Oficial:** Frente a eventos meteorológicos en la Región Metropolitana, las empresas distribuidoras y los comunicados iniciales suelen atribuir las interrupciones masivas a "fuerza mayor", caídas imprevistas del arbolado público e intensidades meteorológicas inéditas.
* **Hipótesis Cuantitativa Contrarian:** La intensidad del temporal (medida en mm acumulados de lluvia y velocidad de ráfagas en km/h) no es la causa raíz del colapso, sino el detonante de una marcada fragilidad estructural preexistente. Las comunas de menores ingresos y zonas periféricas colapsan bajo umbrales meteorológicos sustancialmente menores (10-15 mm de lluvia, ráfagas de 40-50 km/h) que las comunas de mayores recursos (>40 mm, ráfagas >80 km/h), aun aislando el efecto del arbolado urbano, la empresa distribuidora concesionaria y la exposición de la red.

### 1.2 Objetivo del Proyecto
Construir un repositorio de nivel Senior Data Scientist & Data Engineer con reproducibilidad absoluta, tipado estricto, suite de pruebas unitarias y empaquetamiento moderno que permita:
1. Extraer y consolidar series de tiempo de clientes desconectados de la SEC (Superintendencia de Electricidad y Combustibles) y variables meteorológicas de la DMC (Dirección Meteorológica de Chile), con arquitectura dual (Benchmark Replay 2024 + Live Collector).
2. Integrar covariables socioeconómicas (IPS/CASEN, Censo), infraestructura urbana (densidad de clientes por km de red, proxies de cableado aéreo vs. soterrado) y vegetación/arbolado (SIEDU / NDVI satelital).
3. Ajustar modelos de **Curvas y Superficies de Fragilidad** (GLM Logístico Multivariable con interacción Lluvia × Viento × NSE y control por Concesionaria Enel/CGE) y **Modelos de Supervivencia** (Cox Proportional Hazards) para cuantificar el riesgo relativo instantáneo.
4. Desarrollar una aplicación interactiva en Streamlit (Simulador de Estrés Climático) y visualizaciones geoespaciales publicables en LinkedIn que posicionen autoridad analítica.

---

## 2. Arquitectura del Proyecto y Estándares de Ingeniería

El proyecto se estructura bajo estándares de producción en Python: gestión hermética con `uv`, tipado estricto con `mypy`, formateo y linting con `ruff`, y pruebas unitarias con `pytest`.

### 2.1 Árbol de Directorios

```text
gridbreak-cl/
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   │   ├── sec/
│   │   ├── dmc/
│   │   └── socioeconomico/
│   ├── interim/
│   └── processed/
│       └── benchmark_temporales_2024.parquet
├── notebooks/
│   ├── 01_eda_precipitaciones_viento_vs_cortes.ipynb
│   └── 02_modelamiento_curvas_fragilidad.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── sec_collector.py
│   │   ├── dmc_client.py
│   │   ├── historical_seed.py
│   │   └── spatial_join.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fragility_curves.py
│   │   └── survival_analysis.py
│   └── visualization/
│       ├── __init__.py
│       ├── static_charts.py
│       └── map_generator.py
├── app/
│   └── streamlit_app.py
└── tests/
    ├── __init__.py
    ├── test_etl.py
    ├── test_features.py
    └── test_models.py
```

---

## 3. Configuración del Entorno de Desarrollo con `uv`

### 3.1 Inicialización del Proyecto
Comandos de configuración:
```bash
# Inicializar repositorio con uv y Python 3.12
uv init gridbreak-cl
cd gridbreak-cl
uv python pin 3.12

# Dependencias del pipeline analítico
uv add polars pandas geopandas shapely requests beautifulsoup4 pydantic pyyaml \
       scikit-learn statsmodels lifelines plotly folium streamlit

# Herramientas de calidad, testing y linting
uv add --dev pytest pytest-cov ruff mypy ipykernel
```

### 3.2 Archivo `pyproject.toml`
Configuración base con herramientas de control estático y calidad de código:
```toml
[project]
name = "gridbreak-cl"
version = "0.1.0"
description = "Modelamiento causal de fragilidad de la red eléctrica ante eventos climáticos en la RM"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "polars>=0.20.0",
    "pandas>=2.2.0",
    "geopandas>=0.14.0",
    "shapely>=2.0.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
    "pydantic>=2.6.0",
    "pyyaml>=6.0.1",
    "scikit-learn>=1.4.0",
    "statsmodels>=0.14.0",
    "lifelines>=0.28.0",
    "plotly>=5.19.0",
    "folium>=0.16.0",
    "streamlit>=1.32.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.3.0",
    "mypy>=1.9.0",
    "ipykernel>=6.29.0",
]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

---

## 4. Fuentes de Datos y Estrategia de Ingesta Dual

### 4.1 Estrategia Dual: Replay Histórico + Live Collector
Para asegurar reproducibilidad inmediata sin depender de la contingencia climática en vivo:
1. **Modo Benchmark / Replay Histórico:** Se integran eventos consolidados de la RM en `data/processed/benchmark_temporales_2024.parquet`:
   * **Temporal Junio 2024:** Lluvia continua intensa (>60 mm) con viento moderado (30-45 km/h).
   * **Temporal Agosto 2024:** Ráfagas destructivas de viento (>100-124 km/h en Quinta Normal y Pudahuel) y lluvia moderada (~25 mm).
2. **Modo Live Collector (`sec_collector.py`):** Colector de snapshots del portal de interrupciones de la SEC que persiste registros incrementales cuando hay contingencias activas.

### 4.2 Variables Meteorológicas (DMC)
* **Estaciones de Referencia RM:** Quinta Normal, Tobalaba, Pudahuel, La Florida, Talagante, Melipilla, San José de Maipo.
* **Métricas Clave:** Precipitación acumulada en 12h y 24h (`precip_acumulada_mm`), Velocidad media de viento (`viento_kmh`) y Ráfaga máxima registrada (`rafaga_max_kmh`).
* **Cruce Espacial:** Ponderación espacial sobre centroides comunales mediante interpolación inversa de distancia (IDW) o asignación por Voronoi implementada en `GeoPandas`.

### 4.3 Covariables Estructurales, Sociodemográficas y de Red
* **Nivel Socioeconómico (NSE):** Índice de Prioridad Social (IPS) o Ingreso Autónomo Promedio y Pobreza Multidimensional comunal (CASEN / Censo / BCN).
* **Distribuidora Eléctrica (`empresa`):** Variable de control categórica (Enel Distribución Chile vs. CGE).
* **Vulnerabilidad de Infraestructura:** Clasificación de densidad de red (clientes / km lineal de media tensión) y proporción estimada de cableado aéreo vs. soterrado.
* **Vegetación / Arbolado Urbano:** m² de área verde mantenida por habitante (SIEDU/INE) e índice de cobertura vegetal comunal (NDVI).

---

## 5. Formulación Matemática y Modelamiento

### 5.1 Definición de la Variable de Falla
Para la comuna $i$ en el intervalo de tiempo $t$:
$$\text{Rate}(i, t) = \frac{\text{Clientes Afectados}(i, t)}{\text{Clientes Totales}(i)}$$

Definimos el evento de **Colapso Crítico Comunal** como:
$$Y(i, t) = \begin{cases} 1 & \text{si } \text{Rate}(i, t) \ge \alpha \\ 0 & \text{en otro caso} \end{cases}$$
con $\alpha = 0.05$ (más del 5% de los clientes comunales desconectados simultáneamente).

### 5.2 Curvas y Superficies de Fragilidad (GLM Bivariado con Interacciones y Control)
Modelamos la probabilidad de falla mediante un GLM Binomial (logit) con dos estresores climáticos (lluvia y ráfaga de viento), nivel socioeconómico y control por empresa distribuidora:

$$\text{logit}(P(Y(i, t) = 1)) = \beta_0 + \beta_1 R(i, t) + \beta_2 W(i, t) + \beta_3 \text{NSE}(i) + \beta_4 (R \cdot \text{NSE}) + \beta_5 (W \cdot \text{NSE}) + \beta_6 \text{Empresa}_{\text{CGE}}(i) + \gamma^T X(i)$$

donde:
* $R(i, t)$: Precipitación acumulada en 12h (mm).
* $W(i, t)$: Ráfaga máxima de viento (km/h).
* $\text{NSE}(i)$: Nivel socioeconómico comunal estandarizado (mayor valor = mayor nivel de ingresos).
* $\beta_4, \beta_5$: Coeficientes de interacción. Valores negativos estadísticamente significativos demuestran que a igual nivel de lluvia y viento, una comuna de mayor NSE presenta menor fragilidad.
* $\beta_6$: Coeficiente de control por concesionaria (Enel vs. CGE).
* $X(i)$: Densidad arbórea y densidad de red.

#### Umbrales Críticos de Falla ($R_{50}$ y $W_{50}$)
Para un nivel de viento basal $W_0$, el umbral de lluvia para 50% de probabilidad de corte masivo es:
$$R_{50}(i \mid W_0) = -\frac{\beta_0 + \beta_2 W_0 + \beta_3 \text{NSE}(i) + \beta_5 (W_0 \cdot \text{NSE}(i)) + \beta_6 \text{Empresa}(i) + \gamma^T X(i)}{\beta_1 + \beta_4 \text{NSE}(i)}$$

### 5.3 Modelo de Supervivencia: Tiempo hasta el Colapso
Sea $T(i)$ el tiempo transcurrido desde el inicio del temporal hasta la primera interrupción masiva ($Y = 1$). Ajustamos un modelo de riesgos proporcionales de Cox:

$$\lambda(t \mid Z(i)) = \lambda_0(t) \cdot \exp\left(\theta_1 R_{\text{rate}}(i) + \theta_2 W_{\text{max}}(i) + \theta_3 \text{NSE}(i) + \theta_4 \text{Arbolado}(i) + \theta_5 \text{Empresa}(i)\right)$$

El Hazard Ratio (HR) permite concluir cuantitativamente:
> *"A igualdad de viento y lluvia, las comunas del tercil más vulnerable presentan un riesgo instantáneo $\text{HR} = \exp(-\theta_3 \cdot \Delta \text{NSE})$ significativamente mayor de sufrir colapso de red."*

---

## 6. Módulos de Código en Python

### 6.1 Esquemas y Validación de Datos (`src/config.py`)
```python
"""Módulo de configuración y esquemas de validación con Pydantic."""

from datetime import datetime
from pydantic import BaseModel, Field


class SECCutRecord(BaseModel):
    timestamp: datetime
    comuna_id: str
    comuna_nombre: str
    empresa: str
    clientes_sin_suministro: int = Field(ge=0)
    clientes_totales: int = Field(gt=0)

    @property
    def tasa_afectacion(self) -> float:
        return self.clientes_sin_suministro / self.clientes_totales


class WeatherRecord(BaseModel):
    timestamp: datetime
    estacion_codigo: str
    precipitacion_mm: float = Field(ge=0.0)
    viento_kmh: float = Field(ge=0.0)
    rafaga_max_kmh: float = Field(ge=0.0)


class ComunaFeatures(BaseModel):
    comuna_id: str
    comuna_nombre: str
    empresa_principal: str
    nse_score: float  # Estandarizado (media 0, std 1)
    ingreso_autonomo_promedio: float = Field(ge=0.0)
    pobreza_multidimensional: float = Field(ge=0.0, le=1.0)
    arbolado_m2_hab: float = Field(ge=0.0)
    red_aerea_km_ratio: float = Field(ge=0.0, le=1.0)
```

### 6.2 Modelamiento de Fragilidad (`src/models/fragility_curves.py`)
```python
"""Ajuste de modelos logísticos e inferencia de curvas de fragilidad comunal."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults


class FragilityModel:
    def __init__(self) -> None:
        self.model_results: GLMResults | None = None

    def fit(self, df: pd.DataFrame) -> GLMResults:
        """Ajusta un modelo binomial:
        logit(P) ~ precip_acumulada_mm + rafaga_max_kmh + nse_score
                   + precip_x_nse + rafaga_x_nse + es_cge + arbolado_m2_hab
        """
        df = df.copy()
        df["precip_x_nse"] = df["precip_acumulada_mm"] * df["nse_score"]
        df["rafaga_x_nse"] = df["rafaga_max_kmh"] * df["nse_score"]
        df["es_cge"] = (df["empresa"] == "CGE").astype(float)

        feature_cols = [
            "precip_acumulada_mm",
            "rafaga_max_kmh",
            "nse_score",
            "precip_x_nse",
            "rafaga_x_nse",
            "es_cge",
            "arbolado_m2_hab",
        ]
        X = df[feature_cols]
        X = sm.add_constant(X)
        y = df["es_corte_critico"].astype(int)

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
        """Calcula la probabilidad predicha de corte masivo para un escenario climático."""
        if self.model_results is None:
            raise ValueError("El modelo debe ser ajustado previamente con fit().")

        params = self.model_results.params
        logit = (
            params["const"]
            + params["precip_acumulada_mm"] * precip_mm
            + params["rafaga_max_kmh"] * rafaga_kmh
            + params["nse_score"] * nse_score
            + params["precip_x_nse"] * (precip_mm * nse_score)
            + params["rafaga_x_nse"] * (rafaga_kmh * nse_score)
            + params["es_cge"] * es_cge
            + params["arbolado_m2_hab"] * arbolado_m2_hab
        )
        return float(1.0 / (1.0 + np.exp(-logit)))
```

---

## 7. Pruebas Unitarias Automatizadas (`tests/test_models.py`)

```python
"""Validación de convergencia y coherencia física del modelo de fragilidad."""

import numpy as np
import pandas as pd
import pytest
from src.models.fragility_curves import FragilityModel


@pytest.fixture
def synthetic_storm_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    precip = np.random.uniform(5, 70, n)
    rafaga = np.random.uniform(20, 110, n)
    nse = np.random.choice([-1.5, 0.0, 1.5], n)  # Vulnerable, Medio, Acomodado
    es_cge = np.random.choice([0.0, 1.0], n)
    arbolado = np.random.uniform(2, 12, n)

    # Probabilidad con mayor susceptibilidad en NSE bajo ante lluvia y viento
    logit = (
        -3.0
        + 0.08 * precip
        + 0.05 * rafaga
        - 0.80 * nse
        - 0.02 * (precip * nse)
        - 0.015 * (rafaga * nse)
        + 0.30 * es_cge
        + 0.04 * arbolado
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    corte = (np.random.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame(
        {
            "precip_acumulada_mm": precip,
            "rafaga_max_kmh": rafaga,
            "nse_score": nse,
            "empresa": np.where(es_cge == 1.0, "CGE", "ENEL"),
            "arbolado_m2_hab": arbolado,
            "es_corte_critico": corte,
        }
    )


def test_model_fit_and_vulnerability_gap(synthetic_storm_data: pd.DataFrame) -> None:
    model = FragilityModel()
    results = model.fit(synthetic_storm_data)
    assert results is not None
    assert "precip_x_nse" in results.params

    # A mismo temporal (30 mm de agua y 70 km/h de viento):
    prob_vulnerable = model.predict_probability(
        precip_mm=30.0, rafaga_kmh=70.0, nse_score=-1.5
    )
    prob_acomodado = model.predict_probability(
        precip_mm=30.0, rafaga_kmh=70.0, nse_score=1.5
    )

    # La comuna vulnerable debe tener mayor probabilidad de colapso
    assert prob_vulnerable > prob_acomodado
```

---

## 8. Aplicación Streamlit e Impacto en Portafolio

### 8.1 Simulador de Estrés Climático (`app/streamlit_app.py`)
* **Controles Interactivos:**
  * Sliders dobles: *Lluvia Acumulada (0 - 80 mm)* y *Ráfaga de Viento (0 - 120 km/h)*.
* **Salidas en Tiempo Real:**
  * Mapa coroplético interactivo de la RM (Folium/Plotly) coloreando cada comuna por su probabilidad estimada de colapso.
  * Comparador frente a frente de comunas (ej: *Cerro Navia vs. Vitacura* o *San Ramón vs. Las Condes*).
  * Gráficas de Superficie 3D y Curvas Sigmoides de Fragilidad.

### 8.2 Estrategia de Storytelling para LinkedIn (Data Newsjacking & Fact-Checking)
* **Enfoque Central:** Auditoría cuantitativa a la noticia y al relato oficial de las distribuidoras ("evento inédito de fuerza mayor" y "caída masiva de árboles").
* **Estructura del Post:** Contrastar punto por punto qué afirmaciones de los titulares de prensa son **CONTRADICHAS** por los datos (umbrales $W_{50}$ comunales ordinarios, irrelevancia del arbolado frente al tendido aéreo, asimetría de supervivencia en Cox con $\text{HR} = 0.33$) y cuáles son **CONFIRMADAS** (el rol cinético de las ráfagas máximas).
* **Evidencia Visual:** Carrusel de 4 láminas en `reports/figures/` con las curvas de fragilidad, curvas Kaplan-Meier, Forest Plot de Cox y brechas comunales $W_{50}$.
* **Documento Maestro:** Ver detalle completo, matriz de contraste y copias listas para publicar en [docs/linkedin_storytelling.md](file:///Users/sebastianfelipeurzuaborquez/Proyectos/gridbreak-cl/docs/linkedin_storytelling.md).

