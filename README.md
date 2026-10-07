# ⚡ Gridbreak Chile: El Umbral del Apagón

> **Análisis Causal y Modelamiento de Fragilidad Eléctrica ante Eventos Climáticos en la Región Metropolitana**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/pytest-dev/pytest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 1. Tesis Analítica y Narrativa Contrarian

Frente a temporales invernales y estivales en Santiago de Chile, los comunicados de las empresas distribuidoras suelen atribuir las interrupciones masivas a **"fuerza mayor climática"**, intensidades meteorológicas inéditas o caída de ramas y árboles.

**Nuestra Hipótesis Cuantitativa:**  
La intensidad climática (lluvia en mm y ráfagas en km/h) no es la causa raíz del colapso, sino el detonante de una **marcada asimetría estructural preexistente**.
* Las comunas vulnerables de la periferia y poniente colapsan bajo umbrales pluviométricos y eólicos sustancialmente menores (**12-15 mm de lluvia y ráfagas de 45-55 km/h**).
* Las comunas de altos ingresos resisten hasta más de **45 mm y ráfagas sobre 80 km/h**.
* Esta brecha se sostiene incluso aislando el efecto de la masa arbórea comunal y controlando por la empresa distribuidora concesionaria (Enel vs. CGE).

---

## 📐 2. Formulación Matemática

### 2.1 Modelo Lineal Generalizado (GLM Logit Bivariado con Interacciones)
Modelamos la probabilidad de colapso crítico comunal ($Y = 1$, definido como $\ge 5\%$ de clientes desconectados simultáneamente):

$$\text{logit}(P(Y(i, t) = 1)) = \beta_0 + \beta_1 R(i, t) + \beta_2 W(i, t) + \beta_3 \text{NSE}(i) + \beta_4 (R \cdot \text{NSE}) + \beta_5 (W \cdot \text{NSE}) + \beta_6 \text{Empresa}_{\text{CGE}}(i) + \gamma^T X(i)$$

* $R(i, t)$: Precipitación acumulada (mm).
* $W(i, t)$: Ráfaga máxima de viento (km/h).
* $\text{NSE}(i)$: Nivel socioeconómico estandarizado.
* $\beta_4, \beta_5$: Términos de interacción cruzada. Coeficientes negativos estadísticamente significativos confirman que un mayor NSE reduce la susceptibilidad marginal ante lluvia y viento.
* $\beta_6$: Control por concesionaria (Enel vs. CGE).

### 2.2 Derivación Analítica del Umbral Crítico ($R_{50}$)
Para un nivel de viento basal $W_0$, el umbral exacto de lluvia necesario para alcanzar un 50% de probabilidad de colapso es:

$$R_{50}(i \mid W_0) = -\frac{\beta_0 + \beta_2 W_0 + \beta_3 \text{NSE}(i) + \beta_5 (W_0 \cdot \text{NSE}(i)) + \beta_6 \text{Empresa}(i) + \gamma^T X(i)}{\beta_1 + \beta_4 \text{NSE}(i)}$$

---

## 🏗️ 3. Estructura del Repositorio

```text
gridbreak-cl/
├── config/
│   └── settings.yaml          # Configuración geográfica y umbrales de modelamiento
├── data/
│   ├── raw/                   # Datos crudos de SEC, DMC y CASEN/Censo
│   └── processed/
│       └── benchmark_temporales_2024.parquet # Dataset panel horario RM (Temporales 2024)
├── docs/
│   └── El Umbral del Apagón - Blueprint y Especificación Técnica.md
├── src/gridbreak_cl/
│   ├── __init__.py
│   ├── config.py              # Esquemas y validación con Pydantic
│   ├── etl/
│   │   ├── historical_seed.py # Generador y persistencia de datasets benchmark
│   │   └── sec_collector.py   # Colector live de snapshots SEC
│   └── models/
│       └── fragility_curves.py# Curvas y superficies de fragilidad (GLM)
├── app/
│   └── streamlit_app.py       # Simulador interactivo en Streamlit
├── tests/
│   ├── test_etl.py            # Pruebas de esquemas y generación de datos
│   └── test_models.py         # Pruebas de coherencia física y convergencia
├── pyproject.toml             # Gestión moderna de dependencias (uv)
└── README.md
```

---

## 🚀 4. Inicio Rápido (Quickstart)

Este proyecto está construido con [uv](https://github.com/astral-sh/uv), el gestor de paquetes ultrarrápido para Python.

### Requisitos previos
* Python 3.12+
* `uv` instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh` o `brew install uv`)

### Instalación y Ejecución

```bash
# 1. Clonar el repositorio
git clone https://github.com/your-username/gridbreak-cl.git
cd gridbreak-cl

# 2. Sincronizar entorno virtual y dependencias
uv sync

# 3. Ejecutar las pruebas unitarias
uv run pytest -v

# 4. Lanzar la aplicación interactiva de Streamlit
uv run streamlit run app/streamlit_app.py
```

---

## 🧪 5. Verificación de Calidad y Tipado Estricto

El repositorio se adhiere a estándares de producción:

```bash
# Formateo y linter ultra-rápido con Ruff
uv run ruff check .
uv run ruff format .

# Chequeo estático de tipos con Mypy
uv run mypy src
```

---

## 📊 6. Storytelling y Aplicación Interactiva

La aplicación incluye:
1. **Simulador de Estrés Meteorológico:** Sliders de lluvia acumulada y ráfagas con mapa coroplético instantáneo de la RM.
2. **Comparador Frente a Frente:** Análisis directo de asimetría (*ej. Cerro Navia vs. Las Condes*).
3. **Superficie 3D de Fragilidad:** Modelamiento conjunto de probabilidad en función de lluvia y viento.

---

## 📜 Licencia

Distribuido bajo la Licencia MIT. Consulta `LICENSE` para más detalles.
