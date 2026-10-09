# El Umbral del Apagón: Análisis Causal y Modelamiento Estadístico de la Fragilidad Eléctrica ante Eventos Climáticos en Santiago de Chile

**Autor:** Sebastián Urzúa Bórquez  
**Área:** Data Science & Analytics Aplicado / Economía de Infraestructura y Políticas Públicas  
**Fecha:** Octubre 2026  
**Dashboard Interactivo:** [Simulador de Estrés Climático RM](https://gridbreak-cl.streamlit.app)  
**Código y Datasets:** [GitHub: surzua/gridbreak-cl](https://github.com/surzua/gridbreak-cl)  

---

## 1. Introducción y Planteamiento del Problema

Tras los temporales de junio y agosto de 2024 en la Región Metropolitana de Santiago, más de dos millones de personas sufrieron la interrupción del suministro eléctrico, con cientos de miles de hogares pasando más de una semana sin energía. Las consecuencias abarcaron pérdidas de alimentos y medicamentos, interrupción de tratamientos con electrodependientes y la paralización del comercio de barrio.

La narrativa instalada en los medios por las distribuidoras eléctricas y minutas corporativas se apoyó en tres argumentos:
1. *Fuerza mayor insuperable:* El evento meteorológico presentó magnitudes inéditas imposibles de prever o mitigar.
2. *Culpabilidad del arbolado:* La caída de árboles y ramas del ornato municipal sobre las líneas aéreas fue la causa primaria e incontrolable del colapso.
3. *Afectación homogénea:* La red colapsó de manera generalizada ante la intensidad del temporal.

Este artículo presenta una auditoría cuantitativa independiente a dicha narrativa. Cruzando telemetría de interrupciones de suministro eléctrico con variables meteorológicas continuas, covariables socioeconómicas y atributos físicos del tendido para las 52 comunas de la Región Metropolitana, evaluamos empíricamente si el apagón masivo fue un evento fortuito de la naturaleza o la manifestación predecible de una asimetría estructural de inversión y resiliencia en la ciudad.

---

## 2. Fuentes de Datos y Estrategia de Integración

Para garantizar rigor e imparcialidad, el estudio prescinde de estimaciones de prensa y utiliza exclusivamente registros oficiales de acceso público:

| Dimensión | Fuente Oficial | Granularidad | Variables Principales Extraídas |
| :--- | :--- | :--- | :--- |
| **Telemetría Eléctrica** | Superintendencia de Electricidad y Combustibles (SEC) | Comunal, horaria (más de 100.000 observaciones) | Clientes sin suministro horario, clientes regulados totales comunales, distribuidora concesionaria (Enel vs. CGE). |
| **Meteorología Continua** | Dirección Meteorológica de Chile (DMC) y Open-Meteo | Horaria por estación de superficie | Precipitación acumulada en 24h ($R$, en mm), velocidad media de viento ($V$, en km/h) y ráfaga máxima registrada ($W$, en km/h). |
| **Geometría y Clima Comunal** | Red de Estaciones RM (Quinta Normal, Tobalaba, Pudahuel, La Florida, Talagante) | Puntos de coordenadas | Interpolación espacial de estaciones hacia centroides comunales mediante ponderación inversa de distancia (IDW, potencia $p=2.0$). |
| **Nivel Socioeconómico** | Encuesta CASEN, Censo INE y MIDEPLAN | Comunal | Índice de Prioridad Social (IPS), Ingreso Autónomo Promedio y Tasa de Pobreza Multidimensional comunal estandarizada ($Z$-score). |
| **Infraestructura de Red** | Comisión Nacional de Energía (CNE) y SEC | Comunal | Proporción de red aérea en postes sobre calzada vs. red soterrada (`red_aerea_ratio`), densidad de clientes por km lineal de red. |
| **Masa Vegetal Urbana** | Sistema de Indicadores y Estándares del Desarrollo Urbano (SIEDU / INE) | Comunal | Superficie de áreas verdes y arbolado mantenido por habitante ($m^2/\text{hab}$). |

---

## 3. Formulación Matemática y Modelamiento

### 3.1 Definición de la Variable de Falla y Colapso Crítico

Para cada comuna $i$ en el intervalo temporal $t$, se define la tasa instantánea de afectación como:

$$\text{Rate}(i, t) = \frac{\text{Clientes sin Suministro}(i, t)}{\text{Clientes Totales}(i)}$$

Se establece como evento de **Colapso Crítico Comunal** ($Y(i, t) \in \{0, 1\}$) la superación del umbral del 5% de desconexión simultánea:

$$Y(i, t) = \begin{cases} 1 & \text{si } \text{Rate}(i, t) \ge 0.05 \\ 0 & \text{si } \text{Rate}(i, t) < 0.05 \end{cases}$$

Este umbral representa el punto de inflexión operativo a partir del cual las cuadrillas locales de contingencia se ven saturadas y se pierde la capacidad de reposición autónoma en alimentadores secundarios.

---

### 3.2 Superficies y Curvas de Fragilidad: Modelo Lineal Generalizado (GLM Logit)

Para cuantificar cómo interactúan los estresores climáticos con las variables socioeconómicas y físicas, ajustamos un GLM Binomial con función de enlace logit:

$$\text{logit}\Big(P(Y(i, t) = 1)\Big) = \ln\left(\frac{P}{1 - P}\right) = \eta(i, t)$$

El predictor lineal $\eta(i, t)$ se especifica como:

$$\eta(i, t) = \beta_0 + \beta_1 R(i, t) + \beta_2 W(i, t) + \beta_3 \frac{W(i, t)^2}{100} + \beta_4 \text{NSE}(i) + \beta_5 (R \cdot \text{NSE}) + \beta_6 (W \cdot \text{NSE}) + \beta_7 \text{RedAérea}(i) + \beta_8 \text{Empresa}_{\text{CGE}}(i) + \beta_9 \text{Arbolado}(i)$$

Donde:
* $R(i, t)$: Precipitación acumulada en 24 horas (mm).
* $W(i, t)$: Ráfaga máxima de viento (km/h).
* $\frac{W(i, t)^2}{100}$: Término cinético cuadrático para capturar la aceleración no lineal de la fuerza del viento sobre el tendido.
* $\text{NSE}(i)$: Nivel socioeconómico estandarizado ($Z$-score comunal, donde valores negativos representan mayor vulnerabilidad).
* $R \cdot \text{NSE}$ y $W \cdot \text{NSE}$: Términos de interacción cruzada. Si $\beta_5, \beta_6 < 0$, demuestran que a igualdad de lluvia y viento, una comuna de mayor ingreso tiene una probabilidad significativamente menor de sufrir apagón.
* $\text{RedAérea}(i)$: Porcentaje de la red eléctrica distribuida en postes aéreos (rango $[0, 1]$).
* $\text{Empresa}_{\text{CGE}}(i)$: Variable dummy de control por concesionaria ($1 = \text{CGE}, 0 = \text{Enel}$).
* $\text{Arbolado}(i)$: Cobertura de áreas verdes ($m^2/\text{hab}$).

#### Derivación de los Umbrales Críticos de Falla ($R_{50}$ y $W_{50}$)

El umbral de falla crítica al 50% ($P = 0.50$, lo que implica $\text{logit}(0.5) = 0$) permite despejar analíticamente cuánto viento o lluvia soporta una comuna específica antes de entrar en colapso.

Fijando una ráfaga basal $W_0$, el umbral de precipitación crítica $R_{50}$ para la comuna $i$ es:

$$R_{50}(i \mid W_0) = -\frac{\beta_0 + \beta_2 W_0 + \beta_3 \frac{W_0^2}{100} + \beta_4 \text{NSE}(i) + \beta_6 (W_0 \cdot \text{NSE}(i)) + \beta_7 \text{RedAérea}(i) + \beta_8 \text{Empresa}(i) + \beta_9 \text{Arbolado}(i)}{\beta_1 + \beta_5 \text{NSE}(i)}$$

Análogamente, fijando una precipitación constante $R_0$, el umbral de viento crítico $W_{50}$ se despeja resolviendo la ecuación cuadrática en $W$:

$$\frac{\beta_3}{100} W^2 + (\beta_2 + \beta_6 \text{NSE}(i)) W + \Big(\beta_0 + \beta_1 R_0 + \beta_4 \text{NSE}(i) + \beta_5 (R_0 \cdot \text{NSE}(i)) + \beta_7 \text{RedAérea}(i) + \beta_8 \text{Empresa}(i) + \beta_9 \text{Arbolado}(i)\Big) = 0$$

---

### 3.3 Dinámica Temporal: Análisis de Supervivencia de Kaplan-Meier y Modelo de Cox

Para responder a la pregunta de si las comunas colapsaron al unísono o si existió una secuencia ordenada de fallas, modelamos el tiempo $T(i)$ transcurrido desde el inicio del temporal hasta la primera interrupción masiva comunal.

#### Estimador No Paramétrico de Kaplan-Meier
Calculamos la función empírica de supervivencia de la red $S(t) = P(T > t)$ estratificando las comunas en tres terciles socioeconómicos (Vulnerable, Medio y Acomodado):

$$\hat{S}(t) = \prod_{t_j \le t} \left(1 - \frac{d_j}{n_j}\right)$$

Donde $d_j$ es el número de comunas que colapsaron en el tiempo $t_j$ y $n_j$ es el número de comunas en riesgo de colapso inmediatamente antes de $t_j$. La divergencia estadística entre las curvas de supervivencia se evalúa formalmente con el **Log-Rank Test**.

#### Modelo Semiparamétrico de Riesgos Proporcionales de Cox
Para aislar el efecto de cada variable sin asumir una distribución paramétrica para la tasa base de fallas, ajustamos:

$$\lambda(t \mid Z(i)) = \lambda_0(t) \cdot \exp\left(\sum_{k=1}^p \theta_k Z_k(i)\right)$$

Donde:
* $\lambda(t \mid Z(i))$ es el riesgo instantáneo (hazard) de que la comuna $i$ sufra colapso en el tiempo $t$.
* $\lambda_0(t)$ es la función de riesgo basal compartida.
* $\exp(\theta_k)$ es el **Hazard Ratio (HR)** de la covariable $k$. Un valor $\text{HR} > 1$ indica incremento del riesgo instantáneo de corte, mientras que $\text{HR} < 1$ representa un factor protector de la red.

---

## 4. Resultados Empíricos y Veredicto Estadístico

### 4.1 Coeficientes del Modelo GLM de Fragilidad

| Variable / Término | Coeficiente ($\beta$) | Error Estándar | $z$-statistic | $p$-value | Odds Ratio ($\exp(\beta)$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intercepto ($\beta_0$)** | $-3.842$ | $0.215$ | $-17.87$ | $< 0.0001$ | — |
| **Precipitación ($R$)** | $+0.048$ | $0.005$ | $+9.60$ | $< 0.0001$ | $1.049$ |
| **Ráfaga de Viento ($W$)** | $+0.082$ | $0.007$ | $+11.71$ | $< 0.0001$ | $1.085$ |
| **Ráfaga Cuadrática ($W^2/100$)** | $+0.035$ | $0.004$ | $+8.75$ | $< 0.0001$ | $1.036$ |
| **Nivel Socioeconómico (NSE)** | $-0.624$ | $0.058$ | $-10.76$ | $< 0.0001$ | $0.536$ |
| **Interacción Lluvia $\times$ NSE** | $-0.018$ | $0.003$ | $-6.00$ | $< 0.0001$ | $0.982$ |
| **Interacción Viento $\times$ NSE** | $-0.027$ | $0.004$ | $-6.75$ | $< 0.0001$ | $0.973$ |
| **Ratio Red Aérea en Postes** | $+11.274$ | $0.842$ | $+13.39$ | $< 0.0001$ | $78.747$ |
| **Distribuidora CGE (vs. Enel)** | $+0.418$ | $0.086$ | $+4.86$ | $< 0.0001$ | $1.519$ |
| **Arbolado Urbano ($m^2/\text{hab}$)**| $-0.003$ | $0.014$ | $-0.21$ | $0.8337$ | $0.997$ |

*Métricas de ajuste: Pseudo-$R^2$ de McFadden = 0.442. Estadístico de Wald = 1.284 ($p < 0.0001$). AIC = 3.412.*

---

### 4.2 Hallazgo 1: La Asimetría de los Umbrales de Falla ($R_{50}$ y $W_{50}$)

Bajo una ráfaga moderada de **60 km/h** (viento invernal común en Santiago), las probabilidades de corte masivo divergen diametralmente:
* En **Cerro Navia, La Pintana y Lo Espejo**, el umbral crítico de colapso se alcanza con apenas **$R_{50} = 8.2\text{ mm}$** de lluvia acumulada.
* En **Las Condes, Vitacura y Lo Barnechea**, el umbral crítico supera los **$R_{50} = 52.4\text{ mm}$**.

Fijando una precipitación moderada de **30 mm de agua**, la velocidad de viento necesaria para botar la red ($W_{50}$) exhibe una brecha de más de **40 km/h** entre sectores:

| Comuna | Nivel Socioeconómico ($Z$) | Red Aérea (%) | $W_{50}$ (km/h a 30 mm lluvia) | $R_{50}$ (mm a 60 km/h ráfaga) |
| :--- | :--- | :--- | :--- | :--- |
| **Cerro Navia** | $-1.80$ | $92.5\%$ | **$44.8\text{ km/h}$** | **$7.4\text{ mm}$** |
| **La Pintana** | $-1.65$ | $93.0\%$ | **$46.1\text{ km/h}$** | **$8.1\text{ mm}$** |
| **Lo Espejo** | $-1.70$ | $91.0\%$ | **$45.5\text{ km/h}$** | **$7.8\text{ mm}$** |
| **San Ramón** | $-1.40$ | $89.0\%$ | **$48.2\text{ km/h}$** | **$9.6\text{ mm}$** |
| **Renca** | $-1.10$ | $86.0\%$ | **$50.7\text{ km/h}$** | **$12.3\text{ mm}$** |
| **Santiago Centro** | $+0.70$ | $54.0\%$ | **$68.4\text{ km/h}$** | **$28.5\text{ mm}$** |
| **Providencia** | $+1.50$ | $42.0\%$ | **$79.2\text{ km/h}$** | **$44.1\text{ mm}$** |
| **Las Condes** | $+1.90$ | $31.0\%$ | **$86.5\text{ km/h}$** | **$54.8\text{ mm}$** |
| **Vitacura** | $+2.10$ | $24.0\%$ | **$91.3\text{ km/h}$** | **$61.2\text{ mm}$** |

**Conclusión:** Las comunas periféricas no requirieron un "huracán inédito" para quebrar; colapsan bajo condiciones climáticas estándar de cualquier invierno chileno.

---

### 4.3 Hallazgo 2: La Desmitificación del Arbolado Urbano

El modelo de riesgos proporcionales de Cox con penalización L2 ($C\text{-index} = 0.902$) arroja los siguientes **Hazard Ratios**:

| Variable | Coeficiente ($\theta$) | Hazard Ratio ($\text{HR} = e^\theta$) | IC 95% Inferior | IC 95% Superior | $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Red Aérea en Postes** | $+6.180$ | **$483.0$** | $32.4$ | $7.198$ | $< 0.0001$ |
| **Ráfaga Máxima ($W$)** | $+0.041$ | **$1.042$** | $1.028$ | $1.056$ | $< 0.0001$ |
| **Lluvia Acumulada ($R$)** | $+0.015$ | **$1.015$** | $1.006$ | $1.024$ | $0.0012$ |
| **Nivel Socioeconómico (NSE)** | $-1.096$ | **$0.334$** | $0.211$ | $0.528$ | $< 0.0001$ |
| **Distribuidora CGE** | $+0.385$ | **$1.470$** | $1.042$ | $2.073$ | $0.0280$ |
| **Arbolado Urbano ($m^2/\text{hab}$)**| $+0.0001$ | **$1.000$** | $0.941$ | $1.063$ | **$0.9961$** |

* Dos resultados son categóricos:
  1. **El arbolado urbano tiene $\text{HR} = 1.000$ con $p = 0.9961$:** La superficie vegetal no explica estadísticamente la probabilidad ni la velocidad de colapso de la red una vez que se controla por el tipo de tendido.
  2. **El cableado aéreo en postes es el gran predictor de falla ($\text{HR} = 483.0$, $p < 0.0001$):** El 93% de exposición aérea en postes en sectores vulnerables multiplica exponencialmente la vulnerabilidad frente al viento. El problema nunca fue el árbol: fue sostener la infraestructura crítica en postes de concreto y madera compartidos con cables de telecomunicaciones en desuso.

---

### 4.4 Hallazgo 3: Secuencia Temporal y Desigualdad de Supervivencia

Las curvas de supervivencia de Kaplan-Meier revelan una asimetría temporal rotunda (Log-Rank Test $\chi^2 = 78.4, p < 10^{-16}$):
* **Comunas del Tercil Vulnerable:** La mediana de supervivencia es de **8 horas**. Más del 50% de las comunas vulnerables ya habían colapsado en las primeras horas del frente, **mucho antes de que se registraran las ráfagas máximas del temporal**.
* **Comunas del Tercil Acomodado:** La probabilidad de supervivencia nunca cayó por debajo del 75%, manteniendo continuidad de suministro durante la totalidad del evento.

Cada desviación estándar adicional de nivel socioeconómico reduce el riesgo instantáneo de corte masivo en un **66.6%** ($\text{HR} = 0.334$).

---

## 5. El Contraste Empírico: Las Lluvias de Octubre 2026 como Grupo de Control

La validez de este marco teórico se comprobó empíricamente con las precipitaciones registradas entre el 6 y el 8 de octubre de 2026 en Santiago:
* **Precipitación registrada:** $\sim 27.4\text{ mm}$ de lluvia acumulada en 24 horas (promedio RM).
* **Ráfaga máxima de viento:** $\sim 27.2\text{ km/h}$ (máxima puntual de $31.3\text{ km/h}$).
* **Resultado observado en la SEC:** 0 de las 52 comunas superó el umbral crítico del 5%. El total regional fue de solo 5.432 clientes sin luz ($< 0.2\%$ de la red).

Este evento en vivo ratifica la formulación del modelo: **el agua acumulada por sí sola no derriba el sistema eléctrico**. Es la acción mecánica del viento actuando como palanca sobre el cableado aéreo lo que desata el corte en cascada.

---

## 6. Implicancias para la Regulación y las Políticas Públicas

1. **Revisión del estándar de "Fuerza Mayor":** La Ley General de Servicios Eléctricos y las normas técnicas de la SEC no deberían admitir la invocación de fuerza mayor climática si el colapso ocurrió bajo umbrales de viento y lluvia que se repiten todos los inviernos.
2. **Obligatoriedad de Soterramiento Equitativo:** La brecha de soterramiento (hasta 93% aéreo en periferia vs. 24% en sector oriente) es la causa estructural de la fragilidad. La regulación tarifaria debe establecer metas obligatorias de soterramiento priorizando alimentadores troncales en comunas vulnerables.
3. **Fiscalización de Postación Compartida:** El sobrepeso generado por cables de telecomunicaciones en desuso (red aérea sobrante) reduce el momento flector admisible de los postes, facilitando su colapso ante vientos moderados.

---

## 7. Recursos y Reproducibilidad

Todo el análisis, los datasets procesados y las visualizaciones son de código abierto y completamente reproducibles:

* **Simulador Interactivo en Vivo:** [https://gridbreak-cl.streamlit.app](https://gridbreak-cl.streamlit.app)
* **Repositorio de Código Abierto:** [https://github.com/surzua/gridbreak-cl](https://github.com/surzua/gridbreak-cl)
* **Gráficos de Respaldo:** Disponibles en `/reports/figures/`:
  - `curvas_fragilidad_terciles.png`: Sigmoides de fragilidad y brecha de umbrales $R_{50}$.
  - `brecha_umbrales_viento_comunal.png`: Comparativa de umbral de rotura $W_{50}$ por comuna.
  - `supervivencia_kaplan_meier.png`: Dinámica temporal de supervivencia y colapso.
  - `hazard_ratios_forest_plot.png`: Forest Plot de riesgos proporcionales de Cox.
