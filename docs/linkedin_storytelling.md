# 📢 Estrategia de Storytelling y Publicación Técnica: LinkedIn

> **Proyecto:** *El Umbral del Apagón — Modelamiento de Fragilidad Eléctrica y Riesgo Climático en la RM*  
> **Enfoque Central:** Data Newsjacking & Fact-Checking Cuantitativo: ¿Qué decían las noticias y comunicados oficiales sobre el apagón masivo vs. qué demuestran los datos empíricos de la SEC y la DMC?  
> **Perfil Objetivo:** Data Science Lead / Senior Data Scientist / Staff Analytics / Políticas Públicas y Energía.

---

## 🎯 Ángulo Editorial: De la Arquitectura de Software a la Auditoría de la Noticia

En LinkedIn, los posts técnicos que se limitan a enumerar librerías o herramientas de software (*"usé Python, Pydantic, Mypy y Ruff"*) generan poco engagement porque confunden **las herramientas** con **el valor del análisis**.

El verdadero valor de este proyecto es el **Newsjacking de Alto Impacto**: auditar de manera rigurosa, cuantitativa e imparcial una noticia de conmoción pública nacional (los temporales y apagones que afectaron a más de 2 millones de personas en Santiago).

### 📰 La Noticia y el Relato Oficial (Titulares de Prensa)
Tras los temporales de junio y agosto de 2024 en la Región Metropolitana, distribuidoras y minutas oficiales instalaron tres premisas en los medios:
1. *"Fue un evento climático inédito de fuerza mayor con ráfagas huracanadas impredecibles."*
2. *"La causa fundamental del colapso fue la caída imprevista del arbolado urbano sobre las líneas."*
3. *"La red colapsó de manera generalizada e inevitable ante la furia del temporal."*

### 📊 El Veredicto de los Datos
Al cruzar más de 100.000 registros de clientes desconectados de la SEC con telemetría meteorológica de la DMC mediante modelos econométricos (GLM Logit) y bioestadísticos (Supervivencia de Cox), **los datos contradicen frontalmente las dos primeras afirmaciones y matizan profundamente la tercera**.

---

## 📝 Copia Maestra para LinkedIn (Opción 1: Fact-Checking Titular vs. Datos)

```markdown
Titular de prensa vs. Datos duros: ¿Qué tan cierto fue que el gran apagón de Santiago se debió a "fuerza mayor" y a la "caída de árboles"? ⚡🇨🇱

Tras los temporales de 2024 que dejaron a más de 2 millones de personas sin luz en la Región Metropolitana —algunos durante más de una semana—, los comunicados corporativos y las minutas iniciales instalaron una narrativa unánime:
"Frente meteorológico inédito, ráfagas récord y caída masiva de ramas sobre el tendido (fuerza mayor)".

Pero cuando una crisis de servicios básicos se justifica apelando a la naturaleza, el deber de la ciencia de datos es auditar las afirmaciones contra la evidencia empírica.

Cruzamos más de 100.000 registros de telemetría de interrupciones de la SEC con observaciones meteorológicas horarias de la DMC y Open-Meteo para las 52 comunas de Santiago.

Ajustamos modelos de fragilidad física (GLM Logit Bivariado) y análisis de supervivencia temporal (Cox Proportional Hazards con penalización L2, Concordancia C = 0.90).

¿Se confirma o se contradice el relato público? Estos son los resultados:

❌ 1. CONTRADICHO: "Fue un temporal inédito que ningún sistema podía soportar"
Los datos demuestran que no se requirió un temporal histórico para quebrar la red en los sectores vulnerables.
• A una ráfaga habitual de 60 km/h, comunas como Cerro Navia o La Pintana cruzan el umbral de colapso crítico (>5% de la comuna a oscuras) con apenas 8 mm de lluvia acumulada.
• En contraste, comunas como Las Condes o Vitacura requieren más de 50 mm de lluvia para cruzar el mismo umbral.
• En viento seco (30 mm de agua), comunas vulnerables colapsan a los 45–50 km/h (W₅₀ = 45 km/h, vientos comunes de invierno), mientras que el sector oriente tolera más de 85 km/h.
👉 Veredicto: El temporal no fue la causa raíz; fue únicamente el detonante de una fragilidad estructural que ya existía antes de la primera gota.

❌ 2. CONTRADICHO: "El arbolado urbano fue el gran responsable"
Al aislar multivariadamente la masa vegetal (m² de arbolado por habitante) frente a la infraestructura física de la red:
• El arbolado per cápita no muestra significancia estadística en la aceleración del tiempo de colapso en el modelo de Cox (p = 0.996, Hazard Ratio ≈ 1.00).
• El verdadero predictor devastador es la proporción de cableado aéreo en postes (p < 0.001, Hazard Ratio = 483, Odds Ratio = 79.000).
• Las comunas de bajos ingresos tienen entre un 85% y 93% de sus líneas expuestas en postes aéreos en calzadas estrechas, frente a un alto estándar de soterramiento en comunas acomodadas.
👉 Veredicto: El problema no era el árbol per se, sino mantener a comunas enteras dependiendo de postes y cables aéreos precarios frente a vientos moderados.

❌ 3. CONTRADICHO: "Un colapso súbito, parejo e imprevisible"
El análisis de supervivencia de Cox demuestra que la caída de la red no fue aleatoria:
• Cada incremento de 1 desviación estándar en el nivel socioeconómico comunal reduce el riesgo instantáneo de apagón masivo en un 67% (Hazard Ratio = 0.33, p < 0.0001).
• Las curvas de Kaplan-Meier revelan que las comunas vulnerables colapsaron a las primeras 6 a 10 horas de iniciado el frente, mucho antes del peak de ráfagas máximas del evento.

✅ 4. LO QUE SÍ SE CONFIRMA: El viento es el factor cinético crítico
Los modelos confirman una relación cuadrática no lineal: la aceleración del apagón responde exponencialmente a las ráfagas máximas de viento más que a los milímetros acumulados de agua. En esto la física no miente: el viento es el gatillante directo, pero actuó sobre una red cuya resistencia estaba predeterminada por la comuna.

---

💡 CONCLUSIÓN DE POLÍTICA PÚBLICA Y REGULACIÓN:
Si la probabilidad de que una familia pase una semana a oscuras ante un temporal invernal depende más de su código postal y de la falta de soterramiento que de los km/h de viento sobre su techo, no estamos ante una "fuerza mayor meteorológica": estamos ante una asimetría estructural de inversión y resiliencia urbana.

La regulación no debería aceptar el argumento de fuerza mayor climática sin antes exigir estándares mínimos y homogéneos de soterramiento y refuerzo de red.

Para auditar y replicar estos hallazgos, dejé el pipeline analítico completo, los datasets procesados y un Simulador Interactivo de Estrés Climático en código abierto:

🔗 Repositorio con datos y metodología: https://github.com/surzua/gridbreak-cl
📊 Simulador interactivo en vivo: [Enlace a Streamlit Cloud]

¿Debe la regulación eléctrica chilena fijar cuotas obligatorias de soterramiento e inversión diferenciadas según la vulnerabilidad de la comuna? Los leo en los comentarios. 👇

#DataScience #Chile #PoliticasPublicas #Energia #Econometria #SurvivalAnalysis #SmartGrid #Python #OpenSource #DataNewsjacking
```

---

## 📝 Copia Alternativa (Opción 2: Enfoque Ejecutivo / 90 Segundos de Lectura)

```markdown
"Fuerza mayor y árboles caídos": la explicación oficial de las eléctricas tras el apagón de Santiago que los datos desmienten. ⚡📉

Cuando el temporal de 2024 dejó a 2 millones de santiaguinos sin suministro eléctrico, los comunicados apuntaron a un chivo expiatorio cómodo: un temporal inédito y árboles impredecibles.

Decidí poner a prueba el titular con datos duros.

Construí un modelo econométrico y de supervivencia (Cox Proportional Hazards, C = 0.90) cruzando más de 100.000 registros oficiales de la SEC y la DMC para las 52 comunas de la RM.

El veredicto empírico es rotundo:

1️⃣ No fue fuerza mayor: A los mismos 60 km/h de viento, comunas como Cerro Navia o La Pintana colapsan con apenas 8 mm de lluvia. Vitacura y Las Condes resisten más de 50 mm. El temporal no provocó la fragilidad; solo la exhibió.

2️⃣ No fue el arbolado: Al controlar por múltiples factores, el arbolado urbano per cápita no explica el tiempo de falla (p = 0.996). El predictor determinante es el cableado aéreo en postes (Hazard Ratio = 483): las comunas vulnerables tienen hasta un 93% de su red expuesta al aire, versus menos del 35% en comunas de altos ingresos.

3️⃣ Tu riesgo de apagón depende de tu comuna: Por cada desviación estándar que sube el nivel socioeconómico comunal, el riesgo instantáneo de colapso se desploma en un 67% (Hazard Ratio = 0.33, p < 0.001).

Conclusión: La red no colapsó por la furia del clima. Colapsó por una asimetría de inversión y soterramiento que toleramos en días despejados.

Pipeline de datos, modelos y simulador comunal en código abierto para auditoría ciudadana:
👉 Repo: https://github.com/surzua/gridbreak-cl
👉 Simulador interactivo: [Enlace a Streamlit Cloud]

¿Debería la SEC invalidar el argumento de "fuerza mayor" cuando la red falla por falta de soterramiento básico? Abro debate.

#DataScience #PoliticasPublicas #Energia #Chile #SmartGrid #Econometrics
```

---

## 📊 Matriz de Auditoría: Titular de Prensa vs. Evidencia Cuantitativa

Esta tabla resume el contraste exacto para responder preguntas o defender los hallazgos en los comentarios del post:

| Afirmación de la Noticia / Minuta Oficial | Lo que decían los medios / distribuidoras | Lo que revelan los datos del modelo | Métrica Cuantitativa de Soporte | Veredicto |
| :--- | :--- | :--- | :--- | :---: |
| **"Evento climático inédito y extremo"** | "Ráfagas históricas y temporales impredecibles que ninguna red puede aguantar." | Comunas periféricas colapsan con ráfagas de 45–50 km/h y 8–15 mm de lluvia (vientos comunes de invierno). | Umbrales $W_{50} = 45\text{ km/h}$ vs. $85\text{ km/h}$; $R_{50} = 8.2\text{ mm}$ vs. $>50\text{ mm}$. | ❌ **CONTRADICHO** |
| **"La culpa es de los árboles caídos"** | "La caída de árboles y ramas sobre las líneas botó la infraestructura." | El arbolado per cápita no explica el fallo al controlar por tipo de red. El factor crítico es tener la red colgada de postes aéreos. | Modelo Cox: Arbolado $p = 0.996$, $\text{HR} \approx 1.00$. Red Aérea: $p < 0.001$, $\text{HR} = 483$. | ❌ **CONTRADICHO** |
| **"Colapso generalizado, súbito y fortuito"** | "Un golpe de la naturaleza donde la red aguantó lo humanamente posible." | La supervivencia depende del nivel socioeconómico. Comunas vulnerables caen en las primeras 6 horas con vientos moderados. | Log-Rank $p = 3.8 \times 10^{-15}$; Hazard Ratio $\text{NSE} = 0.33$ ($p < 0.0001$). C-index = 0.90. | ❌ **CONTRADICHO** |
| **"Las ráfagas de viento provocaron el daño"** | "El viento fue el elemento más destructivo del temporal." | Las ráfagas cuadráticas aceleran exponencialmente la probabilidad de corte, mucho más que el agua acumulada. | Coeficiente GLM Ráfaga cuadrática $p = 0.030$; pendiente sigmoide pronunciada. | ✅ **CONFIRMADO** |

---

## 🖼️ Carrusel Visual de 4 Láminas (Alineado con el Fact-Checking)

Para acompañar el post, subir en formato carrusel de imágenes (PDF o imágenes secuenciales) las figuras generadas en `reports/figures/`:

1. **Lámina 1 (Desmintiendo el "evento inédito"):**  
   📂 `reports/figures/curvas_fragilidad_terciles.png`  
   *Curvas Sigmoides de Fragilidad por Tercil Socioeconómico:* Demuestra cómo a 60 km/h de viento, las comunas vulnerables ya colapsaron con 8 mm de lluvia mientras el sector oriente sigue en 0% de falla.

2. **Lámina 2 (Desmintiendo el "colapso súbito y parejo"):**  
   📂 `reports/figures/supervivencia_kaplan_meier.png`  
   *Curvas de Supervivencia Kaplan-Meier:* Muestra cómo las comunas vulnerables pierden la supervivencia en las primeras 6 a 12 horas del temporal ($p = 3.8 \times 10^{-15}$).

3. **Lámina 3 (Desmintiendo "la culpa es del árbol"):**  
   📂 `reports/figures/hazard_ratios_forest_plot.png`  
   *Forest Plot de Hazard Ratios de Cox:* Exhibe visualmente que el arbolado está sobre la línea nula ($\text{HR} = 1.0$), mientras la red aérea y el nivel socioeconómico son los factores abrumadoramente determinantes.

4. **Lámina 4 (La Brecha Comunal de Viento Crítico):**  
   📂 `reports/figures/brecha_umbrales_viento_comunal.png`  
   *Comparativa de Ráfaga Crítica $W_{50}$:* Evidencia la distancia entre los 45 km/h que derriban a Cerro Navia y los 85+ km/h que requiere Vitacura.

---

## ⏰ Recomendaciones Tácticas para la Publicación
- **Horario clave:** Martes o Miércoles entre 08:15 y 09:30 AM (Hora de Santiago de Chile).
- **Primer comentario:** Publicar inmediatamente el enlace al repositorio de GitHub y el enlace a la app en Streamlit, junto con etiquetas a referentes de energía o regulación (ej. CNE, SEC, académicos) para estimular el debate técnico.
- **Tono en respuestas:** Mantener neutralidad técnica basada en datos: citar p-values, intervalos de confianza del 95% y la metodología de replicación ante cualquier intento de desvío partidista.
