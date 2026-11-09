# Automatidata – NYC Taxi Analytics & Predictions: diagnóstico y plan de mejoras

Documento de trabajo para preparar `automatidata_taxi_analytics_and_predictions.ipynb` antes de publicarlo en el portafolio.

**Cómo se hizo esta revisión.** Leí todo el código y el texto del notebook y contrasté las afirmaciones con las salidas guardadas (tablas impresas y gráficos). **No re-ejecuté el notebook** ni tuve acceso al CSV, así que lo que dependa de re-correr código figura como "por verificar".

**Prioridades**
- **P0** – Errores que contradicen tus resultados, invalidan una conclusión o rompen la presentación. Corregir sí o sí.
- **P1** – Debilidades metodológicas que un revisor técnico notaría.
- **P2** – Calidad de código, reproducibilidad y presentación del repo.
- **P3** – Extras que aumentan el valor del proyecto (sección 9).

---

## 0. Veredicto

El proyecto tiene una arquitectura ambiciosa (EDA, prueba de hipótesis, regresión, clasificación, recomendaciones) y buena idea de integrar la regla del tarifa fija de JFK. Pero en su estado actual **las conclusiones más importantes no están respaldadas por los resultados del propio notebook**:

1. **El modelo de propinas es débil y el texto no lo dice.** ROC-AUC de 0.61–0.64 y exactitud (0.640–0.642) **por debajo** de predecir siempre "propina estándar" (65.2%). Aun así se presenta como base de recomendaciones operativas.
2. **Hay fugas y una variable identificadora en el modelo.** `Unnamed: 0` (el índice del CSV) es la variable #1 del Random Forest, y las "características de ruta" incluyen información del propio viaje.
3. **Se declara campeón al modelo equivocado.** XGBoost supera a Random Forest en las cinco métricas, pero la matriz de confusión y la narrativa usan RF.
4. **Varias cifras del texto no coinciden con las salidas** (diferencia de tarifas, R², MAE), y los montos en dólares se corrompieron en el Markdown.
5. **La "prueba A/B" no es un experimento**, y la recomendación 1 (y la proyección de ingresos) comete la falacia de inferir causalidad de una comparación observacional.

Es un capstone de Google Advanced Data Analytics y los datos son una muestra de 22,699 viajes de 2017, así que lo que diferenciará el proyecto es el rigor y la honestidad sobre los límites del modelo.

---

## 1. Cifras del texto vs. salidas reales (P0)

| # | Afirmación en el texto | Lo que muestra tu salida | Acción |
|---|---|---|---|
| 1.1 | Tarifa media: tarjeta **$13.04** vs. efectivo **$12.21** (+**6.8%**, diferencia $0.83) | Tarjeta **$13.2387** vs. efectivo **$12.1711**; diferencia **$1.0675** (**+8.8%**); IC 95% [0.77, 1.36] | Corregir en la interpretación y en la Recomendación 1 |
| 1.2 | R² del modelo de tarifas "**0.868**" (Recomendación 2) | R² test **0.8779**; con regla JFK sobre todo el dataset **0.8964** | Corregir y aclarar sobre qué conjunto se calcula |
| 1.3 | MAE "just \$?.12" (dígitos perdidos) | MAE test **2.0643**; con JFK **1.9160** | Corregir |
| 1.4 | "~346 features" (sección 5) | **343** features | Corregir |
| 1.5 | "5-Fold Stratified CV" para los dos modelos | RF usa `cv=5`; XGBoost usa **`cv=4`** | Unificar y declarar |
| 1.6 | Recomendación 3: "los tips generosos están fuertemente gobernados por `predicted_fare`, `mean_duration`, ubicación de destino y `pm_rush`" | RF top-5: `Unnamed: 0`, `predicted_fare`, `mean_duration`, `mean_distance`, `VendorID_2`. `pm_rush` **no aparece** en el top 10 de ningún modelo; XGBoost tiene `VendorID_2` muy por encima de todo (≈0.115 vs ≈0.014 la siguiente) | Reescribir con lo que muestran los gráficos |
| 1.7 | Exactitud presentada como resultado del modelo | RF 0.6403 y XGB 0.6416, **menores que 0.652** (clase mayoritaria: propina estándar 65.2%) | Reportar el baseline y admitir que el modelo no lo supera en exactitud |
| 1.8 | Proyección: "+15 puntos porcentuales de pago con tarjeta → más de \$?.5 millones de ingresos anuales" | No hay ningún cálculo en el notebook | Eliminar o calcular con supuestos explícitos (ver 4.1) |

**Verificadas y correctas:** 22,699 registros brutos; 165 viajes con distancia/duración/tarifa ≤ 0 (0.73%); 52 outliers extremos (0.23%); 22,486 registros retenidos (99.06%); N tarjeta = 15,180 y N efectivo = 7,178; Welch t = 7.07, p = 1.67e-12; Cohen's d = 0.10; 495 viajes JFK ajustados; balance de la meta 65.2% / 34.8%.

**Corrección estructural.** Calcula las cifras en código y muéstralas con f-strings en las conclusiones:

```python
from IPython.display import Markdown, display
display(Markdown(
    f"Los pagos con tarjeta tienen una tarifa media de **\\${card_mean:.2f}** frente a "
    f"**\\${cash_mean:.2f}** en efectivo (+{diff_mean/cash_mean:.1%}, IC 95% "
    f"[\\${ci_low:.2f}, \\${ci_high:.2f}], d = {cohens_d:.2f})."
))
```

---

## 2. Errores de renderizado, codificación y gráficos (P0)

- [ ] **Dólares con dígitos perdidos.** Montos como `\\.83`, `\\.12`, `\\.00`, `\\.5 million`, `>\\ or >3 hrs` y `JFK \\ flat fare`: el `$` seguido de dígitos se interpretó como LaTeX o como referencia de regex y se comieron las cifras. Escapa los dólares como `\$` en Markdown. También `**^2 = 0.868$**` y `( < 0.001$)` perdieron `$R$` y `$p$`.
- [ ] **F-strings con la expresión perdida:**
  - Etiquetas del KDE: `label=f'Credit Card (Mean: )'`: no muestran la media.
  - Salida impresa: `Extreme telematics outliers (>50 miles, >\ or >3 hrs)`.
  - Comentario de JFK: `Flat Fare = )`.

  Reescríbelos con `f"... \${card_mean:.2f} ..."`.
- [ ] **Mojibake (UTF-8 mal decodificado):** `RÂ²` en las salidas y `16:00Ã¢â‚¬â€œ20:00` en las recomendaciones. Guarda el notebook en UTF-8 y usa `R²` / `–` normales.
- [ ] **Leyenda errónea en el scatter tarifa vs. distancia.** `ax.legend(labels=[...])` reasigna etiquetas a handles en otro orden: el gráfico muestra "Credit Card" en naranja y "Cash" en azul (al revés de la paleta), "No Charge" en naranja repetido y "Dispute" en verde. Mapea `payment_type` a texto (`df['pago'] = df['payment_type'].map({1:'Tarjeta',2:'Efectivo',3:'Sin cargo',4:'Disputa'})`) y usa `hue='pago'` sin tocar `labels=`.
- [ ] **Eje Y con la etiqueta "None"** en los gráficos de importancias (`barplot` con una Series sin nombre). Añade `ax.set_ylabel('')`.
- [ ] **Escapes sospechosos en el Markdown:** `\numpy` (en lugar de `numpy`) y `\trip_distance` (ahí `\t` es un tabulador). Revísalo al renderizar.
- [ ] Mostrar un único `n` y la muestra usada en el scatter (3,000 puntos al azar de 22k): indícalo en el título.

---

## 3. Limpieza de datos y EDA (P1)

### 3.1 Auditoría de calidad incompleta
La sección promete una "auditoría de higiene de datos", pero no hay conteo de **valores faltantes**, **duplicados**, tipos de dato (`info()`), ni revisión de `passenger_count == 0` o de valores válidos de `RatecodeID`/`payment_type`. Añade esas comprobaciones y reporta resultados.

### 3.2 Umbrales de filtrado arbitrarios
`distancia ≤ 50 mi`, `tarifa ≤ 150`, `duración ≤ 180 min` no están justificados. Justifícalos con la distribución (percentiles, IQR) o con reglas de dominio y reporta cuántas filas elimina cada regla por separado. Verifica que una tarifa de $150 no sea un viaje legítimo a aeropuerto con peajes.

### 3.3 Fechas
Las fechas llegan como `03/25/2017 8:55:43 AM`. Usa `pd.to_datetime(..., format='%m/%d/%Y %I:%M:%S %p')` para evitar inferencias distintas según la versión de pandas.

### 3.4 El EDA no tiene conclusiones escritas
Hay 4 gráficos sin una sola frase de interpretación. Lo que se ve: demanda mínima de 3 a 5 h y máxima a las 18–19 h; lunes y domingo con menos viajes; distribución de distancias muy asimétrica; el "escalón" de $52 en el scatter es la tarifa plana JFK. Escríbelo.

### 3.5 "Spatio-Temporal" sin análisis espacial
El título habla de dinámica espacio-temporal, pero no hay ningún análisis de zonas. Añade el análisis por zona (con la tabla `taxi_zone_lookup`) y/o por *borough*, o cambia el título.

### 3.6 Contexto
- Declara que el "cliente" y la consultoría son un **escenario ficticio** del capstone y que los datos son una **muestra** de 22,699 viajes de 2017, no representativa de toda la flota.
- Cita fuente para "13,000 medallions" y para "propinas = 20–30% de los ingresos de los conductores".
- `Unnamed: 0` (índice del CSV) se arrastra durante todo el notebook: elimínalo al cargar (`index_col=0`).

---

## 4. Prueba de hipótesis: tarjeta vs. efectivo (P0/P1)

### 4.1 No es un A/B test (P0)
Los pasajeros no fueron asignados al azar a pagar con tarjeta o efectivo: es una **comparación observacional**. Renombra la sección ("comparación de dos muestras") y elimina el lenguaje de causalidad.

### 4.2 Inferencia causal indebida en las recomendaciones (P0)
- La Recomendación 1 propone obligar a pagar con tarjeta porque "los clientes de tarjeta generan una tarifa mayor". La tarifa depende de la distancia y la duración del viaje; los viajes más largos pueden preferir tarjeta, pero **forzar tarjeta no alargará los viajes**. La diferencia es de selección, no de efecto.
- La proyección "+15 pp de adopción → X millones" arrastra la misma falacia y no tiene cálculo. Elimínala o reformúlala (p. ej. el beneficio plausible es el registro de propinas con tarjeta y menor fricción, no mayor tarifa).
- La explicación "viajes más largos y viajeros de negocios sin efectivo" es especulativa: compruébala (distancia media por tipo de pago) o márcala como hipótesis.

### 4.3 Tamaño del efecto no interpretado
Cohen's d = **0.10** (efecto pequeño) se calcula pero no se menciona; el texto dice "rechazamos decisivamente H0" solo por el p-valor (con N de ~22k, casi cualquier diferencia sale significativa). Reporta significancia **y** magnitud.

### 4.4 Mejoras del análisis
- Controla por distancia/duración: compara tarifa **por milla** o ajusta una regresión `fare ~ pago + distancia + duración`. Es probable que la "diferencia" cambie mucho (por verificar).
- Verifica supuestos (las tarifas están sesgadas a la derecha): complementa con Mann–Whitney o bootstrap del IC.
- El IC usa 1.96; está bien con este N, pero indícalo.
- Declara la hipótesis alternativa y el α **antes** de ver los resultados (ya lo haces bien) y explica por qué se excluyen los pagos 3 y 4.

---

## 5. Regresión lineal para la tarifa (P0/P1)

### 5.1 Fuga en las variables de ruta (P0)
`mean_distance` y `mean_duration` se calculan con `groupby` sobre **todo** el dataset (incluyendo filas de test y **el propio viaje**). Tu salida lo muestra: la ruta `188_97` tiene `trip_distance = 3.70` y `mean_distance = 3.70`, y `duration = 30.25` igual a `mean_duration` (ruta con un único viaje). En esos casos el "benchmark histórico" es la distancia/duración **reales** del viaje, que no se conocen antes de subir al taxi, lo que contradice la promesa de "pre-estimación de tarifa".

**Acción:** calcula las medias de ruta **solo con train**, con *leave-one-out* o por *folds*, y define un plan para rutas no vistas (media por zona de origen/destino, mediana global, o distancia haversine entre centroides de zona). Reevalúa; el R² probablemente bajará y esa cifra honesta será la que importa.

### 5.2 "Verificación matemática" de supuestos que no se cumplen (P1)
- Dices que el VIF debe ser < 5, y tu salida da **8.03** (`mean_duration`) y **6.23** (`mean_distance`): incumples tu propio criterio y no lo comentas.
- `variance_inflation_factor` se calculó **sin constante** (`X_reg.values` sin `add_constant`), lo que puede distorsionar los valores. Recalcula con `sm.add_constant`.
- Los gráficos de residuos muestran **colas muy pesadas** en el Q-Q plot (residuos de hasta ≈ +77), asimetría positiva y una línea descendente a la derecha de ~$52 (el modelo sobreestima los viajes JFK). La normalidad **no se cumple** y no lo mencionas. Los residuos deben interpretarse por escrito.
- La normalidad de residuos no es un supuesto de Gauss–Markov (sirve para inferencia exacta). Linealidad, homocedasticidad e independencia sí; la linealidad no se comprueba.
- `passenger_count` (coef. 0.019) no aporta: elimínalo o discútelo.
- Dos predictores muy correlacionados (distancia/duración): considera usar solo uno, o regularización (Ridge).

### 5.3 Métricas "finales" sobre datos de entrenamiento (P1)
Las métricas con la regla JFK (R² 0.8964, MAE 1.916) se calculan sobre **todo** `df_reg` (train + test) y el ajuste JFK se aplica al conjunto completo. Reporta todo sobre **test**, con y sin la regla JFK, y con IC (bootstrap).

### 5.4 La regla JFK
- `52.00` está escrito a mano. Defínela como constante nombrada (`JFK_FLAT_FARE = 52.0`) y cita la fuente de la tarifa vigente en 2017.
- Cuenta cuántos de los 495 viajes JFK tienen `fare_amount == 52` realmente, y reporta el error residual.
- Habla de "Manhattan ↔ JFK", pero `RatecodeID == 2` no distingue origen/destino: precisa el alcance.
- "Alinea perfectamente con la regulación" es excesivo.

### 5.5 Baselines
Compara contra: (a) la fórmula del taxímetro (base + por milla + por minuto) con `trip_distance`/`duration` reales (cota superior), (b) una regresión con solo `mean_distance`, (c) un modelo con interacciones o de árboles. Sin baseline, un R² de 0.88 no es interpretable.

---

## 6. Clasificación de propinas (P0/P1)

### 6.1 `Unnamed: 0` es la variable #1 del Random Forest (P0)
Es el índice del archivo (`Unnamed: 0` aparece en el `head()`). No es una característica de negocio; probablemente refleja el orden de las filas o del muestreo. Elimínala y re-entrena **todo**. Es posible que el rendimiento baje todavía más (por verificar).

### 6.2 El modelo no supera el baseline y el texto no lo reconoce (P0)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (siempre "estándar") | **0.652** | 0 | 0 | 0 | 0.50 |
| Random Forest | 0.6403 | 0.4465 | 0.1381 | 0.2110 | 0.6071 |
| XGBoost | 0.6416 | 0.4705 | 0.2337 | 0.3123 | 0.6350 |

Con AUC ≈ 0.61–0.64 el modelo discrimina apenas mejor que el azar, y detecta solo 14%–23% de las propinas generosas. Las Recomendaciones 3 (heatmaps y reposicionamiento) y la narrativa de "predecir propinas antes de que termine el viaje" **no son sostenibles** con estos resultados. Reescribe la sección con una conclusión honesta: "con estos datos la señal es débil", y explica por qué (ver 6.6).

### 6.3 Campeón mal elegido (P0)
XGBoost supera a RF en **las cinco métricas**, pero la matriz de confusión se rotula "Champion: Random Forest" y la narrativa también lo usa. Corrige: decide el campeón con un criterio explícito y grafica ese modelo (o ambos).

### 6.4 Fugas y variables inconsistentes (P1)
- **`predicted_fare` como característica.** Lo excluyes de la lista de fugas (`fare_amount`) y luego incluyes `predicted_fare`, que es casi `fare_amount` (R² ≈ 0.9). Es contradictorio. Además `predicted_fare` se generó con un modelo entrenado en filas que luego caen en el test del clasificador (el split de la regresión y el del clasificador son independientes), y con las medias de ruta calculadas sobre todo el dataset (ver 5.1). Usa predicciones *out-of-fold* o excluye la variable.
- **`trip_distance` marcada como fuga** pero se usan `mean_distance`/`mean_duration`, que para rutas con un solo viaje equivalen a la distancia/duración reales. Criterio inconsistente.
- **Condicionar a `payment_type == 1`.** El método de pago se define al final del viaje: el modelo predice entre quienes ya pagaron con tarjeta, no entre "pasajeros que acaban de subir". Documéntalo como limitación del caso de uso "antes de completar el viaje".
- **`RatecodeID`, `VendorID` y zonas como categóricas:** correcto tratarlas como nominales, pero 343 columnas con ~12k filas dispersa la señal (ver 6.5).

### 6.5 Rendimiento y búsqueda de hiperparámetros (P1)
- RF: `max_features` ∈ {3, 4} con 343 columnas es extremadamente bajo (el default √343 ≈ 18), y el óptimo cae en el **borde** de la rejilla (`max_features=4`, `min_samples_leaf=1`, `min_samples_split=2`, el menor valor). `max_depth` tiene un solo valor (`[None]`), no es una búsqueda.
- XGBoost: `max_leaf_nodes` no es un parámetro de XGBoost (se usa `max_leaves` con `grow_policy='lossguide'`); creo que se ignora silenciosamente y la rejilla repite ajustes idénticos (por verificar con `get_params()`). `learning_rate` tiene un solo valor, `n_estimators=200` es el máximo explorado.
- Optimizas `f1` al umbral por defecto 0.5 en una meta desbalanceada (34.8%): con recall bajo, parte del problema es el umbral, no el modelo. Añade curva precision-recall, PR-AUC y barrido de umbrales.
- No hay baseline (regresión logística, árbol simple, o "siempre estándar") ni validación de que los ensambles aporten.
- Alinea CV (5 folds, estratificado, para ambos) con lo que dice el texto. El texto promete "calibrar" los modelos; no hay calibración.
- Agrupa zonas por *borough* o usa *target encoding* con CV en lugar de 300+ dummies.

### 6.6 Definición de la meta
- `tip_percent = tip_amount / (total_amount − tip_amount)` y umbral `>= 0.20`: la comparación con decimales puede clasificar mal casos exactamente en 20% por redondeo de coma flotante. Redondea o usa tolerancia (por verificar cuántas filas están en el borde).
- Los terminales de pago ofrecen opciones predefinidas de propina, y es plausible que muchas propinas se concentren en valores como 20% o 25%. Explora la distribución de `tip_percent` (histograma) antes de elegir un corte arbitrario; prueba una regresión sobre `tip_percent` o una meta de varios niveles.
- Justifica el 20% (por qué "generosa").

### 6.7 Interpretabilidad (P1)
- MDI (Gini) en 343 columnas, muchas dummies: sesgado. Usa **permutation importance** agrupada (por variable original: zona, vendor, hora) o **SHAP**.
- El título del gráfico de XGBoost dice "(Weight)"; creo que el default de `feature_importances_` en la API de scikit-learn es *gain* (verifica `best_xgb.importance_type`).
- `VendorID_2` domina en XGBoost por un amplio margen: es el hallazgo más llamativo y no lo discutes. Plausible hipótesis: diferencias entre los terminales de cada proveedor (opciones de propina por defecto), no entre pasajeros. Márcalo como hipótesis y valídalo (tasa de propina generosa por `VendorID`).

---

## 7. Narrativa, recomendaciones y tono (P1/P2)

- **Recomendación 1 (tarjeta):** reformula (ver 4.2).
- **Recomendación 2 (estimador de tarifa):** válida si corriges la fuga de rutas (5.1) y reportas métricas honestas en test. Añade intervalos de predicción.
- **Recomendación 3 (reposicionamiento):** no hay análisis espacial que respalde "Midtown → Upper Manhattan y JFK/LGA entre 16 y 20 h", y `pm_rush` no está en el top 10. Elimínala o respáldala con un análisis de tasa de propina por zona y franja horaria (con intervalos).
- **Recomendación 4 (ingresos):** eliminar o calcular (ver 4.2).
- Declara que las propinas en efectivo no se registran (el `0` no significa que no hubo propina), y que por eso **no se puede comparar el comportamiento de propina entre tarjeta y efectivo**.
- Tono: "decisively reject", "master portfolio project", "Strategic Consulting", "calibrate", "Spatio-temporal" prometen más de lo que el notebook entrega. Un tono sobrio transmite más seniority.
- Mueve los resultados clave a un **Executive Summary con cifras reales** al inicio, incluyendo la limitación principal del modelo de propinas.
- Añade una sección final de **limitaciones y próximos pasos**.

---

## 8. Código y reproducibilidad (P2)

- [ ] Quitar imports sin uso: `os`, `datetime as dt`, `StandardScaler`, `StratifiedKFold`.
- [ ] No silenciar todas las advertencias con `warnings.filterwarnings('ignore')`; quitar `use_label_encoder=False` (obsoleto en XGBoost ≥ 2.0).
- [ ] Usar `Pipeline` + `ColumnTransformer` (one-hot, agregados de ruta ajustados solo con train) para evitar fugas y facilitar despliegue.
- [ ] Factorizar funciones (`clean_trips`, `build_route_features`, `evaluate_classifier`) con docstrings y type hints.
- [ ] Fijar semillas (ya está `random_state=42`) y versiones en `requirements.txt`.
- [ ] Sustituir números mágicos (50, 150, 180, 52, 0.20, 0.65…) por constantes con nombre.
- [ ] Ejecutar "Restart & Run All" antes de subir.
- [ ] Peso: 614 KB, aceptable; opcional exportar figuras a `reports/figures/`.

### Estructura de repo sugerida
```
automatidata-taxi/
├── README.md
├── requirements.txt
├── data/                  # o instrucciones de descarga + licencia
├── notebooks/automatidata_taxi.ipynb
├── src/                   # limpieza, features, evaluación
├── reports/figures/
└── model_card.md
```
El README debe contener: pregunta de negocio, 3–5 hallazgos con cifras reales, métricas elegidas y por qué, **limitaciones**, y cómo reproducir.

---

## 9. Checklist ordenado de ejecución

**Primero (P0, ~2–3 h)**
- [ ] Eliminar `Unnamed: 0` al cargar y re-ejecutar todo.
- [ ] Corregir cifras de la sección 1 (y generarlas por código); arreglar dólares, f-strings y mojibake.
- [ ] Corregir leyenda del scatter y etiquetas de ejes.
- [ ] Cambiar el campeón (o justificar la elección) y graficar el modelo correcto.
- [ ] Renombrar "A/B test" y reescribir Recomendaciones 1 y 4.

**Después (P1, ~1–2 días)**
- [ ] Recalcular las medias de ruta solo con train (o leave-one-out) y reevaluar la regresión sobre test.
- [ ] Recalcular VIF con constante; interpretar residuos; baselines de regresión.
- [ ] Baseline y umbral del clasificador; curva PR; ampliar rejillas; arreglar `max_leaf_nodes`.
- [ ] Revisar `predicted_fare` y `trip_distance` (consistencia de fugas).
- [ ] Controlar por distancia en la comparación tarjeta/efectivo.
- [ ] Permutation importance/SHAP y análisis de `VendorID`.
- [ ] Reescribir la narrativa del modelo de propinas con sus límites.

**Al final (P2)**
- [ ] Auditoría de calidad de datos (faltantes, duplicados), justificación de umbrales, interpretación del EDA.
- [ ] Limpieza de código, `Pipeline`, `requirements.txt`, README, tono.
- [ ] Restart & Run All.

---

## 10. Técnicas y enfoques para darle más valor (P3)

Ordenadas por impacto sobre el portafolio frente a esfuerzo.

### 10.1 Alto impacto, esfuerzo moderado
1. **Reformular la pregunta de propinas con honestidad.** Pasa de "predecir 20%" a un análisis de *qué se asocia con propina alta* con intervalos de confianza, e incluye un modelo de baseline. Una conclusión de "señal débil, y por qué" vale más que métricas infladas.
2. **Regresión de la propina continua** (`tip_percent`) y/o clasificación multinivel, con *quantile regression* para ver el comportamiento en colas. Mira la distribución antes de elegir el corte.
3. **Estimación de tarifa sin fugas con incertidumbre.** Medias de ruta calculadas en train con *leave-one-out*, fallback por zona, intervalos de predicción (regresión cuantílica o *conformal prediction*) y una evaluación separada de rutas **vistas vs. no vistas**.
4. **Modelo híbrido tarifa = regla + residuo.** En lugar de sobrescribir con $52, modela explícitamente `RatecodeID` (JFK, Newark, negociada) y deja que un modelo de árboles/boosting aprenda el residuo del taxímetro. Compara contra el enfoque actual.
5. **SHAP global y local**, con *dependence plots* de distancia, duración, hora y vendor.
6. **Análisis de `VendorID`:** si el efecto es de terminal (opciones de propina por defecto), eso es un hallazgo de producto muy accionable.

### 10.2 Estadística e inferencia
7. **Comparación tarjeta/efectivo ajustada:** regresión con covariables, *propensity score matching* o *inverse probability weighting* para acercarse a una comparación justa, con la advertencia de que sigue siendo observacional.
8. **IC por bootstrap** para todas las métricas y para la diferencia de medias; pruebas no paramétricas (Mann–Whitney) como robustez.
9. **Modelos robustos para la regresión:** Huber/RANSAC o transformación logarítmica de la tarifa dado el sesgo y las colas pesadas; modelos con interacciones (distancia × hora, distancia × zona).

### 10.3 Espacial y temporal
10. **Análisis por zona y por borough:** mapas coropléticos de demanda, tarifa media y tasa de propina (con `geopandas` y el shapefile de zonas TLC) y *clustering* de zonas por perfil de demanda.
11. **Estacionalidad y rutina:** demanda por hora × día (mapa de calor), efecto de lluvia/festivos con datos externos de clima.
12. **Series de tiempo de demanda** (agregada por hora/zona) con un modelo base estacional y uno de boosting; sirve para la idea de "reposicionamiento" con respaldo real.

### 10.4 Modelado avanzado
13. **LightGBM/CatBoost** con soporte nativo de categóricas (zonas) y búsqueda con Optuna, comparando contra el baseline.
14. **Calibración de probabilidades** (curva de calibración, Brier score, isotónica o Platt) si vas a usar la probabilidad para decidir.
15. **Selección de umbral por valor de negocio**, p. ej. maximizar la propina esperada capturada bajo una restricción de operación.

### 10.5 Comunicación y producto
16. **Mini-app (Streamlit/Gradio):** ingresa origen, destino y hora → tarifa estimada con intervalo, y (si el modelo lo justifica) probabilidad de propina alta con su explicación.
17. **Dashboard** (Power BI o Streamlit) con demanda por zona/hora, comparación de pagos y simulador de tarifa.
18. **Model card y datasheet:** uso previsto, datos, métricas honestas, sesgos (muestra de 2017, solo tarjeta), limitaciones.
19. **Sección de ética:** riesgos de usar predicción de propinas para decidir qué pasajeros atender (discriminación por zona/barrio), y privacidad (ubicaciones y horarios).
20. **Ingeniería ligera:** `joblib` para serializar, `pytest` para la limpieza y las funciones de features, y GitHub Actions que ejecute el notebook o las pruebas.

### Combinación recomendada si el tiempo es limitado
Una versión "de portafolio" sólida sale con: **correcciones P0 + 5.1 (rutas sin fuga) + 6.1–6.3 + 6.5 (baseline y umbral) + 10.1 (puntos 1, 3 y 6) + 10.3 (punto 10) + mini-app (16) + model card (18)**.
