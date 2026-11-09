# Retail Banking – Churn Prediction & K-Means Segmentation: diagnóstico y plan de mejoras

Documento de trabajo para preparar `customer_churn_and_segmentation.ipynb` antes de publicarlo en el portafolio.

**Cómo se hizo esta revisión.** Leí todo el código y el texto del notebook y contrasté las afirmaciones con las salidas guardadas (tablas impresas y gráficos). **No re-ejecuté el notebook** ni tuve acceso al CSV, así que lo que dependa de re-correr código figura como "por verificar".

**Prioridades**
- **P0** – Errores que contradicen tus resultados, invalidan una conclusión o rompen la presentación. Corregir sí o sí.
- **P1** – Debilidades metodológicas que un revisor técnico notaría.
- **P2** – Calidad de código, reproducibilidad y presentación del repo.
- **P3** – Extras que aumentan el valor del proyecto (sección 10).

---

## 0. Veredicto

Es el más sólido de los tres notebooks en cuanto a **idea de negocio**: combinar un modelo de churn con una segmentación y priorizar por **saldo en riesgo** (la "matriz 2D") es exactamente el tipo de síntesis que se valora en un portafolio. Las métricas del modelo son razonables y consistentes entre CV y test.

Los problemas están en cómo se sostiene esa síntesis:

1. **La matriz 2D usa probabilidades infladas y evaluadas en datos de entrenamiento.** El modelo se entrena con `scale_pos_weight`, que distorsiona las probabilidades (el riesgo medio predicho es entre 1.5 y 2.5 veces la tasa observada en todos los clústeres), y luego se puntúa el 100% de los clientes, incluidos los 8,000 de train.
2. **La segmentación tiene estructura débil** (silhouette máximo 0.1645) y el "k óptimo = 6" está escrito a mano en el gráfico, no deducido.
3. **Las recomendaciones no están conectadas con los resultados**: personas sin nombre, "créditos altos" que no existen en los datos, umbrales distintos entre texto y código, y cifras de ROI con los dígitos perdidos.
4. **Comparación de modelos no equivalente**: cada modelo maneja el desbalance de forma distinta (árbol sin pesos, RF balanceado, XGBoost con `scale_pos_weight`), así que precisión/recall/F1/exactitud no son comparables entre ellos.

Además, este es un dataset muy popular y probablemente sintético (ver 3.4), por lo que el valor diferencial vendrá de la decisión basada en costos, la calibración y la honestidad sobre los límites.

---

## 1. Cifras y afirmaciones del texto vs. salidas reales (P0)

| # | Afirmación en el texto | Lo que muestra tu notebook | Acción |
|---|---|---|---|
| 1.1 | Modelos ajustados con CV "optimizando **ROC-AUC y F1**" | Los tres `GridSearchCV` usan **`scoring='roc_auc'`** únicamente | Corregir el texto o ajustar con ambas métricas (`refit`) |
| 1.2 | "XGBoost y Random Forest logran **Recall > 70%**" | XGBoost **0.7469**; Random Forest **0.6757** | Corregir (solo XGBoost supera 70%) |
| 1.3 | "ROC-AUC ≈ 0.86" para ambos | RF **0.8569**, XGB **0.8683** (consistente) | OK, pero reportar los dos valores |
| 1.4 | Sección titulada "ROC-**PR** Diagnostics" | Se importa `precision_recall_curve`, pero **no se grafica** ninguna curva PR | Añadirla o cambiar el título |
| 1.5 | Silhouette evaluado en "k ∈ [2, 8]" | El código usa **`range(2, 15)`** | Corregir |
| 1.6 | "k = 6 es óptimo" (leyenda "Optimal k = 6") | La línea roja está **escrita a mano** (`axvline(x=6)`); el silhouette máximo es **0.1645** y el codo no es identificable | Ver sección 5 |
| 1.7 | Recomendación 4: marcar clientes con riesgo **≥ 0.60** | El código usa un umbral de **0.50** | Unificar |
| 1.8 | Recomendación 4: "**Quarterly** Execution: run … **monthly**" | Contradicción interna | Elegir una cadencia |
| 1.9 | Recomendación 1: persona "Affluent Prime Savers" con "**altos puntajes de crédito**" | `CreditScore` es prácticamente igual en los 6 clústeres (647–656) | Eliminar la afirmación o reformular |
| 1.10 | Recomendación 1: "proteger solo el 5% de balances en riesgo preserva **\\ a \\**" | No hay cálculo; los dígitos se perdieron | Calcular con supuestos explícitos (ver 6.4) |
| 1.11 | Recomendación 3: Alemania churn alto "**independientemente de la antigüedad**" | No hay ningún análisis por antigüedad en el notebook | Probarlo o quitarlo |
| 1.12 | Recomendación 2: "los datos **prueban** que 3–4 productos implican 82.7%–100% de churn" | Tasas correctas en el gráfico (27.7%, 7.6%, 82.7%, 100%), pero **sin `n`** por grupo | Mostrar `n` y reformular "prueban" (ver 4.7) |

**Verificadas y correctas:** 10,000 clientes × 14 columnas; cero faltantes; 36.2% con saldo cero; split 8,000/2,000 estratificado (20.4% / 20.3% de churn); métricas de test (árbol AUC 0.8429, RF 0.8569, XGB 0.8683); coherencia CV–test (0.840/0.843, 0.862/0.857, 0.865/0.868); tasas de churn por productos; matriz de confusión de XGBoost (TN 1292, FP 301, FN 103, TP 304).

**Por verificar (no están impresas):** las tasas citadas por país (32.4% / 16.1% / 16.7%) y por género (25.1% / 16.5%) y las medianas de edad (45 vs. 36). Son coherentes con las barras por país×género (p. ej. Alemania 37.6% mujeres y 27.8% hombres), pero no se calculan en ninguna celda.

**Corrección estructural.** Calcula las cifras en código y muéstralas con f-strings en las conclusiones:

```python
from IPython.display import Markdown, display
by_geo = df.groupby('Geography')['Exited'].agg(['mean','count'])
display(Markdown(
    f"Alemania presenta un churn de **{by_geo.loc['Germany','mean']:.1%}** "
    f"(n={by_geo.loc['Germany','count']:,}) frente a {by_geo.loc['France','mean']:.1%} en Francia."
))
```

---

## 2. Errores de renderizado y gráficos (P0)

- [ ] **Dólares y LaTeX corruptos en Markdown:** `$\\text{k}+$`, `**\text{--}100\%$**` y `\\ to \\` perdieron cifras y símbolos. Escapa los dólares como `\$` y revisa cada celda con `$...$`.
- [ ] **Variable `k` perdida en títulos:** "Determining Optimal Clusters ($)" y "across $\in [2,8]$" perdieron el `k`.
- [ ] **Leyenda sin marcadores en el scatter balance vs. salario.** En el gráfico aparecen "Retained" y "Exited" sin color: `ax.legend(labels=[...])` sobrescribe los handles de seaborn. Mapea `Exited` a texto (`df['Estado'] = df['Exited'].map({0:'Retained',1:'Exited'})`) y usa `hue='Estado'` sin `labels=`.
- [ ] **Silhouette con eje Y truncado** (0.140–0.165): visualmente exagera un pico que corresponde a una diferencia de 0.02. Fija el eje a `[0, 1]` o añade una línea de referencia en 0.25 ("estructura débil por debajo").
- [ ] **Paleta `coolwarm` en churn por clúster.** Los colores se asignan por ID de clúster, no por valor: el clúster 4 (churn más bajo, 8.2%) aparece en tono cálido. Usa un solo color o ordena y colorea por tasa.
- [ ] **Línea roja "Optimal k = 6"** dibujada a mano (ver sección 5).
- [ ] **Ruta con backslash:** `"data\Churn_Modelling.csv"` solo funciona en Windows. Usa `Path("data") / "Churn_Modelling.csv"`.
- [ ] **`persona_names` incompleto** (`'Persona A:'`, …, nombres truncados, probablemente por el mismo problema de `$`) y **nunca se usa**. Define y aplica nombres reales (ver sección 5.3).
- [ ] Mostrar `n` en las barras por número de productos y por clúster.
- [ ] Redondear `Mean_Predicted_Risk` (se imprime como `32.099998`).

---

## 3. Limpieza de datos, EDA y contexto (P1)

### 3.1 Auditoría incompleta
Solo se comprueban faltantes. Añade: duplicados (`CustomerId` único, filas idénticas), consistencia (`Balance == 0` por país, `HasCrCard` e `IsActiveMember` binarios, rangos de `Age`/`CreditScore`) y outliers. Reporta resultados.

### 3.2 El EDA no sostiene algunas frases
- "El churn se acelera entre los 45 y 60 años": se apoya en un KDE de la **distribución** de edad, que no muestra tasas. Grafica **churn por tramos de edad** con `n` e IC.
- "Alemania churn el doble": revisa la confusión con el saldo. En muchos datos de este tipo los clientes alemanes casi no tienen saldo cero (por verificar con `pd.crosstab(df.Geography, df.zero_balance)`); si es así, parte de la diferencia geográfica puede ser un efecto de saldo, no de país.
- Brecha de género: se observa en los tres países (≈ 8–10 puntos), lo cual descarta que sea un artefacto de país. Dilo, y evalúa si persiste controlando por edad/saldo/actividad (regresión logística).
- El scatter balance vs. salario no aporta: no hay relación visible y el salario parece distribuido uniformemente. Sustitúyelo por churn según tramos de saldo.

### 3.3 Variables dudosas
- `EstimatedSalary` muestra mín 11.58, máx 199,992 y cuartiles ≈ 51k / 100k / 149k: parece **uniforme**, típico de una variable sintética sin señal. `balance_to_salary` hereda ese ruido. Evalúa su aporte (permutation importance) antes de defenderla como "indicador financiero".
- `CreditScore`, `Tenure` y `EstimatedSalary` casi no varían entre clústeres (ver sección 5).

### 3.4 Contexto y fuente
Es el dataset `Churn_Modelling.csv`, muy usado en Kaggle y probablemente sintético (por verificar su procedencia). Decláralo en la introducción, con enlace y licencia. Cita fuente para "5 a 7 veces más caro retener que adquirir" o quítalo.

### 3.5 Número de productos: ¿fenómeno o artefacto?
3 productos → 82.7% y 4 productos → 100% de churn. En un banco real, un 100% sería sospechoso. Tu explicación ("disputa o venta cruzada agresiva") es especulación. Alternativas a considerar: el conteo se registró **después** del cierre (fuga de información), o es un artefacto de la generación de datos. Muestra `n` (los grupos de 3–4 productos suelen ser pequeños, por verificar) y entrena una versión **sin `NumOfProducts`** para ver cuánta capacidad predictiva depende de ella.

---

## 4. Modelado supervisado (P0/P1)

### 4.1 Los modelos no son comparables entre sí (P0)
- Árbol: sin pesos de clase. Random Forest: `class_weight='balanced'`. XGBoost: `scale_pos_weight`. Cada uno opera en un punto distinto de la curva precisión–recall, así que precisión, recall, F1 y exactitud **no son comparables**. Solo ROC-AUC y PR-AUC lo son.
- **Exactitud:** la clase mayoritaria es 79.6%. XGBoost tiene 0.798 (≈ baseline) y el árbol 0.861 con recall de solo 0.43. La exactitud no es informativa; elimínala o compárala con el baseline.
- **Acción:** (a) misma estrategia de desbalance para los tres, o ninguna y se ajusta el umbral; (b) compara en un punto operativo común (por ejemplo, precisión con recall fijo en 70%, o recall a una tasa de alertas fija del 20%).

### 4.2 Elección del campeón sin criterio explícito (P1)
Se rotula "Champion: XGBoost" sin explicar el criterio. XGBoost gana en AUC (0.868) y recall (0.747), pero RF gana en F1 (0.629) y precisión (0.588), y el árbol en precisión (0.787). Las curvas ROC de los tres casi se superponen (AUC 0.843–0.868). Declara el criterio (p. ej. costo esperado, ver 4.3) y reporta IC por bootstrap; la diferencia RF–XGB probablemente no sea significativa (por verificar).

### 4.3 Umbral y decisión de negocio (P1)
- Se usa el umbral por defecto (0.5) y la matriz de confusión del campeón muestra **301 falsas alarmas por 304 aciertos** (precisión 50.2%): por cada cliente en riesgo real que se contacta, otro leal recibe la oferta.
- **Acción:** curva PR, PR-AUC, gráfico de **lift/ganancia acumulada** (¿qué porcentaje de los churners captura el 10% o 20% con mayor puntaje?) y selección del umbral con costos explícitos (costo de la oferta vs. valor del cliente retenido).

### 4.4 `scale_pos_weight` distorsiona las probabilidades (P0)
Con pesos de clase las probabilidades dejan de estar calibradas. Tu propia matriz lo demuestra:

| Clúster | Riesgo medio predicho | Churn observado |
|---|---|---|
| 0 | 32.1% | 12.6% |
| 1 | 29.1% | 16.4% |
| 2 | 51.9% | 32.0% |
| 3 | 55.7% | 32.4% |
| 4 | 19.6% | 8.2% |
| 5 | 46.5% | 31.4% |

El modelo sobreestima en todos los clústeres. Por eso "Mean_Predicted_Risk_%" **no es una probabilidad**. Opciones: entrenar sin pesos y mover el umbral, o calibrar después (isotónica/Platt) y comprobar con curva de calibración y Brier score.

### 4.5 Búsqueda de hiperparámetros en el borde (P1)
- RF: `max_features=0.6` (el máximo explorado), `min_samples_leaf=5` (máximo), `n_estimators=200` (máximo).
- XGB: `max_depth=3` y `n_estimators=100` (mínimos), `learning_rate=0.1` y `colsample_bytree=1.0` (máximos).
- Árbol: `min_samples_split=10` (mínimo).

Amplía las rejillas hacia esos lados o usa búsqueda aleatoria/Optuna con *early stopping*. Para XGBoost, añade `min_child_weight`, `reg_lambda` y `gamma`.

### 4.6 Texto que no coincide con el código (P1)
- "Random Forest con *balanced bootstrapping*": el código usa `class_weight='balanced'` (para bootstrap balanceado sería `balanced_subsample` o `imbalanced-learn`).
- "Stratified 5-Fold": `GridSearchCV(cv=5)` estratifica por defecto con clasificadores, pero declara `StratifiedKFold(shuffle=True, random_state=42)` explícitamente (hoy está importado y sin usar).

### 4.7 Interpretabilidad
- Importancias: XGBoost (gain) y RF (MDI) coinciden en `NumOfProducts`, `Age`, `IsActiveMember`; XGBoost añade `zero_balance` y `Geography_Germany`. MDI favorece variables continuas: complementa con **permutation importance** o **SHAP**.
- Una importancia no es un efecto ni una dirección: usa SHAP *dependence plots* o curvas PDP/ICE para decir "a mayor edad, mayor riesgo hasta X".
- Evalúa un baseline interpretable (regresión logística con efectos no lineales de `Age` y `NumOfProducts` como categórica) para ver cuánto ganan realmente los ensambles.

### 4.8 Uso de variables sensibles (P1)
El modelo usa **género, edad y país** como predictores y la narrativa propone acciones por geografía. Añade una auditoría de equidad (tasa de alertas, recall y calibración por género/país/tramo de edad) y discute si deben excluirse variables protegidas en un uso real (regulación bancaria, RGPD).

### 4.9 El caso de uso "early-warning" está sobre-afirmado (P1)
El conjunto es una **foto** (`Exited` es el estado final); no hay dimensión temporal. Variables como `IsActiveMember` o `NumOfProducts` pueden haberse medido cerca o después de la decisión. Decláralo en limitaciones y modera "imminent risk" y "early-warning".

---

## 5. Segmentación con K-Means (P0/P1)

### 5.1 La estructura de clústeres es débil
- Silhouette: 0.1645 en k = 6 y entre 0.141 y 0.155 para el resto de valores: por debajo de ~0.25 se considera estructura débil o inexistente.
- El gráfico de inercia **no tiene codo** (descenso suave de 59,856 a 33,295).
- La línea "Optimal k = 6" se dibujó a mano antes de analizar. Reformula: "k = 6 maximiza el silhouette (0.164), pero la separación es débil" y elige k también por **interpretabilidad** y **estabilidad**.

### 5.2 Las variables elegidas no son las que separan
Los centroides muestran que `CreditScore` (647–656), `Tenure` (4.8–5.2) y `EstimatedSalary` (93k–103k) son casi idénticos entre clústeres: son ruido que diluye la distancia. Lo que realmente separa: **saldo (≈0 vs ≈121k)**, **`IsActiveMember`** (1.00 / 0.00 / 0.88 / 0.00 / 1.00 / 0.07), **`NumOfProducts`** y **edad** (el clúster 2 con 60.9 años).
- K-Means con euclídea asume variables continuas y esféricas; aquí hay una binaria (`IsActiveMember`) y una **bimodal** (`Balance`, 36% de ceros). Pruebas alternativas: K-Prototypes/Gower, GMM con BIC, o segmentar primero por `zero_balance` y después dentro de cada grupo.
- Excluye variables de resultado/ruido y reevalúa silhouette.

### 5.3 Perfiles sin nombre ni vínculo con las recomendaciones (P0)
- `persona_names` está truncado y no se usa.
- La recomendación 1 habla de "Affluent Prime Savers" y la 4 de "young transactors", pero ninguna se asocia a un ID de clúster.
- Lectura de tus centroides (por verificar con perfiles completos):
  - **C0:** saldo alto (≈122k), 100% activo, 1.3 productos, churn 12.6% → clientes activos con saldo alto, bajo riesgo.
  - **C3:** saldo alto (≈121k), 1.0 producto, 0% activo, churn 32.4% → mayor exposición ($128.7M en riesgo).
  - **C5:** saldo alto, 2.2 productos, 7% activo, churn 31.4%.
  - **C2:** edad ≈ 61, 88% activo, churn 32.0%.
  - **C1 y C4:** saldo ≈ 1k–5k, churn 16.4% y 8.2%.
- Nombra segmentos desde los datos, genera una tabla de perfiles con z-scores y mapea cada recomendación al clúster correspondiente. La persona "Affluent Prime Savers" que describes parece corresponder al C3 (saldo alto, un solo producto, inactivo), no a "altos puntajes de crédito".

### 5.4 Validación del clustering
- Estabilidad: bootstrap con *Adjusted Rand Index* entre ejecuciones.
- Visualización: el scatter edad × saldo solapa los clústeres 0, 3 y 5 porque se separan por actividad/productos (no visibles). Usa PCA/UMAP o un heatmap de centroides estandarizados.
- Usa el mismo `n_init` en la búsqueda (10) y en el modelo final (15), o justifícalo.

---

## 6. Síntesis: la matriz 2D (P0)

### 6.1 Puntuación en datos de entrenamiento
`best_xgb.predict_proba(X)` puntúa **todos** los clientes, incluidos el 80% con los que se entrenó el modelo; las probabilidades están sobreajustadas. Usa predicciones *out-of-fold*:

```python
from sklearn.model_selection import cross_val_predict, StratifiedKFold
cv = StratifiedKFold(5, shuffle=True, random_state=42)
df['p_churn_oof'] = cross_val_predict(best_xgb, X, y, cv=cv, method='predict_proba')[:, 1]
```

### 6.2 Riesgo binario vs. pérdida esperada
La exposición se calcula con un corte binario (≥ 0.5) sobre probabilidades inflables (4.4). Mejor: **pérdida esperada = P(churn calibrada) × saldo** (× margen), ordenada por clúster y por cliente. Valida contra el **saldo realmente perdido** observado por clúster.

### 6.3 Lectura de la matriz actual
- Se marcan 2,984 clientes (29.8%) frente al 20.4% real, con ≈ $280M de $765M de depósitos en riesgo (cifras de tu salida).
- Prioridad por exposición: C3 ($128.7M), C0 ($56.5M), C2 ($48.4M), C5 ($46.3M); C1 y C4 casi no mueven depósitos.
- Atención: C0 tiene churn observado de solo 12.6% pero se marca con 18% de riesgo (452 clientes) y la segunda mayor exposición; es un efecto del sobre-riesgo (4.4), no un segmento prioritario.

### 6.4 ROI sin cálculo
La cifra de "5% de balances protegidos" no se deriva de nada. Construye un modelo de valor con parámetros explícitos: tasa de aceptación de la oferta, costo por contacto, margen sobre depósitos, y un análisis de sensibilidad. Presenta el resultado como rango, no como cifra única.

---

## 7. Narrativa y recomendaciones (P1/P2)

- **"Los datos prueban…" / "catastróficos":** el análisis es observacional. Usa "se asocia con".
- **Recomendación 1:** vincularla a un clúster concreto (C3) y retirar "altos puntajes de crédito".
- **Recomendación 2:** reformular según 3.5 (posible artefacto) y proponer una validación (¿se repite en otros periodos?).
- **Recomendación 3:** demostrar "independiente de la antigüedad" (churn por país × tramos de antigüedad) y comprobar la confusión con saldo (3.2).
- **Recomendación 4:** umbral y cadencia coherentes con el código, y condiciones para el despliegue (calibración, equidad, monitoreo).
- Tono: "Hybrid Engine", "Unifying", "high-ROI" prometen más de lo que se demuestra. Un tono sobrio transmite más seniority.
- La integración "supervisado + no supervisado" es superficial: el clúster no se usa para mejorar o diagnosticar el modelo. Añade desempeño por segmento (¿el modelo funciona igual de bien en todos?) o uso del clúster como variable.
- Mueve los resultados clave a un **Executive Summary** con cifras reales al inicio y añade una sección de **limitaciones y próximos pasos**.

---

## 8. Código y reproducibilidad (P2)

- [ ] Quitar imports sin uso: `os`, `silhouette_samples`, `plot_tree`, `classification_report`, `StratifiedKFold` (o usarlo).
- [ ] No silenciar todas las advertencias; retirar `use_label_encoder=False` (obsoleto en XGBoost ≥ 2.0).
- [ ] `Pipeline` + `ColumnTransformer` (one-hot, ratios) para evitar fugas y facilitar el despliegue.
- [ ] Constantes con nombre (umbrales, `k`, semillas) y funciones con docstring y type hints.
- [ ] Sustituir la `lambda` de exposición por una máscara booleana (más clara y rápida).
- [ ] `requirements.txt` con versiones; "Restart & Run All" antes de subir.
- [ ] Peso: 1 MB por imágenes embebidas; aceptable, opcional exportar figuras.

### Estructura de repo sugerida
```
bank-churn-segmentation/
├── README.md              # problema, resultados clave, limitaciones, cómo correr
├── requirements.txt
├── data/                  # o instrucciones de descarga + licencia
├── notebooks/churn_segmentation.ipynb
├── src/                   # features, modelado, evaluación, segmentación
├── reports/figures/
└── model_card.md
```

---

## 9. Checklist ordenado de ejecución

**Primero (P0, ~2–3 h)**
- [ ] Corregir cifras de la sección 1 (generadas por código) y los `$`/LaTeX/`k` perdidos.
- [ ] Arreglar leyenda del scatter y el eje/paleta de los gráficos (sección 2).
- [ ] Corregir la afirmación "Recall > 70%", el "optimizando F1" y las contradicciones de umbral/cadencia.
- [ ] Reformular "k = 6 óptimo" y dar nombre a los personas, mapeados a clústeres.
- [ ] Puntuar la matriz 2D con predicciones *out-of-fold*.

**Después (P1, ~1–2 días)**
- [ ] Misma estrategia de desbalance para los tres modelos o ajuste de umbral; PR-AUC, lift y umbral por costos.
- [ ] Calibración de probabilidades y pérdida esperada en la matriz.
- [ ] Ampliar las rejillas; IC por bootstrap para elegir campeón.
- [ ] Clustering: quitar variables ruido, probar alternativas (K-Prototypes/GMM), estabilidad con ARI, perfiles por z-score.
- [ ] Análisis de equidad; churn por edad/antigüedad/saldo; revisar `NumOfProducts` y la confusión Alemania–saldo.
- [ ] Permutation importance/SHAP.

**Al final (P2)**
- [ ] Auditoría de datos completa, limpieza de código, `Pipeline`, README, `requirements.txt`, tono, "Restart & Run All".

---

## 10. Técnicas y enfoques para darle más valor (P3)

Ordenadas por impacto sobre el portafolio frente a esfuerzo.

### 10.1 Alto impacto, esfuerzo moderado
1. **Decisión basada en valor.** Pérdida esperada por cliente (`P(churn) × saldo × margen`) y por segmento, con un modelo de ROI parametrizable (costo de oferta, aceptación, margen) y análisis de sensibilidad. Es la corrección que más convierte este notebook en una decisión de negocio.
2. **Calibración y umbral por costos.** Entrena sin pesos (o calibra después), curva de calibración, Brier score y barrido de umbral con costo esperado. Presenta una tabla "si contactamos al X% de clientes, capturamos Y% de los churners y $Z".
3. **Lift y ganancia acumulada por decil**, con foco en cuánto del saldo en riesgo se cubre con presupuestos de contacto distintos.
4. **SHAP global y local** con *dependence plots* (edad, productos, saldo) y explicaciones individuales para el equipo comercial.
5. **Segmentación más sólida:** K-Prototypes o GMM/BIC, estabilidad con ARI, heatmap de centroides estandarizados y nombres de segmentos derivados de los datos.
6. **Auditoría de equidad:** tasa de alertas, recall y calibración por género, país y tramo de edad; discusión de variables protegidas.

### 10.2 Análisis y modelado
7. **Investigar la anomalía de `NumOfProducts`.** Modelo con/sin la variable, análisis de `n`, y discusión de si es artefacto o fuga.
8. **Interacciones y confusión:** país × saldo cero, edad × productos, con regresión logística con términos de interacción y PDP/ICE; evita la lectura simplista "Alemania = riesgo".
9. **Baselines y alternativas:** regresión logística regularizada con splines, **Explainable Boosting Machine**, LightGBM/CatBoost con Optuna y restricciones monótonas.
10. **Análisis de supervivencia exploratorio** (Kaplan–Meier por segmento, Cox) con `Tenure` como tiempo y `Exited` como evento; aclara que es una foto transversal, no un seguimiento real.
11. **Desempeño por segmento:** evalúa el modelo dentro de cada clúster (AUC, recall, calibración) y prueba "cluster-then-predict" (¿mejora añadir el clúster como variable?).
12. **Incertidumbre y comparación rigurosa:** validación cruzada repetida, IC por bootstrap y prueba pareada (McNemar/DeLong) para saber si los modelos realmente difieren.

### 10.3 Ir más allá de la predicción
13. **Uplift/persuadibilidad (conceptual).** Distingue "quién se irá" de "a quién ayuda la intervención". Sin datos de tratamiento no se puede estimar, pero explicar la limitación y qué experimento (A/B de retención) haría falta es una señal de madurez.
14. **Valor de vida del cliente (CLV) simple** para priorizar por valor y no solo por saldo.
15. **Escenarios what-if:** simulación de impacto de políticas (p. ej. reducir la proporción de inactivos con saldo alto) con la advertencia de que son escenarios predictivos, no efectos causales.

### 10.4 Comunicación y producto
16. **Mini-app (Streamlit/Gradio):** ingresa un perfil → probabilidad calibrada, segmento, explicación SHAP y acción recomendada.
17. **Dashboard** (Power BI o Streamlit) con la matriz 2D interactiva, simulador de presupuesto y umbral.
18. **Model card y datasheet:** uso previsto, datos, métricas honestas, límites (foto transversal, datos sintéticos), sesgos y equidad.
19. **Ingeniería ligera:** `joblib`, `pytest` para features y evaluación, GitHub Actions que ejecute el notebook o las pruebas, y registro de experimentos (MLflow o CSV).
20. **Monitoreo conceptual:** *drift* (PSI), recalibración periódica y reentrenamiento.

### Combinación recomendada si el tiempo es limitado
Una versión "de portafolio" sólida sale con: **correcciones P0 + 4.1–4.4 + 6.1–6.2 + 10.1 (puntos 1, 2, 5 y 6) + 10.2 (punto 7) + mini-app (16) + model card (18)**.
