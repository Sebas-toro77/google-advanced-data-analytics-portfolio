# Salifort Motors – Churn Prediction: diagnóstico y plan de mejoras

Documento de trabajo para preparar `salifort_employee_churn_prediction.ipynb` antes de publicarlo en el portafolio.

**Cómo se hizo esta revisión.** Leí todo el código y el texto del notebook y contrasté las afirmaciones del texto con las salidas guardadas (tablas impresas y gráficos). **No re-ejecuté el notebook** ni tuve acceso al CSV, así que todo lo que dependa de re-correr código figura como "por verificar".

**Prioridades**
- **P0** – Errores que contradicen tus propios resultados o rompen la presentación. Corregir sí o sí.
- **P1** – Debilidades metodológicas que un revisor técnico notaría.
- **P2** – Calidad de código, reproducibilidad y presentación del repo.
- **P3** – Extras que aumentan el valor del proyecto (sección 8).

---

## 0. Veredicto

Buena estructura, narrativa clara y decisiones correctas de base (dedup justificada, split estratificado, scaler ajustado solo con train, CV para tuning, comparación contra baseline). Los problemas no están en el esqueleto sino en tres frentes:

1. El texto afirma cifras que **no coinciden con tus gráficos** y una leyenda de gráfico está **invertida**.
2. Hay **incoherencias entre lo que prometes y lo que haces** (matriz de costos, prioridad del recall, umbral 0.65, "Bayesian-level").
3. Varias **explicaciones causales son especulación** presentada como hallazgo.

Además, este dataset es muy común (capstone de Google Advanced Data Analytics y sintético), así que el valor diferencial vendrá del rigor y de los extras de la sección 8.

---

## 1. Cifras del texto vs. salidas reales (P0)

| # | Afirmación en el texto | Lo que muestra tu salida | Acción |
|---|---|---|---|
| 1.1 | 6 proyectos: "~55%" de rotación | **44.9%** | Corregir |
| 1.2 | 3–4 proyectos: "<5%" (EDA) y "<4%" (recomendaciones) | **1.1%** (3) y **6.4%** (4). Además el texto es inconsistente consigo mismo | Corregir y unificar |
| 1.3 | Año 5 de antigüedad: "más de 50%" | **45.4%** | Corregir |
| 1.4 | "Con 6+ años casi nunca se van" | Año 6 = **20.1%**; solo 7, 8 y 10 años = 0% | Corregir; mostrar `n` de cada grupo (los grupos de 7+ años probablemente son pequeños) |
| 1.5 | Salario bajo "casi el doble" que alto | **20.5% vs 4.8%** (más de 4 veces) | Corregir |
| 1.6 | "Trampa de los 7 proyectos" omite el grupo de 2 proyectos | 2 proyectos = **54.2%**, el segundo extremo de la curva en U | Mencionarlo: es lo que justifica el "U-curve" que afirmas |
| 1.7 | Recomendación 2: "tope de 4 a 5 proyectos" y título "4-Project Standard" | 5 proyectos = 15.4% frente a 6.4% (4) y 1.1% (3) | Alinear título y política con los datos (3–4 como zona óptima) |

**Verificadas y correctas:** 14,999 filas brutas; 3,008 duplicados (20.05%); 11,991 filas tras dedup; 16.6% de rotación en train y test; 145 empleados con 7 proyectos y los 145 se fueron; ORs de la regresión logística tal como se citan.

**Por verificar (no están impresas en el notebook):** las desviaciones estándar citadas en la interpretación de ORs (~1.3 años, ~49.9 h, ~0.25) y los límites numéricos de los "clusters" A y B.

**Corrección estructural.** Hay ~30 números escritos a mano en celdas Markdown. Calcúlalos y muéstralos con código para que no se desincronicen:

```python
rate = df_clean.groupby('number_project')['left'].mean().mul(100).round(1)
n    = df_clean.groupby('number_project').size()
print(pd.DataFrame({'n': n, 'turnover_%': rate}))
```

Y en las conclusiones usa `display(Markdown(f"... {rate[6]:.1f}% ..."))` o una tabla generada.

---

## 2. Errores de gráficos y renderizado (P0)

- [ ] **Leyenda invertida en el scatter horas vs. satisfacción.** Los puntos naranjas son quienes se fueron (Cluster A y B, anotados como tales), pero la leyenda dice "Stayed" para naranja y "Left" para azul. Causa: `ax1.legend(labels=[...])` reasigna etiquetas a los handles en un orden distinto al esperado. Solución: no pases `labels=`; deja que seaborn genere la leyenda a partir de `hue`, o mapea `left` a texto (`df['status'] = df['left'].map({0:'Stayed',1:'Left'})`) y usa `hue='status'`. También desaparece de la leyenda la línea "Standard Threshold".
- [ ] **Sobreposición de puntos** en ese scatter: los azules tapan a los naranjas. Dibuja primero los que se quedaron y después los que se fueron, o usa `alpha` menor / hexbin / 2D-KDE.
- [ ] **Montos en dólares perdidos**: `\,000â€“\,000+` y `\.2M and \.5M`. Los `$` se interpretan como LaTeX y hay un problema de codificación (`â€“` es un guion "–" mal decodificado). Escapa como `\$` y guarda el archivo en UTF-8.
- [ ] **Carácter de control** en `\u0007verage_montly_hours` (la nota del typo). Reescribe esa celda Markdown.
- [ ] **Ruta con backslash**: `"data\HR_capstone_dataset.csv"` solo funciona en Windows. Usa `Path("data") / "HR_capstone_dataset.csv"`.
- [ ] Verificar que el resto de celdas Markdown con `\n`/`\numpy` no tengan secuencias de escape rotas al renderizar en GitHub.

---

## 3. Hallazgos metodológicos (P1)

### 3.1 Elección del "campeón" mal justificada
- Sección 5 declara Recall y F1 como criterios principales, pero tuneas con ROC-AUC y declaras Random Forest campeón sin explicarlo.
- Resultados en test: RF recall **0.9257** (el más bajo de los tres árboles), DT y XGB 0.9277; XGB tiene el mejor AUC (**0.9851**); RF gana en precisión (0.9914) y F1 (0.9574).
- Matriz de confusión de RF: TN 2496, FP 4, **FN 37**, TP 461. Las diferencias entre modelos son de uno o dos casos sobre 498 positivos y las curvas ROC se superponen casi por completo.
- **Acción:** añade una celda "Selección del modelo" que explique el criterio real. Opciones honestas: (a) tratarlos como técnicamente empatados y elegir por interpretabilidad/simplicidad (el árbol de profundidad 6 rinde casi igual); (b) elegir por costo esperado (ver 3.3). Reporta incertidumbre (bootstrap sobre test) en lugar de decimales.

### 3.2 Umbral de decisión incoherente
- Todas las predicciones usan el umbral por defecto 0.5, pero la recomendación 5 propone intervenir con probabilidad ≥ 0.65, que **reduce el recall** que dices priorizar. Ese 0.65 no se calcula en ninguna parte.
- La regresión logística (recall 0.187 a umbral 0.5) queda penalizada por el umbral, no solo por el modelo; su AUC (0.83) es razonable.
- **Acción:** curva precision-recall, PR-AUC (hay desbalance 16.6%) y barrido de umbrales. Justifica el umbral elegido con el costo (3.3) o cámbialo en la recomendación.

### 3.3 La "matriz de costos asimétrica" prometida no existe
- La sección "Execute" de la introducción y la sección 5 hablan de ella, pero no hay código que la calcule.
- **Acción:** parametriza `cost_FN` y `cost_FP` (aunque sean supuestos explícitos), calcula el costo esperado por umbral y elige el óptimo. Añade análisis de sensibilidad (¿cambia el umbral si `cost_FN/cost_FP` pasa de 3 a 10?). Es la corrección con mejor relación esfuerzo/valor.

### 3.4 Realismo del problema: `satisfaction_level` y la ilusión de "alerta temprana"
- `satisfaction_level` es la variable más importante (0.416 en RF) y probablemente se mide cerca de la decisión de irse. Con métricas de ~98% de accuracy, el notebook vende un sistema de alerta temprana que el dato no necesariamente soporta (es una foto transversal de quienes ya se fueron y quienes siguen).
- **Acción:** entrena y reporta una versión **sin** `satisfaction_level` (y quizá sin `last_evaluation`), y discute cuánta señal queda. Añade una sección de limitaciones con esto explícito. Evita términos como "production champion" y "HRIS deployment" sin ese matiz.

### 3.5 Regresión logística: interpretación sobre-afirmada
- Usas `LogisticRegression` de scikit-learn (regularizada L2 por defecto), sin intervalos de confianza ni p-values, pero hablas de "evidencia empírica" y de "demuestra". **Acción:** ajusta con `statsmodels` (`Logit`) para obtener IC 95% de los ORs, o modera el lenguaje.
- **Dummies estandarizadas:** "1 desviación estándar de `department_sales`" no tiene interpretación. Estandariza solo las variables continuas (usa `ColumnTransformer`).
- **Contradicción con tu EDA:** el OR de `tenure` (1.51, relación lineal) choca con la curva en forma de campana por antigüedad; lo mismo con `number_project` (ya lo señalas). Prueba antigüedad y proyectos como categóricas, o términos cuadráticos/splines. Así el baseline deja de ser un hombre de paja.
- La conclusión "la rotación es por política de carga, no por cultura departamental" se apoya solo en ORs cercanos a 1 de unas pocas dummies. Hazla con un test conjunto (likelihood ratio o permutation importance agrupada de `department`).
- Prueba `class_weight='balanced'`.

### 3.6 Explicaciones causales especulativas
Frases presentadas como hallazgos que los datos no pueden respaldar:
- Accidentes laborales protegen porque los empleados "pasan a roles de soporte, reciben monitoreo de salud o temen perder el seguro".
- Los mejor evaluados son "blanco de headhunters".
- Los de 3–5 años "perciben un estancamiento y buscan cambios laterales en el mercado automotriz".
- Los empleados que se van "perciben explotación institucional".

**Acción:** marcarlas explícitamente como *hipótesis* ("una posible explicación, no verificable con este dataset es…"). Quita las que no aporten.

### 3.7 Políticas recomendadas vs. evidencia
- Los topes (200 h, 4–5 proyectos) se derivan de asociaciones, no de efectos causales; los que se quedan también trabajan 200–275 h (se ve en la densidad). Añade la advertencia "asociación ≠ causalidad" y presenta las políticas como piloto a evaluar.
- La recomendación 5 cita un ROI ("25 ingenieros → entre X y Y millones") sin cálculo ni datos de salario en el dataset. Pon el cálculo y sus supuestos con la fórmula, o elimina la cifra.
- "50%–200% del salario anual" como costo de reemplazo necesita fuente o matiz de "rango habitual en la literatura".

### 3.8 Interpretabilidad
- MDI/Gini favorece variables continuas y de alta cardinalidad. Complementa con **permutation importance** (en test) o **SHAP**.
- El gráfico de XGBoost dice "(Weight)", pero creo que el valor por defecto de `feature_importances_` en la API de scikit-learn de XGBoost es *gain*. Verifícalo con `best_xgb.importance_type` y corrige la etiqueta.
- `overworked` (>200 h) se crea pero nunca se discute ni aparece en importancias. Decide: usarla en el análisis o eliminarla (es redundante con `average_monthly_hours` para árboles). Documenta el umbral: 200 h/mes ≈ 46 h/semana con 4.33 semanas; el curso de Google usa una base de ~166.7 h (40 h × 50 semanas / 12).

### 3.9 Hiperparámetros en el borde de la rejilla
Varios óptimos caen en un extremo de lo explorado, lo que sugiere que la búsqueda no está agotada:
- RF: `max_depth=8` (valor más bajo de [8, 12, None]), `min_samples_split=2`, `max_features=0.6` (el más alto).
- XGB: `learning_rate=0.05` y `max_depth=4` (los más bajos), `subsample=1.0` y `colsample_bytree=1.0` (los más altos), `n_estimators=200` (el más alto).
- DT: `min_samples_leaf=2` (el más bajo).

**Acción:** ampliar rejillas hacia ese lado o usar búsqueda aleatoria/Optuna. Y reemplazar "Bayesian-level grid tuning" por una descripción exacta ("grid search con CV de 5 folds").

### 3.10 Otros
- **Selección sobre el test set:** comparas y eliges el campeón en el mismo test. Mejor: validación cruzada anidada o reservar un validation set; o al menos reconocerlo.
- **Dedup:** documenta que es una *suposición* (sin ID de empleado no se puede saber si son registros repetidos). Haz un análisis de sensibilidad: resultados con y sin dedup, y qué proporción de duplicados pertenece a `left=1` (por verificar).
- **Outliers de `tenure`:** los cuantificas (824 filas, 6.87%) pero no documentas la decisión. Escribe que se conservan y por qué (árboles robustos), o evalúa la logística con y sin ellos.
- **Tercer grupo de renunciantes:** en el scatter se aprecian renunciantes con satisfacción alta (≈0.7–0.9) y muchas horas, que tu texto no menciona (solo describes los clusters A y B). Revísalo y descríbelo.
- **Intervalos para tasas pequeñas:** muestra `n` y un IC (Wilson) en los gráficos por antigüedad y proyectos.
- **Overfitting:** reportas CV-AUC y test-AUC; son consistentes (0.973/0.976, 0.983/0.981, 0.987/0.985). Menciónalo como evidencia de que no hay sobreajuste visible.

---

## 4. Narrativa y redacción (P2)

- Tono demasiado promocional: "rigorous", "Bayesian-level", "production champion", "profound". Un tono sobrio y preciso transmite más seniority.
- Mueve las conclusiones y resultados clave a un **Executive Summary con cifras reales** al inicio (3–5 bullets con números calculados).
- Reduce la prosa repetitiva de la interpretación de ORs: una tabla (feature, OR, IC, lectura) comunica mejor que párrafos largos.
- Declara en la introducción que el dataset es **ficticio/sintético** (capstone de Google Advanced Data Analytics), con enlace y licencia.
- Añade una sección final **"Limitaciones y próximos pasos"** (ver 3.4, 3.6, 3.7).

---

## 5. Código y reproducibilidad (P2)

- [ ] Quitar imports sin uso: `os`, `scipy.stats`, `plot_tree`, `StratifiedKFold`, `classification_report`.
- [ ] No silenciar todas las advertencias con `warnings.filterwarnings('ignore')`; filtra solo las concretas. Retirar `use_label_encoder=False` (obsoleto en XGBoost ≥ 2.0).
- [ ] Usar `Pipeline` + `ColumnTransformer` (escalado, one-hot y codificación ordinal de `salary` dentro del pipeline) para evitar fugas y facilitar el despliegue.
- [ ] `random_state` ya está fijado: bien. Fijar también versiones en `requirements.txt`.
- [ ] Función de evaluación: completar type hints y docstring; añadir `average_precision_score` (PR-AUC) y Brier score.
- [ ] Reordenar imports por bloques y quitar `%matplotlib inline` redundante si no hace falta.
- [ ] Ejecutar el notebook completo "Restart & Run All" antes de subirlo, para confirmar que corre sin errores en un entorno limpio.
- [ ] Peso del archivo: 1.3 MB por imágenes embebidas. Es aceptable en GitHub; opcional limpiar salidas y exportar figuras a `reports/figures/`.

---

## 6. Empaquetado del repositorio (P2)

Estructura sugerida:

```
salifort-churn/
├── README.md                 # problema, resultados clave, cómo correr
├── requirements.txt
├── data/                     # o instrucciones de descarga + licencia
├── notebooks/
│   └── salifort_churn.ipynb
├── src/                      # funciones reutilizables (opcional)
├── reports/figures/
└── model_card.md             # (ver sección 8)
```

El README debe tener: pregunta de negocio, 3–5 hallazgos con cifras, métrica(s) elegidas y por qué, limitaciones, cómo reproducir, y un enlace a una vista previa (nbviewer o imagen del dashboard).

---

## 7. Checklist ordenado de ejecución

**Primero (P0, ~1–2 h)**
- [ ] Corregir cifras de la sección 1 (y generarlas por código).
- [ ] Arreglar leyenda invertida y orden de dibujo del scatter.
- [ ] Arreglar `$`, codificación, carácter de control y ruta.

**Después (P1, ~1 día)**
- [ ] Selección explícita del modelo + intervalos de confianza por bootstrap.
- [ ] Curva PR, PR-AUC, barrido de umbrales y matriz de costos con sensibilidad.
- [ ] Modelo sin `satisfaction_level` y sección de limitaciones.
- [ ] Logística con `statsmodels`, dummies sin estandarizar, `tenure`/`number_project` no lineales.
- [ ] Marcar hipótesis vs. hallazgos; revisar políticas y ROI.
- [ ] Ampliar rejillas de hiperparámetros.
- [ ] Permutation importance / SHAP y corregir etiqueta de XGBoost.

**Al final (P2)**
- [ ] Limpieza de código, `Pipeline`, `requirements.txt`, README, tono del texto.
- [ ] Restart & Run All.

---

## 8. Técnicas y enfoques para darle más valor (P3)

Ordenadas por impacto sobre el portafolio frente a esfuerzo.

### 8.1 Alto impacto, esfuerzo moderado
1. **Decisión basada en costos.** Matriz de costos parametrizable, curva de costo esperado frente al umbral y análisis de sensibilidad. Convierte el notebook de "modelo con buen AUC" en "decisión de negocio".
2. **Escenario realista sin variables de encuesta.** Compara modelo completo vs. sin `satisfaction_level`/`last_evaluation`. Muestra criterio sobre qué es predecible *antes* de que alguien decida irse.
3. **SHAP global y local.** Beeswarm para importancia, *dependence plots* para ver la forma no lineal de `number_project`, `tenure` y `average_monthly_hours`, y explicaciones individuales ("por qué este empleado tiene riesgo alto").
4. **Análisis de errores.** Inspecciona los 37 falsos negativos del RF: ¿qué perfil tienen? (hipótesis por verificar: renunciantes con satisfacción alta). Es probablemente el hallazgo más interesante del proyecto.
5. **Calibración de probabilidades.** Curva de calibración y Brier score; recalibrar con isotónica o Platt si hace falta. Necesaria si vas a usar un corte como "probabilidad ≥ X".
6. **Incertidumbre y comparación rigurosa.** Validación cruzada repetida estratificada, IC por bootstrap en test y un test pareado (McNemar o DeLong) para decir si los modelos realmente difieren.

### 8.2 Modelado
7. **Baselines más justos e interpretables.** Regresión logística con splines/categóricas, **Explainable Boosting Machine** (`interpret`) o un GAM: muestran curvas por variable y suelen competir con los ensambles.
8. **LightGBM/CatBoost** con búsqueda bayesiana real (Optuna) y, si tiene sentido, **restricciones monótonas** (p. ej. riesgo no creciente con la satisfacción).
9. **Segmentación de perfiles** con clustering (GMM/K-Means) sobre horas, proyectos, satisfacción y evaluación; calcula tasa de rotación por clúster y valida formalmente los grupos "burnout" e "infrautilizado" que hoy solo se ven a ojo.
10. **Vista de supervivencia exploratoria** (Kaplan–Meier / Cox con `tenure` como tiempo). Aclara que es una foto transversal, no un seguimiento real, así que se presenta como exploración del "riesgo por antigüedad".
11. **Contrafactuales y simulación de políticas** (p. ej. DiCE o una simulación: "si limitamos a 4 proyectos, ¿cuánto baja el riesgo predicho?"), siempre con la advertencia de que son escenarios predictivos, no efectos causales.

### 8.3 Responsabilidad y comunicación
12. **Uso responsable.** Sección de ética: riesgos de usar predicción de renuncia para penalizar o discriminar, privacidad, auditoría de desempeño por `department` y `salary`. Distingue "intervenir para ayudar" de "vigilar".
13. **Model card y datasheet.** Un `model_card.md` corto (uso previsto, datos, métricas, limitaciones, sesgos) es una señal fuerte de madurez.
14. **EDA con rigor estadístico.** Tests (chi-cuadrado, Mann–Whitney) con tamaños de efecto (V de Cramér, Cliff's delta) y IC en las tasas, en lugar de solo barras.

### 8.4 Entrega y producto
15. **Mini-app** en Streamlit o Gradio: formulario con los datos de un empleado → probabilidad, explicación SHAP y recomendación. Un enlace desplegado vale más que mil celdas.
16. **Dashboard de BI** (Power BI o Streamlit) con segmentos de riesgo, tasa por departamento y simulador de umbral/costo.
17. **Ingeniería ligera:** modelo serializado con `joblib`, pruebas mínimas (`pytest`) para el preprocesamiento, GitHub Actions que ejecute el notebook o los tests, y versionado de experimentos (MLflow o un CSV de resultados).
18. **Monitoreo conceptual:** una sección breve sobre drift de datos y reentrenamiento (qué vigilarías y con qué métricas), aunque no lo implementes.

### Combinación recomendada si el tiempo es limitado
Una versión "de portafolio" sólida y diferenciada sale con: **corrección P0 + 3.1–3.4 + 8.1 (puntos 1–4) + mini-app (15) + model card (13)**.
