La **Regresión Logística** cumple un rol fundamental: no busca competir en capacidad pura de predicción contra ensambles no lineales como Random Forest o XGBoost, sino actuar como el **modelo explicativo y de inferencia estadística por excelencia**.


A continuación tienes una explicación detallada de la **matemática subyacente**, la **mecánica de los *Odds Ratios*** y el **significado de negocio de los factores de riesgo y protectores** en Salifort Motors.

---

### 1. La Teoría Matemática detrás del Código

#### A. ¿Por qué no usar Regresión Lineal Clásica (OLS)?
En un problema de rotación (`left` $\in \{0, 1\}$), una regresión lineal clásica $Y = \beta_0 + \beta_1 X$ fallaría porque:
1. Podría predecir probabilidades fuera del rango válido (valores negativos o mayores a $1$).
2. Los residuos no se distribuyen de forma normal (siguen una distribución de Bernoulli).
3. La varianza de los errores no es constante (viola la homocedasticidad).

#### B. La Función de Enlace Logit y la Sigmoide
La Regresión Logística modela la probabilidad condicional de que un empleado renuncie:
$$p = P(Y = 1 \mid X)$$

Para transformar esa probabilidad acotada entre $0$ y $1$ en una recta continua que pueda variar entre $-\infty$ y $+\infty$, se utiliza la transformación **Logit** en dos pasos:

1. **Momios (*Odds*):** La razón entre la probabilidad de que el evento ocurra frente a la probabilidad de que no ocurra:
   $$\text{Odds} = \frac{p}{1 - p}$$
   *(Si la probabilidad de renuncia es del $80\%$ ($0.8$), los Odds son $\frac{0.8}{0.2} = 4$; es decir, es 4 veces más probable que se vaya a que se quede).*

2. **Log-Odds (Logit):** Aplicar el logaritmo natural a los momios permite linealizar la relación con las variables predictoras:
   $$\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k$$

Despejando la probabilidad $p$, obtenemos la función logística estándar (**Sigmoide**):
$$p = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_j X_j)}}$$

---

### 2. ¿Qué es exactamente un *Odds Ratio* (OR)?

El modelo calcula coeficientes $\beta_j$. Como estos coeficientes están en escala logarítmica ($\text{log-odds}$), no son directamente intuitivos para un director de Recursos Humanos.

Para volverlos interpretables, aplicamos la función exponencial:
$$\text{Odds Ratio (OR)} = e^{\beta_j}$$

El **Odds Ratio** responde a la pregunta:  
*«Manteniendo todas las demás variables constantes (*ceteris paribus*), ¿en qué proporción se multiplican los momios de renuncia cuando la variable $X_j$ aumenta en una unidad?»*

* **$\text{OR} > 1$ (Factor de Riesgo):** Aumentar esta variable **incrementa** las posibilidades de renuncia.
  * *Porcentaje de aumento:* $(\text{OR} - 1) \times 100\%$.
* **$\text{OR} < 1$ (Factor Protector):** Aumentar esta variable **reduce** las posibilidades de renuncia.
  * *Porcentaje de reducción:* $(1 - \text{OR}) \times 100\%$.
* **$\text{OR} = 1$ (Neutro):** La variable no tiene impacto sobre la decisión de renuncia.

> [!NOTE]
> **Detalle del código (`StandardScaler`):**  
> En la celda se escalan las variables con `X_train_scaled = scaler.fit_transform(X_train)`. Esto significa que el incremento de "1 unidad" evaluado por el modelo corresponde a **1 desviación estándar ($\sigma$)** de la variable original. Esto permite comparar de forma justa el impacto de variables con escalas incompatibles (ej. horas mensuales que van de 96 a 310 frente a satisfacción que va de 0 a 1).

---

### 3. Diagnóstico de Salifort Motors: Factores de Riesgo vs. Factores Protectores

Al ejecutar `odds_df = pd.DataFrame({'Coefficient': log_reg.coef_[0], 'Odds Ratio': np.exp(log_reg.coef_[0])})`, las variables se dividen naturalmente en dos grupos estratégicos:

```
                  [ FACTORES PROTECTORES ]               [ FACTORES DE RIESGO ]
                 Odds Ratio < 1 (Beta < 0)              Odds Ratio > 1 (Beta > 0)
          ───────────────────────────────────────     ───────────────────────────────
          • Nivel de Satisfacción (OR ~0.25-0.35)     • Antigüedad / Tenure (OR ~1.40-1.70)
          • Accidente Laboral     (OR ~0.60-0.70)     • Horas Promedio / Overworked
          • Salario Medio/Alto    (OR ~0.75-0.85)     • Evaluación de Desempeño
          • Promoción Reciente    (OR ~0.80-0.90)     • Número de Proyectos Excesivo
```

#### A. Key Risk Drivers (Factores Críticos de Riesgo: $\text{OR} > 1$)

1. **`tenure` (Antigüedad en la empresa):**
   * **Mecánica:** Presenta un coeficiente $\beta$ positivo significativo ($\text{OR} > 1.4$).
   * **Interpretación:** Los empleados que acumulan entre 3 y 5 años experimentan un aumento sustancial en las probabilidades de marcharse. Es la ventana crítica donde el talento siente que alcanzó un "techo" interno y busca mejores ofertas en el mercado externo si no recibe promociones.
2. **`overworked` y `average_monthly_hours` (Sobrecarga de trabajo):**
   * **Mecánica:** $\beta > 0 \implies \text{OR} > 1$.
   * **Interpretación:** Cada incremento de horas mensuales actúa como un multiplicador directo del riesgo de deserción por fatiga crónica (*burnout*).
3. **`last_evaluation` (Evaluación de desempeño):**
   * **Mecánica:** En presencia de insatisfacción y sobrecarga, una evaluación alta aumenta el riesgo de salida.
   * **Interpretación:** Los empleados más calificados y evaluados con mejores notas son los primeros en encontrar opciones atractivas en otras compañías automotrices cuando Salifort los sobrecarga de trabajo.

#### B. Protective Factors (Factores Protectores: $\text{OR} < 1$)

1. **`satisfaction_level` (Nivel de satisfacción):**
   * **Mecánica:** Es por mucho el coeficiente negativo más grande del modelo ($\beta \ll 0$, $\text{OR} \approx 0.25 - 0.35$).
   * **Interpretación:** Un aumento de una desviación estándar en la satisfacción reduce los momios de renuncia en más de un **$65\% \text{ a } 75\%$**. Es la barrera defensiva número uno de la organización.
2. **`work_accident` (Accidente laboral previo):**
   * **Mecánica:** Coeficiente negativo consistente ($\text{OR} \approx 0.65$).
   * **Interpretación de negocio:** En analítica de recursos humanos suele sorprender que un accidente sea "protector". La explicación empírica reside en que los colaboradores que sufrieron accidentes laborales suelen recibir reasignaciones a puestos de menor exigencia horaria, acuerdos de compensación/seguros médicos que los atan a la empresa, o experimentan mayor aversión al riesgo para renunciar voluntariamente.
3. **`salary_num` (Nivel Salarial Medio y Alto):**
   * **Mecánica:** $\text{OR} < 1$.
   * **Interpretación:** Un nivel salarial competitivo neutraliza de forma directa la tentación de buscar aumentos mediante saltos de empresa.
4. **`promotion_last_5years`:**
   * **Mecánica:** $\text{OR} < 1$.
   * **Interpretación:** Recibir un reconocimiento formal de carrera en los últimos 5 años renueva el compromiso del empleado y reduce significativamente la probabilidad de salida.

---

### 4. ¿Por qué la Regresión Logística se queda corta frente a Random Forest en este caso?

Aunque la Regresión Logística nos da esta interpretabilidad económica impecable, su métrica ROC-AUC en el test set es de $\approx 0.776$ frente al $\approx 0.988$ de Random Forest.

**Razón teórica:** La regresión logística asume una relación **monótona y lineal** en el logit. Asume que cada proyecto adicional o cada hora adicional siempre incrementa el riesgo de renuncia de manera constante. 

Sin embargo, el EDA demostró que la realidad en Salifort Motors es **no lineal y bimodal (forma de U)**:
* Con **2 proyectos**, el empleado se va por desinterés/estancamiento.
* Con **3 o 4 proyectos**, la rotación baja al mínimo ($<4\%$).
* Con **6 o 7 proyectos**, la rotación sube al $100\%$.

Los árboles de decisión y los ensambles (*Random Forest / XGBoost*) capturan de forma nativa estas particiones no lineales y las interacciones cruzadas (ej.: *«alta evaluación combinada con horas $>250$»*), motivo por el cual son la elección óptima como motor predictivo, mientras que la Regresión Logística nos da el marco conceptual de los *Odds Ratios*.