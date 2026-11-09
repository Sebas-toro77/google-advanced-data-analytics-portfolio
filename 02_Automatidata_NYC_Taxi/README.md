# Automatidata: NYC Yellow Taxi Analytics & Predictive Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Inference-green.svg)](https://www.statsmodels.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **End-to-End Data Science & Machine Learning Consulting Project:** Synthesizing the complete 5-stage analytics lifecycle for the **New York City Taxi & Limousine Commission (TLC)**—from spatio-temporal data cleaning and statistical A/B testing to Multiple Linear Regression fare estimation and XGBoost tip classification.

---

## 1. Executive Summary & Consulting Mandate

**Client:** New York City Taxi and Limousine Commission (NYC TLC)  
**Engagement Partner:** Automatidata Analytics Practice  
**Core Business Problem:** The yellow medallion taxi industry in New York City faces intense competition from app-based rideshare platforms (Uber, Lyft). To protect driver livelihoods and maintain regulatory compliance, the TLC tasked Automatidata with developing an integrated data intelligence suite to:
1. **Understand Payment Behavior:** Determine if passenger payment methods influence total fare revenue.
2. **Predict Upfront Fares:** Build a transparent, explainable fare estimation engine for riders before trips begin.
3. **Maximize Driver Tips:** Identify the core variables driving generous tipping ($\ge 20\%$) so drivers can optimize their shift locations and timing.

### Projected Business Impact
* **Revenue Uplift:** Shifting cash trips to card payments via in-vehicle tap-to-pay terminals is estimated to generate **\$18.5M in incremental annual revenue** for medallion owners and drivers.
* **Fare Transparency:** Upfront pricing with **MAE < \$2.15** enhances passenger trust and competitiveness against dynamic rideshare pricing.

---

## 2. Dataset Architecture & Hygiene

The project analyzes the official **2017 Yellow Taxi Trip telematics dataset** (22,699 raw trip records) provided by the NYC TLC.

| Feature | Type | Description |
|---------|------|-------------|
| `tpep_pickup_datetime` | Datetime | Telematics timestamp when the taximeter was engaged |
| `tpep_dropoff_datetime` | Datetime | Telematics timestamp when the taximeter was disengaged |
| `passenger_count` | Integer | Number of passengers in vehicle (1–6) |
| `trip_distance` | Float | Elapsed trip distance reported by the taximeter in miles |
| `PULocationID` / `DOLocationID` | Integer | TLC Taxi Zone IDs for pickup and dropoff |
| `payment_type` | Categorical | Numeric payment code: `1 = Credit card`, `2 = Cash`, `3 = No charge`, `4 = Dispute` |
| `fare_amount` | Float | **Target 1:** Base time-and-distance fare calculated by the meter |
| `tip_amount` | Float | Credit card tip amount (cash tips not captured in TLC data) |
| `total_amount` | Float | Total amount charged to passenger (excluding cash tips) |
| `RatecodeID` | Categorical | Final rate code (`1 = Standard`, `2 = JFK Flat Rate`, `3 = Newark`, etc.) |

### Data Hygiene & Outlier Filtering
* **Duration Calculation:** Synthesized `duration` in minutes from timestamps ($T_{\text{dropoff}} - T_{\text{pickup}}$).
* **Sanitization:** Eliminated records with impossible negative or zero values (`trip_distance <= 0`, `duration <= 0`, `fare_amount <= 0`).
* **Telematics Outlier Boundaries:** Filtered non-representative trips ($>50$ miles, $>\$150$ fare, or $>3$ hours duration), preserving **22,699 clean operational trips**.

---

## 3. The 5-Stage Consulting Lifecycle

```
[ Phase 1: EDA & Demand ] ──► [ Phase 2: Statistical A/B Test ] ──► [ Phase 3: Fare Regression ]
                                                                             │
[ Phase 5: Strategy & ROI ] ◄── [ Phase 4: Tip ML Classification ] ◄─────────┘
```

### Phase 1: Spatio-Temporal Demand Dynamics (Courses 2 & 3)
* **Hourly Peak:** Daily demand peaks sharply during the evening rush hour (**16:00 to 19:00**) and maintains sustained volume until 23:00.
* **Weekly Rhythm:** Demand ramps up progressively from Tuesday through Saturday, with **Saturday evening** exhibiting the highest trip volume and average trip distances.

---

### Phase 2: A/B Testing & Inferential Statistics (Course 4)
We conducted a formal two-sample hypothesis test to evaluate whether credit card users pay significantly higher average fares than cash customers:
* **Hypotheses:**
  $$H_0: \mu_{\text{card}} = \mu_{\text{cash}} \quad \text{vs.} \quad H_A: \mu_{\text{card}} \neq \mu_{\text{cash}}$$
* **Statistical Method:** Welch’s two-sample $t$-test (accounting for unequal sample sizes and variances).
* **Empirical Findings:**
  * Credit Card Mean Fare: **\$13.04** ($N = 15,265$, $s = \$10.54$)
  * Cash Mean Fare: **\$12.21** ($N = 7,280$, $s = \$8.91$)
  * Welch’s $t$-statistic: **$6.87$** | $p\text{-value}: \mathbf{6.84 \times 10^{-12}}$
  * 95% Confidence Interval for Difference: **[\$0.59, \$1.07]**
* **Conclusion:** Reject $H_0$. Customers paying by card spend **+6.8% more per trip**, validating the business case for digital payment investments.

---

### Phase 3: Multiple Linear Regression for Fare Prediction (Course 5)
To build an explainable upfront pricing model, we engineered route-level historical benchmarks:
* **Engineered Features:** `mean_distance` and `mean_duration` grouped across historical `PULocationID` - `DOLocationID` pairs.
* **Gauss-Markov Assumptions Verified:**
  * *Multicollinearity:* All Variance Inflation Factors (VIF) $< 2.5$ (well below the critical threshold of 5.0).
  * *Normality & Homoscedasticity:* Confirmed via residual scatter plots and Normal Q-Q plots.
* **TLC Regulatory Integration:** Incorporated statutory flat-rate rule for **JFK International Airport** (`RatecodeID == 2`), locking fare to **\$52.00**.
* **Performance Benchmark:**
  $$\mathbf{R^2 = 0.868} \quad|\quad \mathbf{\text{MAE} = \$2.12} \quad|\quad \mathbf{\text{RMSE} = \$3.74}$$

---

### Phase 4: Generous Tip Classification with Machine Learning (Course 6)
Drivers depend on tips for 20–30% of their take-home compensation.
* **Target Definition:** Binary classification where `generous = 1` if $\text{tip\_percent} \ge 20\%$ of base fare.
* **Data Leakage Prevention:** Strictly removed `tip_amount`, `total_amount`, `fare_amount`, and other post-trip financial figures from the feature matrix $X$.
* **Predictive Features:** Route distance/duration benchmarks, predicted fare, pickup/dropoff zone IDs, day of week dummies, and time-of-day rush flags (`am_rush`, `daytime`, `pm_rush`, `nighttime`).
* **Model Benchmark (Stratified 5-Fold CV on Test Data):**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest (Champion)** 🏆 | **70.2%** | **70.8%** | **74.1%** | **0.724** | **0.762** |
| **XGBoost Classifier** | 69.5% | 70.1% | 73.2% | 0.716 | 0.758 |

---

### Phase 5: Feature Importance & Driver Behavioral Insights
Decomposing the tree ensembles revealed the primary determinants of generous tips:
1. `predicted_fare` & `mean_duration`: Longer trips with higher baseline costs yield higher probabilities of meeting or exceeding the 20% tipping threshold.
2. `DOLocationID` & `PULocationID`: Specific destination zones (e.g., residential Upper East/West Side, Financial District, and airport terminals) exhibit significantly higher tip generosity.
3. `pm_rush` (16:00 – 20:00): Evening commuters and leisure diners tip more generously than morning rush commuters.

---

## 4. Strategic Consulting Recommendations

### 1. Modernize In-Cab Contactless Terminals
* **Finding:** Card transactions yield **+\$0.83 more in base fare per trip** than cash.
* **Action:** Mandate contactless tap-to-pay (Apple Pay, Google Pay) across all 13,000+ medallion cabs to remove cash transaction friction.

### 2. Implement Upfront Fare Estimates
* **Finding:** The Multiple Linear Regression model predicts trip fares with a Mean Absolute Error of **\$2.12** ($R^2 = 0.868$).
* **Action:** Embed this pricing model into TLC passenger apps, offering upfront price estimates to boost rider confidence against competitors.

### 3. Driver Shift Optimization Heatmap
* **Finding:** Generous tips peak during **evening rush hours (`pm_rush`)** on trips originating in Midtown and terminating at residential hubs and airports.
* **Action:** Provide drivers with a real-time dispatch dashboard guiding vehicle repositioning during high-tip windows.

---

## 5. How to Run This Project

### Prerequisites
* Python 3.8+
* Jupyter Notebook or JupyterLab

### Dependencies
Install the required analytical libraries:
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn xgboost
```

### Execution
Run the master notebook directly:
```bash
jupyter notebook automatidata_taxi_analytics_and_predictions.ipynb
```
The notebook automatically resolves data paths relative to `./data/2017_Yellow_Taxi_Trip_Data.csv`.

