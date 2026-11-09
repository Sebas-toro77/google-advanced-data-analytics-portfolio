# Retail Banking: Customer Churn Prediction & Portfolio Segmentation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering%20%26%20Ensembles-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **End-to-End Enterprise Machine Learning Suite:** Combining **Supervised Classification** (Decision Trees, Random Forest, XGBoost) to detect account closure flight-risk with **Unsupervised K-Means Clustering** (Elbow method, Silhouette analysis) to profile customer archetypes across a portfolio of 10,000 retail banking accounts.

---

## 1. Executive Summary & Business Problem

In retail banking, acquiring a new account holder costs **5 to 7 times more** than retaining an existing customer. Furthermore, churn among affluent depositors directly erodes the bank's core deposit base, Net Interest Margin (NIM), and credit underwriting capacity.

### Core Business Objectives
1. **Supervised Churn Early-Warning:** Train and calibrate high-recall classification models to predict individual customer departure (`Exited = 1`) prior to account closure.
2. **Unsupervised Behavioral Archetyping:** Segment the customer portfolio into natural behavioral clusters using K-Means, evaluated via Inertia and Silhouette scores.
3. **The 2D Decision Matrix:** Cross-tabulate predictive churn risk against customer archetypes to prioritize retention interventions by **at-risk deposit balance**, maximizing retention ROI.

---

## 2. Dataset Architecture & Preprocessing

The analysis utilizes a portfolio dataset of **10,000 retail banking customers** operating across France, Germany, and Spain.

| Attribute | Type | Description |
|-----------|------|-------------|
| `CreditScore` | Integer | FICO-equivalent credit score (350 – 850) |
| `Geography` | Categorical | Country of account (`France`, `Germany`, `Spain`) |
| `Gender` | Categorical | Customer gender (`Female`, `Male`) |
| `Age` | Integer | Customer age (18 – 92 years) |
| `Tenure` | Integer | Number of years as a customer with the bank (0 – 10) |
| `Balance` | Float | Current account balance |
| `NumOfProducts` | Integer | Number of distinct bank products held (1, 2, 3, 4) |
| `HasCrCard` | Binary | Credit card ownership flag (0 = No, 1 = Yes) |
| `IsActiveMember` | Binary | Engagement status flag (0 = Inactive, 1 = Active) |
| `EstimatedSalary` | Float | Estimated annual salary |
| `Exited` | Binary | **Target Variable:** Account closure status (0 = Retained, 1 = Churned) |

### Feature Engineering Pipeline
* **Hygiene:** Eliminated surrogate row IDs, customer keys, and surnames to prevent data leakage.
* **Domain Indicators:**
  * `zero_balance`: Binary flag identifying checking-only customers without deposit reserves (representing ~36% of accounts).
  * `balance_to_salary`: Measures accumulated savings relative to annual earnings.
  * `tenure_to_age`: Proportion of customer's adult lifespan associated with the bank.
* **Categorical Encoding:** One-Hot dummy encoding with `drop_first=True` for `Geography` and `Gender`.
* **Standardization:** `StandardScaler` applied to distance-based algorithms (K-Means).

---

## 3. Exploratory Insights: The Multi-Product Paradox & Regional Churn

```
         [ The Multi-Product Paradox ]                     [ Regional Disparity ]
  1 Product:  27.7% Churn Rate                      Germany: 32.4% Churn Rate
  2 Products:  7.6% Churn Rate (Sweet Spot)         France:  16.1% Churn Rate
  3 Products: 82.7% Churn Rate (Critical)           Spain:   16.7% Churn Rate
  4 Products: 100%  Churn Rate (Total Loss)
```

1. **The Multi-Product Trap:** Customers holding **2 products** demonstrate the highest retention (**7.6% churn**). However, holding **3 or 4 products** leads to near-total defection (**82.7% and 100% churn**). Aggressive cross-selling without proper onboarding generates friction and fee dissatisfaction.
2. **German Market Hazard:** German customers defect at **more than double the rate** of French or Spanish customers (**32.4% vs ~16%**), holding substantially higher average account balances.
3. **Age Curve:** Churn risk accelerates rapidly between ages 45 and 60 (median age of churned customers is 45, vs 36 for retained).

---

## 4. Supervised Learning Benchmark (Churn Prediction)

Three non-linear classification architectures were trained using **Stratified 5-Fold Cross-Validation** and evaluated on an unseen held-out test split (20%, $N = 2,000$).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Selection Rationale |
|-------|----------|-----------|--------|----------|---------|---------------------|
| **Decision Tree (Tuned)** | 84.8% | 68.2% | 46.4% | 0.552 | 0.828 | Interpretable baseline |
| **Random Forest (Balanced)** | 85.6% | 65.8% | 61.2% | 0.634 | 0.854 | Robust bagging ensemble |
| **XGBoost Classifier (Champion)** 🏆 | **83.1%** | **56.5%** | **71.7%** | **0.632** | **0.861** | **Highest Recall for High-Value Retention** |

### Why Recall & ROC-AUC Matter Most
In retail banking, failing to flag a customer who withdraws \$100,000 (False Negative) is exponentially more expensive than sending a proactive check-in offer to a customer who wasn't planning to leave (False Positive). **XGBoost** was selected as the **Champion Model** for achieving the highest **Recall (71.7%)** and **ROC-AUC (0.861)**.

### Feature Importance Ranking
1. `NumOfProducts` (Non-linear relationship to retention)
2. `Age` (Primary demographic risk indicator)
3. `Geography_Germany` (Elevated regional vulnerability)
4. `Balance` & `balance_to_salary` (High balance exposure)
5. `IsActiveMember` (Active engagement reduces churn by ~50%)

---

## 5. Unsupervised Learning: Customer Segmentation (K-Means)

Using financial and behavioral attributes, we evaluated optimal clustering solutions across $k \in [2, 8]$:
* **Inertia (Elbow Method):** Elbow inflection occurs clearly between $k = 3$ and $k = 4$.
* **Silhouette Analysis:** Peak cluster cohesion and separation confirmed at **$k = 4$**.

### Customer Portfolio Archetypes ($k = 4$)

| Cluster | Archetype Name | Size (% Share) | Avg Age | Avg Balance | Avg Products | Active % | Churn Rate |
|:-------:|----------------|:--------------:|:-------:|:-----------:|:------------:|:--------:|:----------:|
| **0** | **Young Digital Transactors** | 35.2% | 34.1 yrs | \$12,400 | 1.8 | 62% | 11.2% |
| **1** | **Affluent Prime Savers** | 26.8% | 46.5 yrs | \$128,600 | 1.2 | 48% | 31.4% |
| **2** | **Disengaged Single-Holders** | 22.1% | 41.2 yrs | \$78,200 | 1.0 | 0% | 29.8% |
| **3** | **Established Multi-Holders** | 15.9% | 38.8 yrs | \$64,500 | 2.6 | 78% | 14.5% |

---

## 6. Strategic Synthesis: The 2D Retention Decision Matrix

Cross-tabulating unsupervised segments with supervised flight risk scores allows retail banks to deploy capital where deposit exposure is concentrated:

```
[ High Balance / High Churn Risk ] ──► Persona B (Affluent Prime Savers): Priority Concierge Outreach
[ Low Balance / Low Churn Risk ]  ──► Persona A (Young Transactors): Automated Mobile Nudges
```

### Portfolio Balance at Risk Analysis
* **Cluster 1 (Affluent Prime Savers):** Represents **over \$340M in core retail deposits**. With a **31.4% predicted churn rate**, over **\$105M in deposits are currently at risk of flight**.
* **Cluster 2 (Disengaged Single-Holders):** 100% inactive accounts. Over **\$38M in balances at risk** due to lack of digital engagement.

---

## 7. Actionable Banking Playbooks

### 1. Dedicated Relationship Management for Affluent Savers
* **Profile:** High balance (\$128k+), ages 40–55, high credit scores, single product.
* **Intervention:** Assign dedicated private bankers, provide tiered high-yield savings interest rates, and offer complimentary wealth management consultations.
* **Financial Impact:** Retaining just **10% of at-risk balances in this segment preserves \$10.5M in bank deposits**.

### 2. Multi-Product Onboarding Reform
* **Profile:** Customers holding 3+ products with >80% churn.
* **Intervention:** Halt aggressive product sales pressure. Shift sales KPIs to **90-day active product utilization**. Simplify multi-product fee structures into a single transparent tier.

### 3. Germany Market Retention Initiative
* **Profile:** German depositors churning at 32.4%.
* **Intervention:** Re-evaluate account maintenance fees against German fintech competitors (N26, Revolut) and enhance mobile app features.

### 4. Operational Automated Workflow
* **Monthly Execution:** Score the active customer base via the XGBoost model. Automatically tag customers with Churn Probability $\ge 0.50$ and route them to segment-specific marketing and retention workflows.

---

## 8. How to Run This Project

### Prerequisites
* Python 3.8+
* Jupyter Notebook or JupyterLab

### Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Running the Notebook
```bash
jupyter notebook customer_churn_and_segmentation.ipynb
```
The notebook automatically loads data from `./data/Churn_Modelling.csv`.

