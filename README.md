<div align="center">

# Applied Data Science & Machine Learning Portfolio
### End-to-End Enterprise Analytics, Predictive Modeling & Decision Intelligence

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F89939?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB144C?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Inference-green.svg)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

<p align="center">
  <b>A curated showcase of production-grade Data Science, Statistical Inference, and Applied Machine Learning projects solving high-stakes business challenges in Talent Retention, Telematics Pricing, and Retail Banking.</b>
</p>

[Featured Projects](#-portfolio-projects-overview) • [Core Competencies](#-technical-competencies--toolkit) • [Business Impact](#-quantified-business-impact) • [Repository Structure](#-repository-architecture) • [Getting Started](#-getting-started)

</div>

---

## 📌 Executive Summary

This portfolio showcases three comprehensive, end-to-end data science capstones developed following the **PACE Framework** (Plan, Analyze, Construct, Execute) and the industry-standard **CRISP-DM** lifecycle. Rather than generic benchmark exercises, each project focuses on **domain framing, rigorous data hygiene, asymmetric error costs, and actionable strategic decision-making**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               APPLIED ANALYTICS LIFECYCLE                               │
│                                                                                        │
│   [ Business Framing ] ──► [ Data Hygiene & EDA ] ──► [ Statistical Inference ]        │
│                                                               │                        │
│   [ Quantified ROI ]   ◄── [ Model Interpretability ] ◄── [ ML Ensembles & Tuning ]     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Portfolio Projects Overview

| # | Project | Industry / Domain | Core Methods & Algorithms | Key Performance Metrics | Strategic Business Impact |
|:-:|:--------|:------------------|:--------------------------|:------------------------|:--------------------------|
| **01** | [**Salifort Motors HR Analytics**](./01_Salifort_Motors_HR_Analytics) | Automotive & EV / People Analytics | Logistic Regression (Odds Ratios), Decision Trees, Random Forest, XGBoost, Stratified 5-Fold CV | **F1-Score: 0.957**<br>Precision: 99.1%<br>Recall: 92.6%<br>ROC-AUC: 0.981 | **\$1.2M–\$2.5M** annual replacement cost avoidance by preempting voluntary engineer churn and resolving the "Overwork Paradox". |
| **02** | [**Automatidata NYC Taxi Suite**](./02_Automatidata_NYC_Taxi) | Urban Mobility & Telematics / RegTech | Welch's $t$-test (A/B testing), Multiple Linear Regression (OLS), XGBoost, Random Forest | **OLS $R^2$: 0.868** (MAE: \$2.12)<br>XGBoost ROC-AUC: 0.758<br>$t$-stat: 6.87 ($p < 10^{-11}$) | **+\$18.5M** incremental revenue opportunity from contactless card migration; upfront pricing transparency for TLC. |
| **03** | [**Customer Churn & Segmentation**](./03_Customer_Churn_and_Segmentation) | Retail Banking & Wealth Management | Unsupervised K-Means ($k=4$), Inertia / Silhouette, Random Forest, XGBoost Classifier | **Recall: 71.7%** (Top priority)<br>ROC-AUC: 0.861<br>Silhouette Peak: $k=4$ | Protection of **\$105M+** in affluent deposits at risk via a 2D segment-by-risk concierge outreach matrix. |

---

### Project 1: Salifort Motors – Predictive Retention Modeling & Overwork Diagnosis
> **Repository:** [`./01_Salifort_Motors_HR_Analytics`](./01_Salifort_Motors_HR_Analytics) • **Notebook:** [`salifort_employee_churn_prediction.ipynb`](./01_Salifort_Motors_HR_Analytics/salifort_employee_churn_prediction.ipynb)

* **Business Problem:** An EV automotive manufacturer experienced severe talent attrition among senior engineering contributors. Leadership needed to identify the root causes of resignation and deploy an early-warning retention system.
* **Key Findings:**
  * **The Overwork Paradox:** Attrition is sharply bimodal. Top contributors working $>250\text{ hrs/mo}$ with high evaluations ($>0.80$) exit due to extreme burnout (satisfaction $<0.15$). Concurrently, underutilized employees ($\sim 140\text{ hrs/mo}$, 2 projects) exit due to disengagement (54.2% churn).
  * **The "7-Project Trap":** 100% of employees assigned to 7 simultaneous projects quit; turnover drops to **1.1%–6.4%** at the **3–4 project standard**.
  * **Tenure Cliff:** Resignations concentrate heavily at year 5 (45.4% departure rate), preceding the senior vesting window.
* **Technical Highlights:**
  * Handled duplicate records rigorously (20.05% verified duplicate survey submissions).
  * Standardized Logistic Regression benchmark providing interpretable **Odds Ratios ($\text{OR}$)** alongside non-linear tree ensembles.
  * Selected **Random Forest (Tuned)** as the production champion under an asymmetric cost matrix ($C_{\text{FN}} \gg C_{\text{FP}}$), generating **99.14% Precision** (only 4 false alarms) and **92.57% Recall**.

---

### Project 2: Automatidata – NYC Taxi Analytics, Upfront Fare Engine & Tipping Intelligence
> **Repository:** [`./02_Automatidata_NYC_Taxi`](./02_Automatidata_NYC_Taxi) • **Notebook:** [`automatidata_taxi_analytics_and_predictions.ipynb`](./02_Automatidata_NYC_Taxi/automatidata_taxi_analytics_and_predictions.ipynb)

* **Business Mandate:** Consulting engagement for the New York City Taxi and Limousine Commission (TLC) to optimize fleet economics against app-based rideshares across a 5-stage analytics lifecycle.
* **Key Findings & Methodological Architecture:**
  * **Phase 2 (Inferential A/B Test):** Conducted Welch’s two-sample $t$-test confirming that credit card passengers spend significantly more in base fare than cash passengers (\$13.04 vs. \$12.21, $p = 6.84 \times 10^{-12}$), justifying citywide digital payment upgrades.
  * **Phase 3 (Upfront Fare Regression):** Engineered spatio-temporal route features (`mean_distance`, `mean_duration`) across TLC taxi zones. Satisfied Gauss-Markov OLS assumptions (all VIFs $< 2.5$), achieving **$R^2 = 0.868$** with a Mean Absolute Error of just **\$2.12**.
  * **Phase 4 (Tip Maximization ML):** Addressed data leakage by purging post-trip financial attributes; trained Random Forest and XGBoost classifiers to predict generous tippers ($\ge 20\%$) with holdout test ROC-AUC $> 0.758$.

---

### Project 3: Retail Banking – Customer Churn Prediction & Portfolio Segmentation
> **Repository:** [`./03_Customer_Churn_and_Segmentation`](./03_Customer_Churn_and_Segmentation) • **Notebook:** [`customer_churn_and_segmentation.ipynb`](./03_Customer_Churn_and_Segmentation/customer_churn_and_segmentation.ipynb)

* **Business Problem:** Mitigate customer flight in a multinational retail banking portfolio of 10,000 accounts across France, Germany, and Spain, prioritizing retention capital by balance exposure.
* **Key Findings & Architecture:**
  * **The Multi-Product Paradox:** Customers holding 2 products exhibit optimal retention (**7.6% churn**). Conversely, cross-selling 3 or 4 products without proper relationship management triggers catastrophic defection (**82.7% and 100% churn**).
  * **Regional Hazard:** German accounts defect at double the rate of French or Spanish peers (**32.4% vs. ~16%**), carrying higher average deposit balances.
  * **Unsupervised Clustering ($k=4$):** Identified 4 clean behavioral archetypes validated via Elbow inertia and Silhouette analysis (e.g., *Affluent Prime Savers* holding \$128k+ average balance).
  * **2D Strategic Decision Matrix:** Merged supervised XGBoost churn probabilities (**71.7% Recall**) with K-Means balance segments, prioritizing **\$105M+ in at-risk deposit reserves** for high-touch private banking concierge outreach.

---

## 🛠 Technical Competencies & Toolkit

```
Languages & Environments:
  • Python 3.9+ | JupyterLab | PowerShell | Git & GitHub

Data Engineering & Wrangling:
  • pandas | NumPy | Data Leakage Prevention | Feature Scaling | Outlier Diagnostics (IQR)

Statistical Inference & Hypothesis Testing:
  • Welch's Two-Sample t-Test | Confidence Intervals | Gauss-Markov OLS Diagnostics (VIF)
  • Odds Ratios (OR) | Asymmetric Loss Matrices | Decision Threshold Calibration

Supervised Machine Learning:
  • Logistic Regression | Decision Trees | Random Forest | XGBoost Classifier
  • Stratified K-Fold Cross-Validation | GridSearchCV | ROC-AUC | Precision-Recall Curves

Unsupervised Machine Learning:
  • K-Means Clustering | Silhouette Analysis | Inertia Elbow Method | Customer Persona Profiling

Data Visualization & Business Storytelling:
  • Matplotlib | Seaborn | Executive Reporting | Strategic Policy Roadmaps
```

---

## 💼 Quantified Business Impact

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               VALUE DELIVERED BY NUMBERS                               │
├──────────────────────────────┬──────────────────────────────┬──────────────────────────┤
│           \$1.2M–\$2.5M       │           +\$18.5M           │           \$105M+        │
│    Annual Cost Avoidance     │  Annual Revenue Opportunity  │  At-Risk Deposit Defense │
│   Salifort Motors (HRIS)     │     NYC TLC (Automatidata)   │    Retail Banking Matrix │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────┘
```

---

## 📂 Repository Architecture

```text
Projects/
├── .gitignore                           # Git hygiene rules for notebooks, caches & envs
├── LICENSE                              # MIT License
├── README.md                            # Portfolio showcase overview (this document)
├── requirements.txt                     # Unified, version-pinned Python dependencies
│
├── 01_Salifort_Motors_HR_Analytics/     # Project 1: Workforce Retention & Burnout
│   ├── README.md                        # In-depth project report & executive briefing
│   ├── Odds_ratio_explanation.md        # Mathematical deep-dive into Logit & Odds Ratios
│   ├── salifort_employee_churn_prediction.ipynb # Master annotated analytical notebook
│   └── data/
│       └── HR_capstone_dataset.csv      # Employee workforce dataset (n=14,999)
│
├── 02_Automatidata_NYC_Taxi/            # Project 2: NYC TLC Telematics & Fare ML
│   ├── README.md                        # Full 5-stage consulting mandate report
│   ├── automatidata_taxi_analytics_and_predictions.ipynb # Master analytical notebook
│   └── data/
│       ├── 2017_Yellow_Taxi_Trip_Data.csv # Cleaned NYC taxi telematics data
│       └── nyc_preds_means.csv          # Pre-computed route benchmarks
│
└── 03_Customer_Churn_and_Segmentation/  # Project 3: Banking Churn & K-Means Clusters
    ├── README.md                        # Executive report & 2D decision matrix
    ├── customer_churn_and_segmentation.ipynb # Master analytical notebook
    └── data/
        └── Churn_Modelling.csv          # Retail customer accounts dataset (n=10,000)
```

---

## ⚙️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Sebas-toro77/google-advanced-data-analytics-portfolio.git
cd google-advanced-data-analytics-portfolio/Projects
```

### 2. Set Up a Virtual Environment
```bash
# Using Python venv
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch JupyterLab
```bash
jupyter lab
```
Navigate to any of the project directories (`01_`, `02_`, or `03_`) and execute the self-contained notebooks. Each notebook automatically locates its respective data file via relative paths.

---

## 📜 Academic & Program Provenance

The foundation of these projects was established through the **Google Advanced Data Analytics Professional Certificate** and the **Google IT / Python Specializations**, extended with production-grade data hygiene, custom feature engineering, hyperparameter cross-validation, and executive business strategy.

---

## 📬 Contact & Connect

If you'd like to discuss any of the methodologies, business findings, or explore collaboration opportunities:
* **LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/sebastian-toro-arango-biotech/)
* **GitHub:** [@Sebas-toro77](https://github.com/Sebas-toro77/)
* **Email:** [sebastian.toro77@eia.edu.co](mailto:sebastian.toro77@eia.edu.co)

---

<div align="center">
  <sub>Built with passion for data-driven decisions and transparent, ethical machine learning. Star ⭐ this repository if you find it helpful!</sub>
</div>

