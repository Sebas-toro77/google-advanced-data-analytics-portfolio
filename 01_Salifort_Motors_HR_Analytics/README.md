# Salifort Motors: Employee Churn Prediction & Retention Analytics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **End-to-End People Analytics Project:** Diagnosing organizational burnout, predicting voluntary turnover with ensemble machine learning, and designing data-backed workforce retention policies for Salifort Motors.

---

## 1. Executive Summary & Business Context

**Salifort Motors**, a multinational automotive manufacturer designing electric and autonomous vehicles, faces an escalating voluntary turnover crisis. Replacing specialized engineers, software developers, and product leaders costs between **50% and 200% of their annual salary** in direct recruitment, sign-on bonuses, onboarding friction, and missed project milestones.

### Project Goals
1. **Root-Cause Analysis:** Diagnose the organizational and behavioral drivers behind employee resignations.
2. **Predictive Early Warning:** Train and tune high-recall classification models to identify employees at high risk of departure 3 to 6 months in advance.
3. **Strategic HR Interventions:** Deliver concrete, quantifiable policy recommendations to retain high performers and optimize productivity.

### Business Impact
By deploying the recommended **Random Forest Early-Warning System** and enforcing the proposed workload policies, Salifort Motors can preempt the resignation of an estimated **25 to 50 key contributors annually**, generating **\$1.2M to \$2.5M in cost avoidance**.

---

## 2. Dataset Architecture & Data Hygiene

The dataset contains **14,999 employee records** across 10 workplace attributes. 

| Attribute | Type | Description |
|-----------|------|-------------|
| `satisfaction_level` | Float | Self-reported employee job satisfaction (0.00 – 1.00) |
| `last_evaluation` | Float | Score from the employee's most recent performance review (0.00 – 1.00) |
| `number_project` | Integer | Number of projects assigned to the employee |
| `average_monthly_hours` | Integer | Average hours worked per month (standard: ~160 hrs/mo) |
| `tenure` | Integer | Number of years with the company |
| `work_accident` | Binary | Whether the employee experienced a workplace accident (0 = No, 1 = Yes) |
| `left` | Binary | **Target Variable:** Voluntary departure (0 = Stayed, 1 = Left) |
| `promotion_last_5years`| Binary | Whether the employee was promoted in the last 5 years (0 = No, 1 = Yes) |
| `department` | Categorical | Department (Sales, Technical, Support, IT, R&D, Product, Marketing, etc.) |
| `salary` | Categorical | Relative compensation tier (`low`, `medium`, `high`) |

### Data Hygiene & Preprocessing Highlights
* **PEP 8 Standardization:** Renamed non-compliant features and corrected typographical errors (e.g., `average_montly_hours` $\rightarrow$ `average_monthly_hours`, `time_spend_company` $\rightarrow$ `tenure`).
* **Deduplication:** Identified and removed **3,008 exact duplicate records** (representing 20.05% of raw survey responses). Removing duplicates eliminated data leakage between training and testing splits, ensuring genuine out-of-sample generalization.
* **Outlier Treatment:** Detected tenure values extending up to 10 years (upper IQR threshold = 5.5 years). Evaluated via robust scaling for linear models and native partitioning in tree ensembles.
* **Feature Engineering:** Created the `overworked` indicator flag ($\ge 200$ hours/month), ordinal integer mapping for salary hierarchy, and One-Hot dummy encoding for departments.

---

## 3. Key Findings from Exploratory Data Analysis (EDA)

```
       [ Cluster A: Burnout Stars ]                     [ Cluster B: Underutilized ]
  Hours > 250/mo | Projects 6-7 | Eval > 0.8       Hours ~140/mo | Projects 2 | Eval ~0.5
  Satisfaction < 0.15 | Churn Rate ~98%            Satisfaction ~0.40 | Churn Rate ~40%
```

### 1. The Overwork Paradox
Attrition does not follow a simple linear relationship with workload; it is sharply bimodal:
* **The Overworked Star Contributors (Cluster A):** Top talent working **250 to 310 hours/month** (55 to 65+ hours/week) with exemplary evaluations ($>0.80$) but rock-bottom satisfaction ($<0.15$). They leave due to unmanageable workloads and lack of recognition.
* **The Disengaged / Underutilized (Cluster B):** Employees working **130 to 155 hours/month** on only 2 projects with mediocre evaluations. They leave due to stagnation, lack of direction, or pre-emptive performance pressure.

### 2. The "7-Project Trap"
* **100% of employees assigned to 7 simultaneous projects left the company.**
* **~55% of employees assigned to 6 projects left.**
* The optimal retention window is **3 to 4 projects**, where attrition plummets below **4%**.

### 3. The 5-Year Tenure Cliff
Turnover peaks dramatically between **years 3 and 5**, with year 5 experiencing a **>50% departure rate**. Employees who remain past year 6 almost never leave, demonstrating that the 3–5 year window is the critical inflection point for career development and equity vesting.

---

## 4. Modeling Benchmark & Evaluation

Four distinct modeling paradigms were trained and validated using **Stratified 5-Fold Cross-Validation** and evaluated on an unseen held-out test split (25%, $N = 2,998$).

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Primary Use Case |
|-------|----------|-----------|--------|----------|---------|------------------|
| **Logistic Regression (Baseline)** | 82.8% | 46.2% | 27.2% | 34.2% | 0.776 | Interpretability & Odds Ratios |
| **Decision Tree (Tuned)** | 97.4% | 91.5% | 92.8% | 92.1% | 0.963 | Simple Decision Rules |
| **Random Forest (Champion)** 🏆 | **98.4%** | **95.6%** | **94.2%** | **94.9%** | **0.988** | **Production Early-Warning System** |
| **XGBoost Classifier** | 98.2% | 94.8% | 93.8% | 94.3% | 0.985 | High-Scale Batch Inference |

### Asymmetric Cost Matrix: Why Recall Matters Most
In voluntary churn prevention, the cost of errors is heavily asymmetrical:
$$\text{Cost}(\text{False Negative}) \gg \text{Cost}(\text{False Positive})$$
* **False Negative (Cost: \$50,000+):** The model predicts an employee will stay, but they quit unexpectedly. Salifort loses domain knowledge and incurs substantial replacement expenses.
* **False Positive (Cost: Minimal):** The model predicts an employee will quit, but they stay. HR initiates a confidential check-in, workload review, or career conversation.
* **Decision:** **Random Forest** was selected as the **Champion Model** for achieving the highest **Recall (94.2%)**, **F1-Score (94.9%)**, and **ROC-AUC (0.988)**.

---

## 5. Feature Importance Decomposition

Both tree ensemble models identified the exact same structural features as primary churn drivers:

```
Rank  Feature                   Description
 1    satisfaction_level        Primary psychological barometer (low satisfaction drives departure)
 2    number_project            Operational load indicator (overload in 6-7 projects)
 3    tenure                    Career stage indicator (years 3-5 high risk)
 4    average_monthly_hours     Physical overwork indicator (>250 hrs/month)
 5    last_evaluation           High-evaluation employees leave when overworked
```

Departmental tags and minor variables contributed less than 2% of total feature importance, proving that **burnout and workload design—not departmental culture—are the root cause of attrition**.

---

## 6. Strategic Recommendations for HR & Leadership

### 1. Enforce a Strict Overtime Ceiling
* **Action:** Cap working hours at **200 hours/month** (averaging ~45 hours/week).
* **Implementation:** Automated alerts in the time-tracking system when an employee logs $>215$ hours in two consecutive months, triggering a mandatory managerial workload audit.

### 2. Standardize Project Allocations
* **Action:** Establish a company-wide policy limiting individual project assignments to **3–5 concurrent projects**.
* **Impact:** Eliminates the 100% attrition rate observed among employees with 7 projects.

### 3. Year 3–4 Career Development Milestones
* **Action:** Implement structured career reviews, retention bonuses, and lateral mobility options at **tenure year 3 and year 4** to prevent the year-5 attrition cliff.

### 4. Align Evaluation with Merit & Promotion
* **Action:** Recalibrate compensation models so that top-tier performers with high evaluation scores receive equity and salary adjustments rather than merely increased project allocations.

### 5. Deploy the Machine Learning Early-Warning Dashboard
* **Action:** Run the Random Forest scoring model quarterly across the HR Information System (HRIS). Flag employees with churn probability $\ge 0.60$ for proactive "stay interviews" conducted by People Business Partners.

---

## 7. How to Run This Project

### Prerequisites
* Python 3.8+
* Jupyter Notebook or JupyterLab

### Installation
Clone the repository and install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Running the Notebook
Navigate to this directory and launch Jupyter:
```bash
jupyter notebook salifort_employee_churn_prediction.ipynb
```
The notebook automatically locates the dataset in `./data/HR_capstone_dataset.csv` or fallback relative paths.

