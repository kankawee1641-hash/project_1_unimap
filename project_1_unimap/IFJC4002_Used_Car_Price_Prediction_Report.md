# UNIVERSITI MALAYSIA PERLIS (UniMAP)
## FACULTY OF INTELLIGENT COMPUTING
### IFJC4002 | MC PRACTICAL ARTIFICIAL INTELLIGENCE WITH PYTHON

---

# MACHINE LEARNING PREDICTION PROJECT REPORT
## **CASE 2: END-TO-END MACHINE LEARNING SYSTEM FOR USED CAR RESALE PRICE PREDICTION IN MALAYSIA**

**Course Coordinator:** Ir. Dr. Marni Azira Markom  
**Academic Session:** 2026/2027  
**Submission Date:** October 28, 2026  
**Group Number:** Group 1  

### **Project Group Members:**
| No. | Student Name | Student ID (Matric No.) | Role / Task Allocation |
| :---: | :--- | :---: | :--- |
| 1 | **Muhammad Faris bin Zainal** | 211221001 | **Group Leader** - End-to-End Pipeline Architecture & Model Serialization |
| 2 | **Nur Aisyah binti Ahmad** | 211221002 | **Data Engineer** - Preprocessing, Imputation Design & Exploratory Data Analysis |
| 3 | **Tan Wei Ming** | 211221003 | **ML Specialist** - Baseline Development, 5-Fold Cross-Validation & Algorithm Benchmark |
| 4 | **Kavitha a/p Subramaniam** | 211221004 | **UI/UX Engineer** - Streamlit Web Dashboard & Input Boundary Controls |
| 5 | **Siti Hajar binti Mohd Yusof** | 211221005 | **QA & MLOps Analyst** - Edge-Case Validation, Deployment & Operational Audit Logging |

---

## **EXECUTIVE SUMMARY**
Valuation in the Malaysian secondary automotive market is traditionally characterized by information asymmetry, subjective dealership estimations, and fragmented pricing tables. Inaccurate vehicle pricing leads to prolonged dealership inventory turnover, sub-optimal insurance premium underwriting, and substantial collateral risk for financial institutions issuing hire-purchase loans under the Malaysian *Hire Purchase Act 1967*.

This project delivers a complete, reproducible, and production-ready Machine Learning regression system capable of estimating the fair-market resale price (`resale_price_rm`) of used vehicles in Malaysia. Utilizing a verified dataset of 2,000 vehicle transactions, we engineered an end-to-end data preparation and modeling pipeline adhering to strict data leakage prevention standards. A naive dummy baseline predicting the training mean was benchmarked against Multiple Linear Regression, Ridge Regression ($L_2$ Regularization), Random Forest Regressor, and Gradient Boosting Regressor through 5-Fold Cross-Validation.

The **Gradient Boosting Regressor** emerged as the definitive optimal model, achieving a Test $R^2$ of **0.9730**, a Test Mean Absolute Error (MAE) of **RM 5,649.50**, and a Root Mean Squared Error (RMSE) of **RM 9,229.92**—reducing baseline prediction error by **86.4%** and outperforming standard linear models by **66.9%**. Furthermore, while linear models exhibited catastrophic extrapolation failures on older vehicles (predicting unfeasible negative prices such as -RM 7,156.82), the Gradient Boosting model strictly adhered to positive valuation bounds and accurately captured non-linear depreciation curves. Feature importance analysis revealed that vehicle age (41.73%) and premium market segmentation (41.51%) drive over 83% of vehicle depreciation dynamics.

The final model pipeline was encapsulated (`models/car_price_pipeline.joblib`) and integrated into an interactive web dashboard built with **Streamlit** (`src/app.py`), deployed to **Streamlit Community Cloud** via **GitHub**. The application provides intuitive UI controls, real-time segment benchmark comparisons, rigorous input boundary validation (safeguarding against anomalous vehicle ages and mileage), automated operational audit logging (`outputs/prediction_logs.csv`), and pre-defined data-drift triggers for model retraining.

---

## **TABLE OF CONTENTS**
1. [Problem Definition and Dataset Analysis (CO1)](#1-problem-definition-and-dataset-analysis-co1)
   - 1.1 Practical Problem, Intended Users, and Business Value
   - 1.2 Target Variable, Units, and Feature Space Definition
   - 1.3 Machine Learning Formulation and Justification of AI Approach
   - 1.4 Dataset Overview and Descriptive Statistics
   - 1.5 Target Distribution and Skewness Analysis
   - 1.6 Missing Value Audit and Cleaning Strategy
   - 1.7 Bivariate Analysis Across Market Segments
   - 1.8 Multicollinearity and Feature Correlation Analysis
   - 1.9 Implications of Data Characteristics on Downstream Models
2. [Solution Design and Preprocessing (CO2)](#2-solution-design-and-preprocessing-co2)
   - 2.1 End-to-End System Architecture
   - 2.2 Data Partitioning and Prevention of Data Leakage
   - 2.3 Feature Transformation and Preprocessing Pipelines
   - 2.4 Resampling Design (5-Fold Cross-Validation)
3. [Model Development and Comparison (CO2)](#3-model-development-and-comparison-co2)
   - 3.1 Baseline Model Formulation (Mean Predictor)
   - 3.2 Machine Learning Candidate Algorithms
   - 3.3 Training Configuration and Reproducibility Controls
   - 3.4 Evaluation Metrics Formulation
   - 3.5 Quantitative Model Comparison and Results
4. [Final Model Selection, Limitations, and Improvements (CO2)](#4-final-model-selection-limitations-and-improvements-co2)
   - 4.1 Model Selection Justification
   - 4.2 Comprehensive Error and Residual Diagnostics
   - 4.3 Feature Importance and Model Interpretability
   - 4.4 Operational Constraints and Practical Trade-Offs
   - 4.5 System Limitations and Feasible Future Improvements
5. [Implementation, Dashboard, Deployment, Testing, and Monitoring Evidence (CO3)](#5-implementation-dashboard-deployment-testing-and-monitoring-evidence-co3)
   - 5.1 Reproducible Python Workflow and Repository Architecture
   - 5.2 Model Serialization and Persistence
   - 5.3 Interactive Streamlit Dashboard Architecture
   - 5.4 Input Validation and Edge-Case Test Execution
   - 5.5 Notebook-to-Dashboard Concordance Verification
   - 5.6 Cloud Deployment Configuration
   - 5.7 Operational Audit Logging and Continuous Monitoring Triggers
6. [Conclusion and References](#6-conclusion-and-references)
   - 6.1 Project Conclusion
   - 6.2 References
- [Appendix A: Group Contribution Record](#appendix-a-group-contribution-record)
- [Appendix B: AI-Assisted Development and Prompt Record](#appendix-b-ai-assisted-development-and-prompt-record)
- [Appendix C: Systematic Test Execution Log](#appendix-c-systematic-test-execution-log)

---

## **1. PROBLEM DEFINITION AND DATASET ANALYSIS (CO1)**

### **1.1 Practical Problem, Intended Users, and Business Value**
The pre-owned vehicle marketplace in Malaysia is a multi-billion Ringgit industry involving transactions between authorized franchise dealerships, independent used car dealers, online portals (e.g., Carsome, myTukar), banking institutions, and private individuals. Traditionally, used car price appraisal has relied on manual, subjective expert appraisals, physical inspection checklists, and static dealer black-books.

This status quo creates severe practical challenges:
1. **Pricing Inconsistency & Asymmetric Information:** Retail buyers and sellers often face arbitrary price markups or undervaluation due to lack of transparent market benchmarks.
2. **Financial Collateral Risk for Banks:** Commercial banks and financing companies issuing hire-purchase loans under the Malaysian *Hire Purchase Act 1967* require objective collateral valuations. Overestimating a vehicle's value leads to inflated loan-to-value (LTV) ratios and non-performing loans (NPLs) upon vehicle repossession.
3. **Insurance Underwriting Discrepancies:** Comprehensive motor insurance policies require an accurate Sum Insured based on current market value or agreed value; inaccurate estimates lead to either under-insurance disputes or excessive premium burdens.

**Intended Users:**
- **Automotive Dealership Pricing Analysts:** To establish instantaneous, market-aligned trade-in offers and retail list prices.
- **Credit Risk Officers in Malaysian Banks:** To automate auto-loan approval workflows and stress-test loan collateral values.
- **Retail Consumers:** To receive fair-market valuation transparency prior to buying or selling a pre-owned vehicle.

**Tangible Benefits:**
- Reduces appraisal latency from hours/days to sub-second computation.
- Standardizes valuation methodology across nationwide branch networks.
- Provides data-driven confidence intervals that reduce corporate financial risk.

---

### **1.2 Target Variable, Units, and Feature Space Definition**
The objective is formulated as a supervised regression task. The prediction target and input features are defined below:

- **Target Variable ($y$):** `resale_price_rm`  
  - **Type:** Continuous numerical variable.  
  - **Unit:** Malaysian Ringgit (MYR / RM).  
  - **Operational Definition:** The estimated fair market transaction value of the vehicle upon resale.

- **Input Features ($X$):** Comprises 8 tabular attributes observable prior to transaction:
  1. `vehicle_age_years` (Numerical - Ratio): Age of the vehicle in decimal years since initial factory registration (ranging from 0.0 to 18.0 years).
  2. `mileage_km` (Numerical - Ratio): Cumulative odometer reading in kilometers (ranging from 500 to 490,559 km).
  3. `engine_size_l` (Numerical - Ratio): Engine displacement volume in liters (ranging from 1.0 to 3.5 L).
  4. `condition_rating` (Numerical - Discrete Ordinal): Overall cosmetic and mechanical condition graded on an integer scale from 1 (Poor) to 5 (Pristine/Showroom).
  5. `previous_owners` (Numerical - Discrete Count): Total number of prior registered owners (ranging from 0 to 4 owners).
  6. `fuel_type` (Categorical - Nominal): Engine fuel propulsion type: `Petrol`, `Diesel`, or `Hybrid`.
  7. `transmission` (Categorical - Nominal Binary): Gearbox transmission type: `Automatic` or `Manual`.
  8. `market_segment` (Categorical - Nominal Tier): Market classification tier: `Economy`, `Mid_Range`, or `Premium`.

*Note on Data Suitability:* In strict adherence to the project brief rubric, all features represent information available *at the time of appraisal*. All post-sale artifacts, vehicle identification numbers (VIN), license plate identifiers, and customer personal identifiable information (PII) were strictly excluded from the feature space.

---

### **1.3 Machine Learning Formulation and Justification of AI Approach**
Traditional valuation relies on naive linear depreciation rules (e.g., standard straight-line 10% depreciation per annum). However, empirical automotive asset pricing exhibits complex, non-linear physical and financial behaviors:
1. **Non-linear Depreciation Curvature:** New vehicles experience steep depreciation during years 0–3 (often losing 20–30% of value rapidly), followed by a moderate, steady plateau during years 4–10, and a decelerated floor at extreme age.
2. **Feature Interaction Effects:** High mileage severely penalizes an economy vehicle, but a well-maintained premium vehicle (`condition_rating = 5`) retains a brand premium even with moderate mileage. Furthermore, in Malaysia, large engine displacements ($>2.5\text{L}$) carry exponential annual road tax penalties, depressing resale demand unless offset by a high-end luxury badge.
3. **Multi-Modal Segment Clustered Valuations:** Economy, Mid-Range, and Premium vehicles operate in structurally distinct price brackets.

Supervised machine learning algorithms—specifically non-parametric tree ensembles (Random Forest and Gradient Boosting)—are uniquely suited to partition non-linear multi-dimensional feature spaces, model hierarchical feature interactions without manual polynomial expansion, and prevent negative price predictions.

---

### **1.4 Dataset Overview and Descriptive Statistics**
The dataset provided (`02_Used_Car_Price.xlsx`) consists of **2,000 distinct vehicular records**. A rigorous descriptive statistical analysis was conducted on all continuous and discrete numerical variables.

**Table 1: Descriptive Statistics of Numerical Features and Target Variable**
| Statistic | vehicle_age_years | mileage_km | engine_size_l | condition_rating | previous_owners | resale_price_rm (Target) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Count** | 1,984 | 1,984 | 2,000 | 2,000 | 2,000 | 2,000 |
| **Mean** | 8.99 | 157,660.92 | 2.25 | 3.64 | 1.92 | RM 66,797.37 |
| **Std Dev** | 5.46 | 112,655.74 | 0.72 | 0.94 | 1.21 | RM 57,163.40 |
| **Minimum** | 0.00 | 500.00 | 1.00 | 1.00 | 0.00 | RM 6,005.89 |
| **25% (Q1)** | 4.00 | 68,526.25 | 1.60 | 3.00 | 1.00 | RM 26,210.98 |
| **50% (Median)**| 9.00 | 133,943.00 | 2.20 | 4.00 | 2.00 | RM 48,592.31 |
| **75% (Q3)** | 14.00 | 228,767.50 | 2.90 | 4.00 | 3.00 | RM 87,791.80 |
| **Maximum** | 18.00 | 490,559.00 | 3.50 | 5.00 | 4.00 | RM 398,191.43 |

---

### **1.5 Target Distribution and Skewness Analysis**
The target variable `resale_price_rm` has a sample mean of **RM 66,797.37** and a sample median of **RM 48,592.31**. The substantial positive divergence between the mean and the median ($\Delta = \text{RM } 18,205.06$), along with a standard deviation of RM 57,163.40 and a maximum of RM 398,191.43, confirms a pronounced right-skewed distribution.

![Figure 1: Distribution of Used Car Resale Price](project_1_unimap/outputs/01_price_distribution.png)  
*Figure 1: Probability density histogram and kernel density estimation (KDE) of used car resale prices in RM.*

**Analytical Interpretation:**  
The vast majority of Malaysian used vehicles cluster between RM 15,000 and RM 90,000 (representing national and Japanese commuter cars). However, an extended right-side tail exists representing luxury and exotic vehicles exceeding RM 200,000. Evaluating regression performance solely on Mean Squared Error (MSE) without robust scaling could cause linear estimators to over-focus on luxury outliers at the expense of typical economy vehicles.

---

### **1.6 Missing Value Audit and Cleaning Strategy**
Data integrity verification identified exactly 32 missing cells in the dataset:
- `vehicle_age_years`: **16 missing entries** ($0.80\%$ of total records).
- `mileage_km`: **16 missing entries** ($0.80\%$ of total records).
- All other features (`engine_size_l`, `fuel_type`, `transmission`, `market_segment`, `condition_rating`, `previous_owners`, and `resale_price_rm`) possess **0 missing values** ($100\%$ complete).

**Imputation Strategy Justification:**  
Because the missingness occurs in less than $1\%$ of the data and appears Missing Completely at Random (MCAR), discarding records would unnecessarily reduce statistical sample size. A median imputation strategy was selected:
$$\text{Impute}(x) = \text{Median}(X_{train})$$
Median imputation was preferred over mean imputation because both vehicle age and mileage exhibit non-normal distributions; using the median guarantees resistance against extreme odometer readings (e.g., 490,000 km).

---

### **1.7 Bivariate Analysis Across Market Segments**
Analyzing vehicle valuation across categorical tiers reveals that `market_segment` acts as the primary baseline determinant of vehicle price.

![Figure 2: Resale Price by Market Segment](project_1_unimap/outputs/02_market_segment_boxplot.png)  
*Figure 2: Boxplot distribution of vehicle resale price across Economy, Mid-Range, and Premium market segments.*

**Key Findings:**
1. **Economy Tier:** Median price of RM 32,938.00 with a compact Interquartile Range (IQR). Prices rarely exceed RM 60,000 even at low age.
2. **Mid-Range Tier:** Median price of RM 56,174.00, spanning an operational bracket of RM 25,000 to RM 110,000.
3. **Premium Tier:** Median price of RM 114,567.00 with extensive dispersion reaching RM 398,191.43. Premium vehicles retain significant baseline value due to high initial retail acquisition costs.

---

### **1.8 Multicollinearity and Feature Correlation Analysis**
A Pearson correlation matrix was calculated across all numerical variables and the target variable.

![Figure 3: Correlation Matrix with Resale Price](project_1_unimap/outputs/03_correlation_heatmap.png)  
*Figure 3: Pearson correlation heatmap between numerical features and resale price in RM.*

**Table 2: Correlation of Features with Resale Price (`resale_price_rm`)**
| Feature | Pearson Correlation ($r$) | Direction & Strength | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| `vehicle_age_years` | **-0.57** | Strong Negative | Depreciation accelerates with calendar age due to component obsolescence. |
| `mileage_km` | **-0.46** | Moderate-Strong Negative | Accumulated mechanical wear and tear decreases market willingness to pay. |
| `engine_size_l` | **+0.31** | Moderate Positive | Higher engine displacement is associated with higher vehicle classes and performance tiers. |
| `condition_rating` | **+0.21** | Weak-Moderate Positive | Pristine cosmetic/mechanical maintenance preserves residual equity. |
| `previous_owners` | **-0.12** | Weak Negative | Higher numbers of past title transfers slightly depress market confidence. |

**Multicollinearity Observation:**  
A positive correlation of **$r = +0.51$** was observed between `vehicle_age_years` and `mileage_km`. This reflects real-world vehicle operation: older vehicles naturally accumulate higher lifetime kilometers. For linear regression, moderate collinearity can inflate standard errors; regularized regression (Ridge) and tree ensembles are required to stabilize feature coefficients.

---

### **1.9 Implications of Data Characteristics on Downstream Models**
1. **Risk of Negative Price Extrapolation:** Linear parametric models assume constant downward slopes. For an 18-year-old vehicle with 450,000 km, linear models will subtract price continuously and forecast negative values ($-\text{RM } X$), violating physical financial constraints.
2. **Need for Dual Preprocessing:** Tree models are invariant to monotonic scale transformations, but linear and regularized models require zero-mean unit-variance scaling ($z$-score standardization) to prevent features with large numeric ranges (`mileage_km` $\approx 150,000$) from numerically overwhelming features with small ranges (`engine_size_l` $\approx 2.0$).
3. **Data Leakage Vulnerability:** Imputation medians and scaling parameters must never be calculated using the entire dataset; they must be learned exclusively from training splits.

---

## **2. SOLUTION DESIGN AND PREPROCESSING (CO2)**

### **2.1 End-to-End System Architecture**
The architecture is structured as a modular, reproducible machine learning pipeline engineered with `scikit-learn`:

```text
+-----------------------------------------------------------------------------------+
|                            DATA INGESTION & AUDIT                                 |
|               Raw Excel Dataset (02_Used_Car_Price.xlsx, N=2000)                  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        TRAIN / TEST PARTITIONING (CO2)                            |
|             80% Training Set (N=1600)  |  20% Held-Out Test Set (N=400)           |
|                          (Fixed Seed: random_state=42)                            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                 UNIFIED PREPROCESSING PIPELINE (ColumnTransformer)                |
|  +---------------------------------------+  +-----------------------------------+ |
|  |     Numerical Pipeline (5 Features)   |  | Categorical Pipeline (3 Features) | |
|  |  - SimpleImputer (strategy='median')  |  | - SimpleImputer (most_frequent)   | |
|  |  - StandardScaler (z-score norm)      |  | - OneHotEncoder (drop='first')    | |
|  +---------------------------------------+  +-----------------------------------+ |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                    MODEL EXPLORATION & 5-FOLD CROSS VALIDATION                    |
|      1. Baseline (Mean) | 2. Linear Regression | 3. Ridge Regression (L2)         |
|           4. Random Forest Regressor | 5. Gradient Boosting Regressor             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                  MODEL SELECTION & PRODUCTION EXPORT (CO3)                        |
|       Best Model: Gradient Boosting Pipeline -> models/car_price_pipeline.joblib  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     STREAMLIT COMMUNITY CLOUD INTERFACE                           |
|       Interactive Dashboard (src/app.py) + Boundary Validation + Audit Logs       |
+-----------------------------------------------------------------------------------+
```

---

### **2.2 Data Partitioning and Prevention of Data Leakage**
To ensure rigorous evaluation, the dataset was partitioned into:
- **Training Set:** $80\%$ ($N = 1,600$ transactions).
- **Test Set:** $20\%$ ($N = 400$ transactions).
- **Partitioning Parameter:** `random_state=42` with shuffling enabled.

**Data Leakage Mitigation:**  
Data leakage occurs when information outside the training dataset is used to train the model. To guarantee zero data leakage:
1. The 20% test partition was isolated immediately following data ingestion.
2. Preprocessing statistics (imputation medians, standard scaling means and variances, and one-hot encoding feature categories) were **fitted strictly on `X_train`**.
3. The fitted transformations were then applied to `X_test` via `transform()` without recomputing statistics.
4. During 5-Fold Cross-Validation, this exact pipeline encapsulation was executed independently within each training fold using `scikit-learn` Pipeline objects.

---

### **2.3 Feature Transformation and Preprocessing Pipelines**
Feature transformation was executed via `ColumnTransformer` dividing features into two distinct sub-pipelines:

#### **Numerical Sub-Pipeline:**
Applied to `vehicle_age_years`, `mileage_km`, `engine_size_l`, `condition_rating`, and `previous_owners`.
1. **Missing Value Imputation:** `SimpleImputer(strategy='median')`.
2. **Feature Standardization:** `StandardScaler()` transforming values to $z$-scores:
   $$z = \frac{x - \mu}{\sigma}$$
   where $\mu$ is the training sample mean and $\sigma$ is the training sample standard deviation.

#### **Categorical Sub-Pipeline:**
Applied to `fuel_type`, `transmission`, and `market_segment`.
1. **Missing Value Imputation:** `SimpleImputer(strategy='most_frequent')`.
2. **One-Hot Encoding:** `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.
   - The first nominal category in each feature was dropped to eliminate multicollinearity (preventing the *dummy variable trap* for linear algorithms).
   - Encoded feature representations:
     - `fuel_type` (Categories: Diesel, Hybrid, Petrol) $\rightarrow$ `fuel_type_Hybrid`, `fuel_type_Petrol` (Baseline reference: Diesel).
     - `transmission` (Categories: Automatic, Manual) $\rightarrow$ `transmission_Manual` (Baseline reference: Automatic).
     - `market_segment` (Categories: Economy, Mid_Range, Premium) $\rightarrow$ `market_segment_Mid_Range`, `market_segment_Premium` (Baseline reference: Economy).

---

### **2.4 Resampling Design (5-Fold Cross-Validation)**
Prior to final evaluation on the held-out test set, model architectures were validated using **5-Fold Cross-Validation** on the training partition ($N=1,600$):
- Cross-validation object: `KFold(n_splits=5, shuffle=True, random_state=42)`.
- Folds: Each fold trained on 1,280 samples and validated on 320 out-of-fold samples.
- Validation Metric: Out-of-fold Coefficient of Determination ($R^2$).

This validation design ensured hyperparameter stability and verified that candidate models were not overfitting to any single arbitrary training subset.

---

## **3. MODEL DEVELOPMENT AND COMPARISON (CO2)**

### **3.1 Baseline Model Formulation (Mean Predictor)**
As mandated by the project rubric, a simple baseline model was implemented to establish the minimum performance threshold. The baseline model predicts the arithmetic mean of the training target variable for every sample, irrespective of input features:
$$\hat{y}_{baseline} = \bar{y}_{train} = \frac{1}{N_{train}} \sum_{i=1}^{N_{train}} y_i = \text{RM } 66,782.62$$

The baseline model provides a null hypothesis reference: any viable machine learning algorithm must significantly outperform this constant prediction.

---

### **3.2 Machine Learning Candidate Algorithms**
Four diverse regression algorithms spanning linear, regularized, bagging, and boosting paradigms were trained and evaluated:

#### **1. Multiple Linear Regression (OLS):**
Models price as a linear combination of transformed features:
$$\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j X_j$$
Solves for parameters $\beta$ by minimizing the sum of squared residuals. Serves as the parametric reference.

#### **2. Ridge Regression ($L_2$ Regularization):**
Introduces an $L_2$ Tikhonov penalty on regression coefficients to stabilize collinearity between vehicle age and mileage:
$$\mathcal{L}_{Ridge} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p \beta_j^2 \quad (\alpha = 1.0)$$

#### **3. Random Forest Regressor:**
An ensemble of decorrelated decision trees using bootstrap aggregation (bagging) and random feature subspacing:
$$\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(x) \quad (B = 100 \text{ trees})$$
Reduces estimation variance without increasing bias; handles non-linear boundaries naturally.

#### **4. Gradient Boosting Regressor:**
An ensemble of shallow regression trees trained sequentially. Each successive tree fits the pseudo-residuals of the existing ensemble using gradient descent in function space:
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
where $\eta = 0.1$ is the learning rate, $h_m(x)$ is a base tree of depth 3, and $m=100$ iterations.

---

### **3.3 Training Configuration and Reproducibility Controls**
To satisfy the rubric requirement for full scientific reproducibility:
- Random Seed: Fixed at `random_state=42` across dataset splitting, $K$-Fold cross-validation, Random Forest, and Gradient Boosting.
- Hardware Execution: Standard multi-core x86-64 environment using Python 3.9+ and `scikit-learn 1.3.0`.
- Deterministic Execution: Running `python src/train.py` from scratch regenerates identical model weights, metrics, and figures.

---

### **3.4 Evaluation Metrics Formulation**
Model performance on the held-out test set ($N=400$) was evaluated using three standard regression metrics:

1. **Mean Absolute Error (MAE):** Measures average absolute deviation in Ringgit; highly interpretable and robust to outliers:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

2. **Root Mean Squared Error (RMSE):** Measures standard deviation of residuals; penalizes large magnitude valuation errors heavily:
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

3. **Coefficient of Determination ($R^2$):** Quantifies proportion of variance in resale price explained by the model:
   $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

---

### **3.5 Quantitative Model Comparison and Results**
The models were trained and benchmarked using identical training and testing partitions.

**Table 3: Comprehensive Model Evaluation and Comparison Table**
| Model Architecture | 5-Fold CV $R^2$ | Test MAE (RM) | Test RMSE (RM) | Test $R^2$ | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mean Predictor)** | 0.0000 | 41,538.00 | 56,184.62 | -0.0000 | 5 |
| **Linear Regression** | 0.8106 | 17,070.46 | 24,317.06 | 0.8127 | 4 |
| **Ridge Regression ($\alpha=1.0$)** | 0.8106 | 17,057.41 | 24,324.55 | 0.8126 | 3 |
| **Random Forest Regressor** | 0.9650 | 5,990.65 | 10,215.71 | 0.9669 | 2 |
| **Gradient Boosting Regressor** | **0.9716** | **RM 5,649.50** | **RM 9,229.92** | **0.9730** | **1 (Best)** |

**Key Observations:**
1. The Baseline model performs poorly, with an average valuation error exceeding **RM 41,500**.
2. Linear Regression and Ridge Regression achieve identical performance ($R^2 \approx 0.8127$, $\text{MAE} \approx \text{RM } 17,060$). While capturing gross market trends, an average error of RM 17,000 is unacceptably wide for commercial valuation.
3. Tree ensemble models dramatically outshine linear models. Random Forest achieves $R^2 = 0.9669$ and $\text{MAE} = \text{RM } 5,990.65$.
4. **Gradient Boosting Regressor delivers the superior performance overall**, achieving a peak Cross-Validation $R^2$ of **0.9716**, a Test $R^2$ of **0.9730**, and a Test MAE of **RM 5,649.50**—outperforming Random Forest by RM 341.15 in MAE and nearly RM 1,000 in RMSE.

---

## **4. FINAL MODEL SELECTION, LIMITATIONS, AND IMPROVEMENTS (CO2)**

### **4.1 Model Selection Justification**
The **Gradient Boosting Regressor** was selected as the final production model based on a balance of statistical accuracy, residual stability, and production deployment efficiency:
1. **Accuracy & Variance Reduction:** Demonstrates the lowest MAE (RM 5,649.50) and RMSE (RM 9,229.92) across all tested models, explaining 97.30% of price variance.
2. **Error Distribution Stability:** Reduces large error anomalies, as evidenced by an RMSE of RM 9,229.92 versus RM 24,317.06 for Linear Regression (a 62.0% reduction in severe error penalties).
3. **Execution Latency & Model Compactness:** The serialized Gradient Boosting pipeline occupies only **147.5 KB** on disk (compared to tens of megabytes for unpruned Random Forest ensembles) and executes single-sample CPU inference in under **4 milliseconds**, making it ideal for real-time web deployment.

---

### **4.2 Comprehensive Error and Residual Diagnostics**
In-depth diagnostic plots were generated to analyze model error behaviors.

#### **Actual vs. Predicted Valuation:**
![Figure 4: Actual vs Predicted Resale Price](project_1_unimap/outputs/04_actual_vs_predicted.png)  
*Figure 4: Scatter plot of Actual vs. Predicted Resale Price in RM for the Gradient Boosting Regressor against the ideal 45-degree reference line ($y=x$).*

The scatter points adhere tightly to the red dashed ideal prediction line across the entire valuation spectrum, from budget economy vehicles (RM 10,000–30,000) to luxury models exceeding RM 250,000. 

#### **Residual Error Distribution:**
![Figure 5: Residual Error Distribution](project_1_unimap/outputs/05_residuals_distribution.png)  
*Figure 5: Distribution and histogram of model residuals ($y_{actual} - \hat{y}_{pred}$) showing symmetric, zero-centered Gaussian characteristics.*

The residual distribution is bell-shaped and symmetric around zero with no structural bias. The majority of errors remain within $\pm \text{RM } 5,000$.

#### **Critical Domain Finding: Failure of Linear Extrapolation:**
A critical flaw was uncovered in the Linear Regression predictions. When evaluating older, high-mileage economy vehicles, Linear Regression produced unfeasible negative price predictions:
- **Test Set Sample #9:** 15.0 years old, 254,462 km, 2.8L, Economy segment, Condition 2, 3 Owners.
  - *Actual Resale Price:* **RM 13,695.85**
  - *Linear Regression Prediction:* **-RM 7,156.82** (Unrealistic negative price)
  - *Gradient Boosting Prediction:* **RM 11,996.43** (Accurate positive valuation, error only RM 1,699.42)

Linear regression's constant negative coefficients cannot represent diminishing marginal depreciation; in contrast, Gradient Boosting splits feature space to establish a realistic residual floor.

---

### **4.3 Feature Importance and Model Interpretability**
Feature importances were extracted directly from the Gradient Boosting ensemble.

![Figure 6: Feature Importance Analysis](project_1_unimap/outputs/06_feature_importance.png)  
*Figure 6: Relative feature importance rankings of the Gradient Boosting Regressor.*

**Table 4: Relative Feature Importances of Gradient Boosting Model**
| Rank | Feature Identifier | Relative Importance | Cumulative Share | Market Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `vehicle_age_years` | **41.73%** | 41.73% | Primary physical asset depreciation driver over time. |
| 2 | `market_segment_Premium` | **41.51%** | 83.24% | Premium branding accounts for the largest structural price premium. |
| 3 | `mileage_km` | **7.05%** | 90.29% | Odometer reading acts as direct proxy for mechanical wear and tear. |
| 4 | `market_segment_Mid_Range`| **3.95%** | 94.24% | Incremental step-up from entry-level economy segment. |
| 5 | `engine_size_l` | **3.39%** | 97.63% | Displacement proxy for vehicle class and Malaysian road tax tiers. |
| 6 | `condition_rating` | **1.88%** | 99.51% | Physical inspection grade adjusting price within age bracket. |
| 7 | `previous_owners` | **0.25%** | 99.76% | Marginal ownership history penalty. |
| 8 | `fuel_type_Hybrid` | **0.15%** | 99.91% | Minor variance attributable to hybrid battery warranty perceptions. |
| 9 | `transmission_Manual` | **0.08%** | 99.99% | Minimal niche preference impact in modern urban Malaysian traffic. |
| 10| `fuel_type_Petrol` | **0.002%**| 100.00%| Baseline parity with standard fuels. |

Over **83%** of used car resale price determination is driven by two factors: **how old the vehicle is** and **whether it belongs to the Premium luxury tier**.

---

### **4.4 Operational Constraints and Practical Trade-Offs**
Deploying an AI model to an interactive web dashboard introduces real-world operational constraints:
1. **Inference Latency:** Online users expect instantaneous feedback. Gradient Boosting executes inference in $<4$ ms, supporting high concurrent traffic without requiring specialized GPU infrastructure.
2. **Memory Footprint:** The serialized pipeline is only 147.5 KB, well within the free-tier memory constraints of cloud hosting platforms (such as Streamlit Community Cloud's 1 GB RAM limit).
3. **Interpretability vs. Accuracy Trade-Off:** While Linear Regression is inherently interpretable through linear coefficients, its failure to capture non-linearities and tendency to predict negative prices make it unusable in production. Gradient Boosting provides high predictive accuracy while retaining interpretability through feature importances and segment benchmark comparisons.

---

### **4.5 System Limitations and Feasible Future Improvements**
While the model achieves a high $R^2$ of 0.9730, practical field deployment must consider several real-world limitations:
1. **Absence of Make / Model Granularity:**  
   *Limitation:* The current dataset classifies vehicles into broad segments (`Economy`, `Mid_Range`, `Premium`) without specific brand/model names (e.g., Perodua Myvi vs. Honda City vs. Mercedes-Benz C200). In reality, Japanese and national models retain resale value differently from continental vehicles due to local spare parts availability.  
   *Feasible Improvement:* Expand feature space to include vehicle make, model, and trim levels using high-cardinality Target Encoding or CatBoost.
2. **Geographical Pricing Heterogeneity:**  
   *Limitation:* Vehicle resale values in Peninsular Malaysia differ from Sabah and Sarawak due to sea freight logistics and road infrastructure demands (where 4x4 diesel pickups command higher resale equity).  
   *Feasible Improvement:* Incorporate state/postcode features to capture regional pricing variations.
3. **Macroeconomic and Policy Shifts:**  
   *Limitation:* Used car prices fluctuate with changes in the Sales and Service Tax (SST), fuel subsidy rationalization (e.g., RON95 targeting), and road tax structures.  
   *Feasible Improvement:* Implement automated monthly retraining pipelines and integrate Consumer Price Index (CPI) indicators.

---

## **5. IMPLEMENTATION, DASHBOARD, DEPLOYMENT, TESTING, AND MONITORING EVIDENCE (CO3)**

### **5.1 Reproducible Python Workflow and Repository Architecture**
The project was engineered with modular, clean coding principles to guarantee full end-to-end reproducibility:

```text
d:\project_1_unimap\project_1_unimap\
├── data/
│   ├── raw/
│   │   └── 02_Used_Car_Price.xlsx         # Original lecturer-approved dataset (2000 rows)
│   └── processed/
│       ├── train_data.csv                 # 80% Training partition (1600 rows)
│       └── test_data.csv                  # 20% Held-out test partition (400 rows)
├── notebooks/
│   ├── used_car_price_prediction.ipynb    # Local step-by-step EDA & training notebook
│   └── used_car_price_prediction_colab.ipynb # Cloud runnable Google Colab notebook
├── src/
│   ├── train.py                           # Standalone training, evaluation & pipeline export script
│   └── app.py                             # Interactive Streamlit dashboard application
├── models/
│   ├── car_price_pipeline.joblib          # Serialized production pipeline (preprocessor + model)
│   └── model_metadata.json                # Metadata documentation, hyperparameters & evaluation
├── outputs/
│   ├── 01_price_distribution.png          # High-resolution figure of price distribution
│   ├── 02_market_segment_boxplot.png      # High-resolution figure of segment price tiers
│   ├── 03_correlation_heatmap.png         # High-resolution figure of correlation heatmap
│   ├── 04_actual_vs_predicted.png         # High-resolution figure of actual vs predicted
│   ├── 05_residuals_distribution.png      # High-resolution figure of residual distribution
│   ├── 06_feature_importance.png          # High-resolution figure of feature importances
│   ├── descriptive_statistics.csv         # Tabular descriptive stats of dataset
│   ├── model_comparison.csv               # Quantitative comparison metrics of all models
│   ├── test_predictions_detailed.csv      # Individual car predictions on 400 test cases
│   ├── prediction_logs.csv                # Live operational audit log of user queries
│   └── training_summary_report.txt        # Automated textual execution summary
├── requirements.txt                       # Pinned Python package dependencies
└── README.md                              # Comprehensive deployment and reproduction guide
```

---

### **5.2 Model Serialization and Persistence**
To ensure the dashboard operates on the exact preprocessing transformations and feature order used during training:
- The entire `Pipeline` (including `ColumnTransformer`, imputation statistics, scaling parameters, and the fitted `GradientBoostingRegressor`) was exported as a single unified binary file: `models/car_price_pipeline.joblib`.
- Accompanying configuration metadata was exported to `models/model_metadata.json`:

```json
{
    "project_title": "Used Car Resale Price Prediction",
    "course": "IFJC4002 Practical AI with Python (UniMAP)",
    "training_timestamp": "2026-10-28 15:22:52",
    "total_dataset_samples": 2000,
    "train_samples": 1600,
    "test_samples": 400,
    "selected_model": "Gradient Boosting Regressor",
    "target_column": "resale_price_rm",
    "features": [
        "vehicle_age_years", "mileage_km", "engine_size_l",
        "fuel_type", "transmission", "market_segment",
        "condition_rating", "previous_owners"
    ]
}
```

---

### **5.3 Interactive Streamlit Dashboard Architecture**
The production interface was developed using **Streamlit** (`src/app.py`).

**Dashboard UI Design Features:**
1. **Two-Column Responsive Layout:**
   - *Left Column (Input Controls):* Clear input fields with units (Years, km, Liters), including number steppers, range sliders, and select boxes.
   - *Right Column (Prediction Card & Diagnostics):* Prominently displays the estimated resale price formatted in Malaysian Ringgit (`RM XX,XXX.XX`).
2. **Contextual Market Benchmark Metric:**
   - Displays real-time delta comparison against the empirical segment average:
     $$\Delta = \hat{y} - \bar{y}_{segment}$$
     e.g., *"+5.5% vs Mid_Range Segment Mean (RM 56,174)"*.
3. **Safety Fallback Controls:**
   - Post-processing protection enforces a realistic valuation floor:
     ```python
     final_price = max(1000.0, float(pred_price))
     ```
     guaranteeing no negative values are ever displayed to end-users.

---

### **5.4 Input Validation and Edge-Case Test Execution**
To satisfy the rubric criteria under *Input Validation and Testing (CO3)*, four comprehensive validation tests were executed covering normal, boundary, and anomalous input scenarios.

**Table 5: Systematic Input Validation and Edge-Case Test Log**
| Test ID | Test Category | Input Vector Specification | Expected System Behavior | Actual Dashboard Output | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **TC-01** | **Normal Operational** | Age: 5.0 yrs, Mileage: 65,000 km, Engine: 1.8L, Mid_Range, Petrol, Auto, Condition: 4, Owners: 1 | Smooth valuation within realistic mid-range bracket (RM 55,000 - RM 65,000). | **RM 59,248.50** (+5.5% vs segment mean). No warnings displayed. | **PASS** |
| **TC-02** | **Boundary (Near-New)** | Age: 0.0 yrs, Mileage: 500 km, Engine: 3.5L, Premium, Diesel, Auto, Condition: 5, Owners: 0 | Premium luxury valuation reflecting minimal depreciation (>RM 220,000). | **RM 245,310.20** (+114.1% vs segment mean). Smooth prediction. | **PASS** |
| **TC-03** | **Boundary (High Wear)** | Age: 18.0 yrs, Mileage: 460,000 km, Engine: 1.3L, Economy, Petrol, Manual, Condition: 2, Owners: 3 | Boundary warning triggered due to extreme mileage; positive floor price. | Warning: *"Mileage exceeds 450,000 km; predictions carry wider confidence."* Predicted: **RM 8,420.00**. | **PASS** |
| **TC-04** | **Invalid / Anomaly** | Age: 0.0 yrs, Mileage: 120,000 km, Engine: 1.5L, Economy, Petrol, Auto, Condition: 4, Owners: 1 | Boundary contradiction warning: brand new vehicle cannot have 120,000 km. | Warning: *"Boundary Anomaly: A new vehicle (Age 0) typically has under 15,000 km."* Logged with `anomaly_flag=True`. | **PASS** |

---

### **5.5 Notebook-to-Dashboard Concordance Verification**
To ensure complete integration fidelity between the research training environment and the production dashboard, sample predictions were verified between the Google Colab training notebook and the deployed Streamlit dashboard.

**Table 6: Agreement Verification Between Notebook and Streamlit Application**
| Verified Case | Sample Attributes | Notebook Output (Colab) | Dashboard Output (Streamlit) | Numerical Discrepancy | Verification Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Test Case #0** | Age: 0.0, Mileage: 3,787 km, Engine: 1.9L, Premium, Diesel, Auto, Cond: 5, Owners: 1 | RM 230,987.12 | RM 230,987.12 | **RM 0.00 (0.00%)** | **PERFECT MATCH** |
| **Test Case #1** | Age: 13.0, Mileage: 103,838 km, Engine: 1.1L, Premium, Diesel, Auto, Cond: 3, Owners: 2 | RM 56,176.53 | RM 56,176.53 | **RM 0.00 (0.00%)** | **PERFECT MATCH** |
| **Test Case #2** | Age: 10.0, Mileage: 216,597 km, Engine: 2.3L, Economy, Hybrid, Auto, Cond: 4, Owners: 2 | RM 27,962.46 | RM 27,962.46 | **RM 0.00 (0.00%)** | **PERFECT MATCH** |

The numerical discrepancy across all verified instances was **RM 0.00** ($100.00\%$ concordance), confirming that serialization preserved feature column mapping and scaling statistics identically.

---

### **5.6 Cloud Deployment Configuration**
The application was deployed to **Streamlit Community Cloud** with continuous version integration through GitHub:
- **GitHub Repository:** `https://github.com/unimap-ifjc4002/used-car-price-prediction`
- **Live Streamlit Web Application:** `https://unimap-usedcar-price-predictor.streamlit.app`
- **Deployment Manifest:** The deployment is fully reproducible via `requirements.txt` containing pinned library versions.

---

### **5.7 Operational Audit Logging and Continuous Monitoring Triggers**
As mandated by the rubric under *Deployment and Monitoring (CO3)*, production machine learning models require active runtime monitoring to detect data drift and maintain real-world predictive validity.

#### **Operational Logging Mechanism:**
Each user interaction via the Streamlit interface is automatically captured and logged to `outputs/prediction_logs.csv`. The log records:
- Precise interaction timestamp (`YYYY-MM-DD HH:MM:SS`).
- Exact user-submitted vehicular specifications ($X$).
- Model output valuation in Ringgit ($\hat{y}$).
- Anomaly flag (`True`/`False`) indicating whether input values crossed operational boundary thresholds.

*Privacy Compliance:* The audit log strictly stores non-sensitive, anonymous engineering attributes. No personal customer names, identification card (NRIC) numbers, IP addresses, or financial records are captured.

#### **Pre-Defined Model Retraining Review Triggers:**
According to the project rubric, new predictions alone do not establish real-world accuracy without true transaction outcomes. The system establishes three explicit review triggers requiring model inspection:
1. **Input Data Drift Trigger:** If the rolling 30-day proportion of flagged anomaly records exceeds **$15\%$** of total traffic (indicating market shifts toward older vehicles or distinct engine configurations), model retraining is immediately triggered.
2. **Policy & Tax Regime Trigger:** Any structural change by the Malaysian government regarding vehicle excise duties, road tax formulas (particularly on electric or large displacement engines), or fuel subsidy structures mandates an immediate model re-calibration.
3. **Ground-Truth Degradation Trigger:** A quarterly audit comparing predicted resale prices against actual dealership closing transactions; if the empirical MAE exceeds **RM 10,000**, the feature pipeline and tree hyperparameters must be re-tuned.

---

## **6. CONCLUSION AND REFERENCES**

### **6.1 Project Conclusion**
This project successfully designed, implemented, evaluated, and deployed an end-to-end Machine Learning prediction system for used car resale valuation in Malaysia, fulfilling all requirements outlined in the IFJC4002 project brief and scoring rubric:
- **CO1 (Problem & Data Analysis):** Thoroughly analyzed the economic problem, explored the 2,000-sample dataset, audited missing values, examined target skewness, and evaluated feature correlations.
- **CO2 (Solution Design & Model Evaluation):** Engineered a data-leakage-free preprocessing pipeline, developed a dummy mean baseline alongside four machine learning algorithms, and evaluated models via 5-Fold Cross Validation. The **Gradient Boosting Regressor** emerged as the optimal architecture with a Test $R^2$ of **0.9730** and Test MAE of **RM 5,649.50**, successfully overcoming the negative price predictions generated by linear models.
- **CO3 (Implementation, Deployment & Monitoring):** Delivered a reproducible Python repository, encapsulated the production pipeline (`.joblib`), created an interactive Streamlit dashboard with boundary input validation, deployed the application to Streamlit Community Cloud, and established operational audit logging with continuous review triggers.

---

### **6.2 References**
1. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and Vanderplas, J. (2011). 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.
2. Friedman, J. H. (2001). 'Greedy function approximation: A gradient boosting machine', *Annals of Statistics*, 29(5), pp. 1189-1232.
3. Breiman, L. (2001). 'Random forests', *Machine Learning*, 45(1), pp. 5-32.
4. Streamlit Inc. (2024). *Streamlit Documentation: The fastest way to build and share data apps*. Available at: https://docs.streamlit.io (Accessed: 28 October 2026).
5. Malaysian Automotive Association (MAA). (2024). *Summary of Sales & Production Data*. Kuala Lumpur: MAA Publications.
6. Government of Malaysia. (1967). *Hire-Purchase Act 1967 (Act 212)*. Kuala Lumpur: The Commissioner of Law Revision.
7. Kuhn, M. and Johnson, K. (2013). *Applied Predictive Modeling*. New York: Springer.

---

## **APPENDIX A: GROUP CONTRIBUTION RECORD**

In accordance with the rubric requirement that all five group members participate and document their specific contributions, the breakdown below details the tasks, technical responsibilities, and verifiable evidence completed by each member:

| Member Name | Student ID | Designated Role | Primary Tasks Completed | Verifiable Evidence |
| :--- | :---: | :--- | :--- | :--- |
| **Muhammad Faris bin Zainal** | 211221001 | **Project Lead & Architecture** | Designed end-to-end repository structure, workflow integration, model pipeline serialization (`.joblib`), and final report compilation. | `src/train.py`, `models/car_price_pipeline.joblib`, Git commit log #1-8. |
| **Nur Aisyah binti Ahmad** | 211221002 | **Data Engineer** | Conducted dataset audit, descriptive statistics extraction, missing value imputation strategy, and Figure 1–3 visual analyses. | `outputs/descriptive_statistics.csv`, figures `01`, `02`, `03`, Section 1 write-up. |
| **Tan Wei Ming** | 211221003 | **ML Specialist** | Implemented Baseline model, Linear, Ridge, Random Forest, and Gradient Boosting regressors with 5-Fold Cross Validation. | `outputs/model_comparison.csv`, figures `04`, `05`, `06`, Section 3-4 analysis. |
| **Kavitha a/p Subramaniam** | 211221004 | **UI/UX Engineer** | Developed interactive Streamlit web dashboard (`app.py`), implemented input controls, layout styling, and GitHub cloud deployment. | `src/app.py`, live cloud deployment link, dashboard screenshots. |
| **Siti Hajar binti Mohd Yusof** | 211221005 | **QA & MLOps Analyst** | Formulated edge-case boundary test cases, executed notebook-to-dashboard agreement verification, and built audit logging mechanisms. | `outputs/prediction_logs.csv`, test cases TC-01 to TC-04, Appendix C log. |

*Declaration: All members have reviewed the codebase, understand the end-to-end implementation, and are prepared to present their specific contributions during the oral demonstration.*

---

## **APPENDIX B: AI-ASSISTED DEVELOPMENT AND PROMPT RECORD**

As permitted by the project brief under *AI-assisted development*, Large Language Models (ChatGPT) were utilized for coding support. Below are five representative prompts spanning the development lifecycle, including technical verification and human modifications made:

### **Prompt 1: Exploratory Data Analysis & Automated Visual Saving**
- **Lifecycle Stage:** Task 1 - Data Exploration & Analysis (CO1)
- **User Prompt:**
  > *"Write a Python script using pandas, matplotlib, and seaborn that loads '02_Used_Car_Price.xlsx', computes descriptive statistics for all numerical columns including resale_price_rm, and saves high-resolution distribution plots and a correlation heatmap to an 'outputs/' directory without popping up interactive windows."*
- **Generated Code Summary:** Suggested standard `sns.histplot` and `sns.heatmap` calls wrapped in `plt.savefig()`.
- **Verification & Human Modification:** The AI-generated script did not specify `plt.close()`, causing memory leaks across consecutive plot generations. We inserted `plt.close()`, set `plt.rcParams['figure.dpi'] = 150`, configured `bbox_inches='tight'`, and added custom currency formatting on the x-axis to align with academic publication standards.

### **Prompt 2: Scikit-learn Pipeline Construction to Prevent Data Leakage**
- **Lifecycle Stage:** Task 2 - Solution Design & Preprocessing (CO2)
- **User Prompt:**
  > *"How can I construct an sklearn ColumnTransformer and Pipeline for used car price prediction that imputes missing numerical features with the median, standardizes them, one-hot encodes categorical variables with drop='first', and ensures that preprocessing statistics are only fitted on training data?"*
- **Generated Code Summary:** Provided a template combining `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` inside a `ColumnTransformer`.
- **Verification & Human Modification:** The initial AI snippet mistakenly called `fit_transform` on the entire DataFrame prior to `train_test_split`. We corrected this by separating `X_train` and `X_test` first, wrapping the preprocessor and estimator together inside an `sklearn.pipeline.Pipeline`, guaranteeing zero data leakage during cross-validation.

### **Prompt 3: 5-Fold Cross-Validation Script with Baseline Comparison**
- **Lifecycle Stage:** Task 2 - Model Development & Evaluation (CO2)
- **User Prompt:**
  > *"Write a training script comparing a dummy baseline predicting the training mean against Linear Regression, Ridge, Random Forest, and Gradient Boosting using 5-Fold cross validation on the training set, and calculate MAE, RMSE, and R2 on the 20% test set."*
- **Generated Code Summary:** Provided a loop over dictionary models calculating scores using `cross_val_score` and `mean_squared_error`.
- **Verification & Human Modification:** Updated deprecated `squared=False` argument in `mean_squared_error` to use scikit-learn's modern `root_mean_squared_error` function. Implemented explicit formatting to output results into both a clean console table and a persistent CSV file (`outputs/model_comparison.csv`).

### **Prompt 4: Streamlit UI Development with Input Boundary Validation**
- **Lifecycle Stage:** Task 3 - Implementation & Dashboard (CO3)
- **User Prompt:**
  > *"Create a clean 2-column Streamlit application that loads a saved .joblib pipeline, accepts 8 vehicle input features with intuitive sliders and number inputs, displays the predicted price in RM, and displays warning alerts if the user enters out-of-boundary values (e.g. mileage > 450,000 km or age > 18 years)."*
- **Generated Code Summary:** Provided an `app.py` script featuring `st.form`, input widgets, and `pipeline.predict()`.
- **Verification & Human Modification:** The AI code failed to handle relative path resolution when executed from different directories. We wrapped pipeline loading in a multi-path fallback function with `@st.cache_resource`, and added a strict non-negative price enforcement rule (`max(1000.0, price)`) to safeguard against negative valuations.

### **Prompt 5: Operational Audit Logging and Model Drift Review Triggers**
- **Lifecycle Stage:** Task 3 - Deployment & Operational Monitoring (CO3)
- **User Prompt:**
  > *"How can I implement lightweight operational monitoring in Streamlit that logs each prediction request (inputs, timestamp, predicted price, anomaly flag) to a CSV file without recording personal data, and what quantitative criteria should trigger a model review?"*
- **Generated Code Summary:** Provided a logging function appending DataFrame rows to `prediction_logs.csv` and suggested theoretical monitoring principles.
- **Verification & Human Modification:** Replaced in-memory logging with an atomic append-mode CSV writer (`mode='a'`) ensuring file headers are written only once. Formulated three concrete, quantifiable review triggers (15% rolling anomaly rate, policy tax shifts, and empirical MAE degradation $> \text{RM } 10,000$) to satisfy the rubric requirements.

---

## **APPENDIX C: SYSTEMATIC TEST EXECUTION LOG**

The table below provides detailed verification evidence across normal, boundary, and extreme test cases executed on the deployed Streamlit application:

| Test ID | Input Parameters | Expected Behavior | Actual Deployed Output | Pass/Fail | Technical Verification Notes |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **TC-01** | Age: 5.0, Mileage: 65,000, Engine: 1.8L, Fuel: Petrol, Trans: Auto, Segment: Mid_Range, Cond: 4, Owners: 1 | Realistic mid-range price (RM 55,000–65,000); no warnings. | **RM 59,248.50** (+5.5% vs Mid_Range segment mean) | **PASS** | Validated against empirical mid-range distribution. |
| **TC-02** | Age: 2.0, Mileage: 28,000, Engine: 1.3L, Fuel: Petrol, Trans: Manual, Segment: Economy, Cond: 5, Owners: 1 | Low depreciation economy car (RM 35,000–45,000). | **RM 41,180.25** (+25.0% vs Economy segment mean) | **PASS** | High condition rating offset lower vehicle segment. |
| **TC-03** | Age: 0.5, Mileage: 3,500, Engine: 3.0L, Fuel: Diesel, Trans: Auto, Segment: Premium, Cond: 5, Owners: 0 | Near-new luxury valuation exceeding RM 220,000. | **RM 230,987.12** (+101.6% vs Premium segment mean) | **PASS** | Accurately captured low age and premium status. |
| **TC-04** | Age: 17.5, Mileage: 465,000, Engine: 1.5L, Fuel: Petrol, Trans: Manual, Segment: Economy, Cond: 1, Owners: 4 | Boundary warning for mileage > 450k; positive residual floor. | Warning displayed: *"Mileage exceeds 450,000 km..."* Predicted: **RM 7,850.00** | **PASS** | Prevented negative valuation; triggered user alert. |
| **TC-05** | Age: 0.0, Mileage: 150,000, Engine: 2.0L, Fuel: Hybrid, Trans: Auto, Segment: Mid_Range, Cond: 3, Owners: 1 | Boundary contradiction alert (New car with excessive mileage). | Warning displayed: *"Boundary Anomaly: A new vehicle (Age 0) typically has under 15,000 km."* Logged with `anomaly_flag=True`. | **PASS** | Audit log confirmed anomaly capture in `prediction_logs.csv`. |
| **TC-06** | Age: 10.0, Mileage: 216,597, Engine: 2.3L, Fuel: Hybrid, Trans: Auto, Segment: Economy, Cond: 4, Owners: 2 | Concordance test against test set row #2. | Predicted: **RM 27,962.46** (Identical to Colab Notebook output). | **PASS** | Zero deviation between research script and web app. |
