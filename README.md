# Impact of AI and Social Media on Student Mental Health

## Predictive Regression Modeling and Empirical Behavioral Analysis

---

### Project Overview

This repository provides an enterprise-grade Machine Learning pipeline and empirical study investigating the statistical impact of digital media immersion on student psychological well-being. Using behavioral records from 16,000 students across various educational tiers, this project models and predicts `Mental_Health_Score` as a function of daily social media consumption, generative AI tool utilization, sleep duration, physical activity, and demographic attributes.

The core implementation utilizes an Ordinary Least Squares (OLS) Multiple Linear Regression framework integrated within a scikit-learn `Pipeline` and `ColumnTransformer` architecture. The modeling workflow includes rigorous 5-fold cross-validation, classical Gauss-Markov assumption diagnostics, standardized feature effect estimation, and automated artifact persistence for production deployment.

---

### Key Research Questions

1. To what extent does daily social media consumption inversely correlate with student mental health scores?
2. Does daily generative AI tool usage demonstrate an independent negative, neutral, or positive association with psychological well-being?
3. Can positive lifestyle interventions, specifically adequate sleep duration and daily physical exercise, statistically buffer the detrimental impacts of elevated digital screen exposure?
4. Do demographic parameters (age, gender identity, education level) account for substantial variance in mental health outcomes compared to behavioral lifestyle patterns?

---

### Dataset Summary

The dataset comprises 16,000 observational student records with zero missing or corrupted values.

* **Source File**: `Dataset/AI_SocialMedia_Student_Dataset.csv`
* **Sample Size**: 16,000 records
* **Total Attributes**: 10 columns

#### Feature Schema

| Attribute | Tipe | Role | Description | Range / Categories |
| :--- | :--- | :--- | :--- | :--- |
| `Student_ID` | String | Identifier | Unique student tracking identifier (excluded from training) | `STU_00001` to `STU_16000` |
| `Age` | Integer | Predictor | Student chronological age | 13 to 25 years |
| `Gender` | String | Predictor | Student gender identity | `Male`, `Female`, `Non-binary` |
| `Education_Level` | String | Predictor | Current educational attainment tier | `High School`, `College`, `University` |
| `Daily_Social_Media_Hours` | Float | Predictor | Self-reported daily social media exposure in hours | 0.00 to 14.00 hours |
| `Daily_AI_Tool_Usage_Hours` | Float | Predictor | Daily usage of generative AI / study assistance tools in hours | 0.00 to 9.50 hours |
| `Sleep_Hours` | Float | Predictor | Average daily sleep duration in hours | 2.00 to 11.15 hours |
| `Physical_Activity_Hours` | Float | Predictor | Daily physical exercise and sports activity in hours | 0.00 to 5.00 hours |
| `Mental_Health_Score` | Float | **Primary Target** | Standardized mental health index | 32.56 to 91.76 (Mean: 72.49) |
| `Physical_Health_Score` | Float | Secondary Target | Standardized physical health index (isolated to avoid leakage) | 48.03 to 99.98 (Mean: 88.02) |

---

### Methodology and Pipeline Architecture

The modeling architecture follows scikit-learn best practices, ensuring zero data leakage between training and evaluation splits.

```
Data Ingestion (16,000 records)
       |
Feature Matrix Splitting (80% Train / 20% Test)
       |
Scikit-Learn ColumnTransformer
  |-- Numeric Features (5): StandardScaler
  |-- Categorical Features (2): OneHotEncoder (drop="first")
       |
Linear Regression Estimator
       |
5-Fold Cross-Validation Evaluation
       |
Out-of-Sample Performance Testing & Residual Diagnostics
       |
Knowledge & Model Artifact Serialization (Model_Knowledge/)
```

1. **Feature Separation**: Identifier columns (`Student_ID`) and secondary targets (`Physical_Health_Score`) are dropped from the predictor matrix.
2. **Train-Test Partitioning**: 80% training partition (12,800 records) and 20% holdout test partition (3,200 records) split with fixed random seed (`random_state=42`).
3. **Preprocessing Pipeline**:
   * Continuous variables (`Age`, `Daily_Social_Media_Hours`, `Daily_AI_Tool_Usage_Hours`, `Sleep_Hours`, `Physical_Activity_Hours`) transformed via `StandardScaler`.
   * Categorical variables (`Gender`, `Education_Level`) encoded via `OneHotEncoder(drop="first", sparse_output=False)` to prevent multi-collinearity dummy traps.
4. **Estimation**: Ordinary Least Squares `LinearRegression` fitted within the serialized pipeline.
5. **Cross-Validation**: 5-Fold stratified K-Fold cross-validation (`shuffle=True`, `random_state=42`) executed on the training partition.

---

### Empirical Findings and Model Interpretation

Because continuous predictors are standardized to zero mean and unit variance ($\mu=0, \sigma=1$), regression coefficients represent standardized effect sizes directly comparable in magnitude.

* **Model Intercept**: 72.52 points (Baseline student profile score)

#### Standardized Feature Impact Ranking

| Rank | Feature Name | Standardized Coefficient | Direction | Analytical Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Daily_Social_Media_Hours` | -2.9110 | Negative | Strongest predictor of mental health degradation. Higher exposure significantly decreases scores. |
| 2 | `Sleep_Hours` | +1.8595 | Positive | Dominant protective factor. Each standard deviation increase yields an approximate 1.86 point increase. |
| 3 | `Physical_Activity_Hours` | +1.7244 | Positive | Strong restorative lifestyle intervention counteracting sedentary digital screen duration. |
| 4 | `Daily_AI_Tool_Usage_Hours` | -0.7836 | Negative | Moderate negative relationship. Unmonitored academic/generative AI usage correlates with lower well-being. |
| 5 | `Education_Level_University` | +0.4802 | Positive | Slight positive adjustment relative to baseline college tier. |
| 6 | `Education_Level_High School` | -0.3836 | Negative | Slight negative adjustment relative to college students. |
| 7 | `Age` | -0.2565 | Negative | Marginal age effect across the 13-25 cohort. |
| 8 | `Gender_Non-binary` | -0.1932 | Negative | Small relative delta compared to female baseline. |
| 9 | `Gender_Male` | -0.1762 | Negative | Negligible difference compared to female baseline. |

#### Key Takeaways

1. **Digital Exposure**: Social media exposure exerts nearly four times the negative impact magnitude of generative AI tools (-2.91 vs -0.78).
2. **Restorative Lifestyle Buffer**: Combined restorative behaviors (Sleep + Physical Activity: combined positive coefficient +3.58) provide strong compensatory protection against digital screen impacts (-3.69 total negative digital coefficient).
3. **Demographic Invariance**: Behavioral habits (screen time, sleep, exercise) account for over 92% of the explained model variance, rendering demographic indicators secondary.

---

### Model Evaluation and Performance Metrics

The model demonstrates strong generalizability and consistency between cross-validation and out-of-sample test splits, indicating no overfitting.

| Metric | Training Split | 5-Fold Cross-Validation | Out-of-Sample Test Split |
| :--- | :---: | :---: | :---: |
| **R-squared ($R^2$)** | 0.2464 | 0.2449 (+/- 0.0097) | **0.2381** |
| **Adjusted R-squared** | - | - | **0.2365** |
| **Mean Absolute Error (MAE)** | 6.31 points | 6.38 points (+/- 0.076) | **6.33 points** |
| **Root Mean Squared Error (RMSE)** | 8.01 points | 8.04 points (+/- 0.091) | **8.02 points** |
| **Mean Squared Error (MSE)** | 64.12 | 64.69 | **64.27** |

*Note: An $R^2$ of ~0.24 is typical and expected for psychological and behavioral human outcomes, reflecting that complex individual mental health is influenced by unobserved environmental, genetic, and academic pressures beyond digital screen hours.*

---

### Regression Diagnostics and Assumption Verification

The trained model was subjected to formal Gauss-Markov diagnostic tests:

1. **Linearity**: Evaluated via Actual vs. Predicted scatter plot; observed values track the theoretical 45-degree identity reference trajectory across the interquartile range.
2. **Normality of Residuals**: Verified via Kernel Density Estimation (KDE) and normal Quantile-Quantile (Q-Q) plot. Residuals exhibit near-perfect symmetry around zero with mild tail deviations at extreme boundaries.
3. **Homoscedasticity**: Evaluated via Residuals vs. Fitted Values plot. Residual variance remains constant across the entire spectrum of fitted values, confirming absence of severe heteroscedasticity.
4. **Multicollinearity**: Variance Inflation Factors (VIF) and Pearson cross-correlations confirm low collinearity among independent predictors ($r < 0.35$).

---

### Repository Structure

```
.
|-- Dataset/
|   `-- AI_SocialMedia_Student_Dataset.csv            # Raw dataset (16,000 records)
|-- Model/
|   `-- Mental_Health_Linear_Regression.ipynb         # Fully executed Jupyter Notebook
|-- Model_Knowledge/
|   |-- mental_health_linear_regression_pipeline.joblib # Serialized scikit-learn pipeline
|   |-- evaluation_metrics.json                       # Comprehensive evaluation metrics
|   |-- coefficients_knowledge.json                   # Standardized coefficients & intercept
|   `-- model_metadata.json                           # Metadata, feature schema & manifests
|-- LICENSE                                           # MIT License terms
|-- README.md                                         # Project documentation and analysis
`-- requirements.txt                                  # Environment dependencies specification
```

---

### Installation and Environment Setup

#### Prerequisites

* Python 3.9+
* pip or conda

#### Dependencies

Clone the repository and install the required packages:

```bash
git clone https://github.com/NumiKun/AI---Social-Media-Impact.git
cd AI---Social-Media-Impact
pip install -r requirements.txt
```

---

### Model Inference Quickstart

The saved model artifact contains the entire preprocessing and estimation pipeline. To score new student profiles without manual feature scaling or encoding:

```python
import joblib
import pandas as pd

pipeline_path = "Model_Knowledge/mental_health_linear_regression_pipeline.joblib"
model_pipeline = joblib.load(pipeline_path)

new_students = pd.DataFrame([
    {
        "Age": 20,
        "Gender": "Female",
        "Education_Level": "University",
        "Daily_Social_Media_Hours": 6.5,
        "Daily_AI_Tool_Usage_Hours": 2.5,
        "Sleep_Hours": 5.0,
        "Physical_Activity_Hours": 0.5,
    },
    {
        "Age": 21,
        "Gender": "Male",
        "Education_Level": "College",
        "Daily_Social_Media_Hours": 1.5,
        "Daily_AI_Tool_Usage_Hours": 1.0,
        "Sleep_Hours": 8.0,
        "Physical_Activity_Hours": 2.5,
    }
])

predictions = model_pipeline.predict(new_students)
for idx, score in enumerate(predictions):
    print(f"Student {idx + 1} Predicted Mental Health Score: {score:.2f}")
```

#### Expected Output

```
Student 1 Predicted Mental Health Score: 66.85
Student 2 Predicted Mental Health Score: 78.42
```

---

### Model Knowledge Artifacts

The `Model_Knowledge/` folder contains versioned machine-readable artifacts designed for auditability and automated continuous integration:

* **`mental_health_linear_regression_pipeline.joblib`**: Full pipeline object encapsulating numerical scalers, categorical encoders, and regression coefficients.
* **`evaluation_metrics.json`**: Key regression indicators ($R^2$, MAE, MSE, RMSE, 5-Fold cross-validation).
* **`coefficients_knowledge.json`**: Intercept and feature weights ordered by absolute impact magnitude.
* **`model_metadata.json`**: Dataset record count, feature schemas, training roles, and file manifests.

---

### Limitations and Future Enhancements

1. **Non-Linear Interactions**: While linear models offer optimal interpretability, non-linear tree ensembles (LightGBM, XGBoost) and polynomial feature interactions could capture subtle non-linear thresholds (such as tipping points in screen exposure).
2. **Cross-Sectional vs Longitudinal**: The observational dataset represents cross-sectional points in time; longitudinal tracking would strengthen causal inference.
3. **Multi-Target Joint Modeling**: Implementing multi-output models to jointly predict `Mental_Health_Score` and `Physical_Health_Score` simultaneously.

---

### Author and Contribution

* **Project**: AI and Social Media Impact Analysis
* **Author**: NumiKun
* **Repository**: [AI---Social-Media-Impact](https://github.com/NumiKun/AI---Social-Media-Impact)

---

### License

This project is licensed under the terms of the [MIT License](LICENSE).
