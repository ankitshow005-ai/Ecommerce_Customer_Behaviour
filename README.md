# Ecommerce_Customer_Behaviour

# E-Commerce Customer Churn Prediction

This project builds a machine learning system to predict customer churn in an e-commerce platform. The objective is to identify customers who are likely to leave the platform so that businesses can proactively intervene with retention strategies.

The project implements a complete machine learning workflow including data validation, preprocessing, feature engineering, model training, feature selection, evaluation, and automated reporting. The pipeline is orchestrated using DVC to ensure reproducibility and structured experimentation.

---

# Problem Statement

Customer churn is a critical challenge for e-commerce companies. Retaining existing customers is significantly cheaper than acquiring new ones, making churn prediction an important business problem.

The goal of this project is to:

- Predict whether a customer will churn
- Understand behavioral patterns associated with churn
- Build a reproducible ML pipeline that can be extended for experimentation

The project places particular emphasis on minimizing **false negatives**, since failing to detect a churner represents a lost opportunity for retention.

---

# Dataset Overview

The dataset contains behavioral and demographic information for **50,000 customers**.

| Property | Value |
|--------|--------|
| Number of rows | 50,000 |
| Number of features | 25 |
| Target variable | Customer churn |
| Retained customers | 71.1% |
| Churned customers | 28.9% |

The dataset includes features representing:

- Customer demographics  
- Purchase behavior  
- Platform engagement  
- Customer support interactions  
- Recency and activity signals  

---

# Exploratory Data Analysis

## Dataset Quality and Structure

- The dataset contains **50,000 rows and 25 features**.
- No duplicate rows or columns were detected.
- No feature had more than **30% missing values**.
- No feature had a single unique value.
- Several numerical columns contained **invalid negative values**, which were handled during preprocessing.

All features were retained initially because they are **business-relevant and potentially predictive**.

---

## Churn Distribution

The dataset is **moderately imbalanced**.

| Class | Percentage |
|------|------------|
| Retained | 71.1% |
| Churned | 28.9% |

This imbalance is manageable and does not require aggressive resampling.

Demographic distributions such as gender, country, quarter, and city show churn rates between **27% and 31%**, indicating demographics alone do not strongly drive churn.

---

## Univariate Insights

Most numerical variables exhibit **right-skewed distributions**, indicating heavy-tailed behavior and the presence of outliers.

Engagement-related variables show strong separation between churned and retained customers:

- Login frequency  
- Session duration  
- Pages viewed  
- Email opens  
- Mobile app usage  

Demographic features such as **Age, Membership Years, and Payment Method Diversity** show minimal separation between churn classes.

---

## Bivariate Insights

Engagement metrics demonstrate the strongest relationship with churn.

Churned customers show:

- **~28% fewer logins**
- Fewer pages viewed
- Lower session duration
- **20–25% lower mobile and email engagement**

Recency and friction indicators further increase churn risk:

- Churned users are inactive **4–5 days longer**
- Cart abandonment is **~10 percentage points higher**
- Customer service calls are **~33% higher**

Purchase-related metrics become meaningful mainly when combined with engagement or recency.

---

## Multivariate Behavioral Patterns

The most important churn signals appear when combining multiple behavioral features.

### Login Frequency × Recency

This interaction is the strongest predictor.

| Behavior | Churn Rate |
|--------|-------------|
| Low engagement + high inactivity | >50% |
| High engagement | ~19–21% |

### Engagement as a protective factor

High engagement consistently reduces churn even when negative behaviors exist.

Examples:

- High cart abandonment + high engagement → moderate churn
- High service calls + high engagement → moderate churn

### Session quality and friction

Low session duration combined with high customer service calls produces churn rates exceeding **50%**.

### Engagement amplification

Mobile usage and email engagement reinforce retention when combined with other engagement signals.

### Purchase behavior

Total purchases appear predictive only in isolation; once engagement is controlled, their independent effect weakens.

---

## Correlation Structure

Feature correlation analysis reveals clear behavioral clusters.

Engagement features form a highly correlated group:

- Login frequency
- Session duration
- Pages viewed
- Email opens
- Mobile usage

Friction indicators such as cart abandonment and service calls are negatively correlated with engagement.

Demographic features show near-zero correlation with behavioral metrics.

Financial aggregates represent customer value but are weaker predictors of churn.

---

## Key EDA Takeaway

Customer churn is primarily driven by **behavioral disengagement and inactivity**, rather than demographic characteristics.

Engagement acts as the strongest protective factor that mitigates negative signals such as cart abandonment, service issues, or inactivity.

---

# Machine Learning Pipeline

The project implements a reproducible ML pipeline using **DVC**.

Pipeline stages:

```
Data Ingestion
Data Validation
Data Preprocessing
Feature Engineering
Train-Test Split and Encoding
Decision Tree Training
Random Forest Training
Feature Selection
Model Evaluation
Report Generation
```

Each stage is version-controlled and reproducible.

---

# Models Implemented

| Model | Description |
|------|-------------|
| Decision Tree | Baseline interpretable model |
| Random Forest | Primary predictive model |
| Random Forest + Feature Selection | Reduced feature model |

---

# Model Performance

The best-performing model is the **Random Forest classifier with threshold tuning**.

| Metric | Score |
|------|------|
| Accuracy | 0.914 |
| Precision | 0.862 |
| Recall | 0.836 |
| F1 Score | 0.8489 |
| ROC-AUC | 0.9248 |

---

# Confusion Matrix

| | Predicted Retained | Predicted Churn |
|---|---|---|
| Actual Retained | 6724 | 386 |
| Actual Churn | 474 | 2416 |

The model successfully captures most churners while maintaining strong precision.

---

# Feature Selection

Random Forest feature importance was used to reduce dimensionality.

Steps:

1. Train Random Forest
2. Extract feature importance
3. Compute cumulative importance
4. Select features covering **90% cumulative importance**

Feature space reduction:

```
32 features → 21 features
```

Performance remains comparable while improving model simplicity.

---

# Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------|------|------|------|------|------|
| Decision Tree | 0.89 | 0.84 | 0.80 | 0.82 | 0.90 |
| Random Forest | 0.914 | 0.862 | 0.836 | 0.8489 | 0.9248 |
| Random Forest + Feature Selection | 0.918 | 0.9117 | 0.7931 | 0.8483 | 0.9258 |

The Random Forest model achieved the best balance between recall and precision while maintaining strong ROC-AUC performance.

---

# Reports Generated

The pipeline automatically generates evaluation reports.

```
reports/
├ confusion_matrix.png
├ roc_curve.png
├ feature_importance.png
├ model_comparison_accuracy.png
├ model_comparison_f1.png
└ model_comparison_table.csv
```

These reports provide visual insight into model behavior and feature importance.

---

# Project Structure

```
Ecommerce_Customer_Behaviour
│
├ data_processed
│   ├ raw
│   ├ cleaned
│   ├ featured
│   └ dependency_split
│
├ models
│   ├ decision_tree
│   ├ rf_tuned
│   └ rf_feature_selection
│
├ reports
│
├ src
│   ├ data_ingestion.py
│   ├ data_validation.py
│   ├ data_preprocessing.py
│   ├ feature_engineering.py
│   ├ data_split_encode.py
│   ├ dt.py
│   ├ rf.py
│   ├ rf_fs.py
│   ├ reporting.py
│   └ model_comparison.py
│
├ params.yaml
├ dvc.yaml
├ requirements.txt
└ README.md
```

---

# Reproducibility

The entire pipeline can be reproduced using DVC.

Run:

```
dvc repro
```

DVC will execute every stage defined in `dvc.yaml` and regenerate models and reports.

---

# Technologies Used

Programming Language  
Python

Machine Learning  
Scikit-learn

Data Processing  
Pandas  
NumPy

Visualization  
Matplotlib  
Seaborn

Pipeline Orchestration  
DVC

Version Control  
Git

---

# Running the Project

Clone the repository:

```
git clone https://github.com/<your-username>/Ecommerce_Customer_Behaviour.git
```

Navigate to the project directory:

```
cd Ecommerce_Customer_Behaviour
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the pipeline:

```
dvc repro
```

---

# Future Improvements

Potential extensions include:

- Automated hyperparameter optimization
- Experiment tracking using MLflow
- Data drift monitoring
- API-based model deployment
- Business intelligence dashboard for churn insights

---


# Business Impact

Churn prediction enables businesses to proactively retain high-risk customers.

Potential applications include:

• Targeted retention campaigns for customers predicted to churn  
• Personalized offers or discounts for disengaged users  
• Early detection of customer dissatisfaction  
• Prioritization of high-value churn risks for customer support teams  

Reducing churn by even a small percentage can significantly improve long-term revenue and customer lifetime value.


---


# Author

Ankit Show

Data Science and Machine Learning