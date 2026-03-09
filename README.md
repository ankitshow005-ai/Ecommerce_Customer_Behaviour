# E-Commerce Customer Churn Prediction

This project builds a machine learning system to predict customer churn in an e-commerce platform. The objective is to identify customers who are likely to leave the platform so that businesses can proactively intervene with retention strategies.

The project implements a structured machine learning workflow including data validation, preprocessing, feature engineering, model training, feature selection, evaluation, and reporting. The entire pipeline is orchestrated using DVC (Data Version Control) to ensure reproducibility and modular experimentation.

---

# Problem Statement

Customer churn is a major challenge for e-commerce businesses. Retaining existing customers is significantly cheaper than acquiring new ones, making churn prediction a critical task.

The goals of this project are:

- Predict whether a customer will churn
- Identify behavioral patterns associated with churn
- Build a reproducible machine learning pipeline
- Compare multiple models and evaluate performance

Special focus is placed on minimizing false negatives because failing to detect a churner means losing the opportunity for retention.

---

# Dataset Overview

The dataset contains behavioral and demographic data for 50,000 customers.

| Property | Value |
|--------|--------|
| Number of customers | 50,000 |
| Number of features | 25 |
| Target variable | Customer churn |
| Retained customers | 71.1% |
| Churned customers | 28.9% |

The dataset includes information related to:

- Customer demographics
- Engagement behavior
- Purchase activity
- Recency and inactivity signals
- Customer service interactions

---

# Exploratory Data Analysis

## Dataset Quality

- No duplicate rows or columns were found
- No features had extreme missingness (>30%)
- No constant features were detected
- Some numerical features contained invalid negative values which were corrected during preprocessing

All features were retained initially because they were considered business-relevant and potentially predictive.

---

## Churn Distribution

The dataset shows moderate class imbalance.

| Class | Percentage |
|------|------------|
| Retained | 71.1% |
| Churned | 28.9% |

This imbalance is manageable without aggressive resampling.

Demographic variables such as gender, country, quarter, and city show churn rates between 27–31%, indicating that demographics alone do not strongly drive churn.

---

## Key Behavioral Insights

Engagement features show the strongest relationship with churn.

Churned users demonstrate:

- ~28% fewer logins
- Fewer pages viewed
- Lower session duration
- ~20–25% lower mobile and email engagement

Recency and friction indicators further increase churn risk:

- Churned users are inactive 4–5 days longer
- Cart abandonment is ~10 percentage points higher
- Customer service calls are ~33% higher

---

## Multivariate Behavioral Patterns

Important churn signals emerge when combining multiple features.

### Login Frequency × Recency

| Behavior | Churn Rate |
|--------|-------------|
| Low engagement + high inactivity | >50% |
| High engagement | ~19–21% |

### Engagement as a protective factor

High engagement significantly reduces churn even when other negative behaviors exist.

Examples:

- High cart abandonment + high engagement → moderate churn
- High service calls + high engagement → moderate churn

### Session quality and friction

Low session duration combined with high service calls produces churn rates exceeding 50%.

---

## Final EDA Takeaway

Customer churn is primarily driven by behavioral disengagement and inactivity rather than demographics or tenure.

Engagement acts as the strongest protective factor against churn.

---

# Machine Learning Pipeline

The project is implemented as a reproducible machine learning pipeline using DVC.

Pipeline stages:

Raw Data → Data Ingestion → Data Validation → Data Preprocessing → Feature Engineering → Train/Test Split + Encoding → Model Training → Model Evaluation → Report Generation

Model training includes:

- Decision Tree
- Random Forest
- Random Forest with Feature Selection

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

## Decision Tree

| Metric | Score |
|------|------|
| Accuracy | 0.8486 |
| Precision | 0.7287 |
| Recall | 0.7585 |
| F1 Score | 0.7433 |
| ROC-AUC | 0.8219 |

---

## Random Forest

| Metric | Score |
|------|------|
| Accuracy | 0.914 |
| Precision | 0.8622 |
| Recall | 0.836 |
| F1 Score | 0.8489 |
| ROC-AUC | 0.9248 |

Confusion matrix values:

| | Predicted Retained | Predicted Churn |
|---|---|---|
| Actual Retained | 6724 | 386 |
| Actual Churn | 474 | 2416 |

---

## Random Forest + Feature Selection

Feature selection reduced the feature space from 32 features to 21 features using cumulative feature importance.

| Metric | Score |
|------|------|
| Accuracy | 0.9145 |
| Precision | 0.8667 |
| Recall | 0.8322 |
| F1 Score | 0.8491 |
| ROC-AUC | 0.9258 |

Confusion errors:

| Error Type | Count |
|------------|-------|
| False Negatives | 485 |
| False Positives | 370 |

---

# Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------|------|------|------|------|------|
| Decision Tree | 0.8486 | 0.7287 | 0.7585 | 0.7433 | 0.8219 |
| Random Forest | 0.9140 | 0.8622 | 0.8360 | 0.8489 | 0.9248 |
| Random Forest + Feature Selection | 0.9145 | 0.8667 | 0.8322 | 0.8491 | 0.9258 |

Random Forest and Random Forest with feature selection significantly outperform the baseline Decision Tree.

---

# Threshold Selection

The classification threshold was adjusted to 0.40 instead of the default 0.50.

Reason:

- Lower threshold increases recall
- More churners are detected
- False negatives are reduced

Reducing false negatives is important because each missed churner represents a lost retention opportunity.

---

# Project Structure

Ecommerce_Customer_Behaviour

data_processed  
├ raw  
├ cleaned  
├ featured  
└ dependency_split  

models  
├ decision_tree  
├ rf_tuned  
└ rf_feature_selection  

reports  

src  
├ data_ingestion.py  
├ data_validation.py  
├ data_preprocessing.py  
├ feature_engineering.py  
├ data_split_encode.py  
├ dt.py  
├ rf.py  
├ rf_fs.py  
├ reporting.py  
└ model_comparison.py  

params.yaml  
dvc.yaml  
requirements.txt  
README.md  

---

# Reproducibility

The entire pipeline can be reproduced using DVC.

Run:

dvc repro

DVC executes all pipeline stages defined in dvc.yaml.

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

Clone the repository

git clone https://github.com/<your-username>/Ecommerce_Customer_Behaviour.git

Navigate to the project directory

cd Ecommerce_Customer_Behaviour

Install dependencies

pip install -r requirements.txt

Run the full pipeline

dvc repro

---

# Business Impact

Churn prediction enables businesses to proactively retain customers who are at risk of leaving.

Potential applications include:

- Targeted retention campaigns
- Personalized discounts for disengaged users
- Early detection of customer dissatisfaction
- Prioritization of high-risk customers for support teams

Reducing churn even slightly can significantly improve long-term revenue and customer lifetime value.

---

# Author

Ankit Show  
Data Science and Machine Learning