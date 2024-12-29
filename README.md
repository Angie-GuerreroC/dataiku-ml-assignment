# Dataiku Interview Assignment

## Overview
This project addresses the task of predicting income levels (greater or less than $50K) based on demographic and socioeconomic data from the U.S. Census Bureau. The repository demonstrates a complete machine learning workflow, including data preprocessing, exploratory analysis, model development, and performance evaluation.

---

## Project Objective
The goal is to identify key factors influencing income levels and develop predictive models for classification. This analysis aids in understanding income disparities and informs resource allocation and policy-making decisions.

---

## Business Value
- **Effective Resource Allocation**: Target regions with higher populations earning less than $50K for focused interventions.
- **Address Systemic Challenges**: Provide insights into factors driving income disparities, such as education and job availability.
- **Optimize Federal Support**: Ensure resources are directed to regions and populations with the greatest need.

---

## Features
1. **Data Preprocessing**:
   - Handling missing values and duplicates.
   - Encoding categorical variables.
   - Feature scaling using Min-Max normalization.

2. **Exploratory Data Analysis (EDA)**:
   - Distribution analysis of key features.
   - Relationships between income and demographic attributes (e.g., gender, education, race).

3. **Feature Engineering**:
   - Creation of new features like `capital_net_gain`.
   - Selection of top features using statistical methods.

4. **Handling Class Imbalance**:
   - Implementation of SMOTE for balanced training data.

5. **Model Development and Evaluation**:
   - Training machine learning models: 
     - Random Forest
     - Logistic Regression
     - XGBoost
   - Hyperparameter tuning for optimal performance.
   - Visualization of performance metrics (AUC, precision-recall curves, confusion matrices).

6. **Model Comparison**:
   - Side-by-side evaluation of models based on AUC and other metrics.
   - Selection of the best model for final evaluation.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`
  - Handling Class Imbalance: `imbalanced-learn`

---

## Project Structure
```plaintext
dataiku-ml-assignment/
│
├── data/                     # Folder for input datasets (train/test CSV files)
├── visuals/                  # Folder for generated plots and visualizations
├── scripts/                  # Python scripts for preprocessing, modeling, and evaluation
│   ├── census_project.ipynb  # Main Script
│
├── README.md                 # Project README
└── requirements.txt          # List of required Python packages