Bank Attrition Analysis

This project analyzes the factors that lead to customer attrition (churn) in a bank using data analytics and machine learning. The goal is to identify key drivers of customer churn and build predictive models to help banks retain their customers effectively.

📋 Project Overview

Customer attrition is one of the major challenges faced by banks. Losing customers directly impacts revenue and growth.
This project focuses on analyzing customer behavior, identifying patterns, and predicting churn using a structured data science workflow.

Objectives:

Explore and understand the dataset using Exploratory Data Analysis (EDA)
Identify key features contributing to customer churn
Build and evaluate predictive models to forecast attrition
Provide actionable insights for improving customer retention

The dataset includes details about customers such as:

Demographic Information – Age, Gender, Geography
Account Details – Balance, Credit Score, Number of Products
Activity Metrics – Active status, Estimated Salary, Tenure
Target Variable – Whether the customer has churned (Exited)
Source: The dataset is included within the notebook or loaded from an external CSV file.

Methodology
Data Preprocessing
Handling missing values
Encoding categorical variables
Scaling numerical features
Exploratory Data Analysis (EDA)
Visualizing churn distribution
Correlation analysis between features
Identifying trends and customer behavior patterns
Model Building
Machine learning models applied:
Logistic Regression
Decision Tree
Random Forest
XGBoost
Model comparison using accuracy, precision, recall, F1-score, and ROC-AUC
Model Evaluation
Confusion Matrix
ROC Curve
Feature Importance visualization

Key Insights:
Customers with low account balance and fewer products are more likely to churn.
Tenure and credit score have a significant negative correlation with attrition.
Geography and gender also play a role in predicting churn behavior.
The Random Forest model performed best among all models, offering high accuracy and balanced precision-recall metrics.
