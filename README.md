# Customer Segmentation and RFM Analysis

Customer segmentation using RFM (Recency, Frequency, Monetary) analysis and KMeans clustering on real e-commerce transaction data. Built with Python, Pandas, Scikit-learn, Plotly, and Flask.

## Overview

Performed behavioral customer segmentation on the UCI Online Retail dataset containing real transactions from a UK-based online retail company. Engineered RFM features from over 500,000 transactions across 4,338 customers and applied KMeans clustering to identify 5 distinct customer segments with actionable retention strategies.

## Results

| Segment | Customers | Revenue Share | Strategy |
|---------|-----------|---------------|----------|
| Recent Customers | 3,048 | 45.72% | Welcome series and onboarding |
| Loyal Customers | 221 | 35.67% | Referral and cross-sell programs |
| Champions | 6 | 12.88% | VIP loyalty tier |
| Lost | 1,063 | 5.73% | Low-cost newsletter only |

## Features

- 541,909 real transactions from the UCI Machine Learning Repository
- RFM feature engineering: Recency, Frequency, Monetary value per customer
- KMeans clustering with automated elbow method for optimal k selection
- Strategic CRM segment labeling with business recommendations per segment
- 3D interactive Plotly scatter plot of customer segments
- 4-panel executive dashboard with dark theme
- CSV exports for Power BI and Tableau integration

## Live Demo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z1WtfYnwjbWKGWNnQyR_zA6FfmmoTYlb?usp=sharing)

**Live Demo:** [https://customer-segmentation-rfm-opir.onrender.com](https://customer-segmentation-rfm-opir.onrender.com)

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Plotly, Scikit-learn, Flask

## Deployment

The model is deployed as a Flask web application on Render with customer lookup by ID, segment label with color badge, RFM metric comparison against averages, and strategic business recommendations.
