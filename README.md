# Mall Customer Segmentation Using K-Means Clustering

## Overview

This project segments mall customers into distinct groups based on annual income and spending behaviour using K-Means clustering.

The goal is to help marketing teams understand different customer profiles and design more targeted campaigns, promotions, and retention strategies.

## Business Problem

A mall wants to better understand its customers. Instead of treating all customers the same, the business can segment them based on:

- Annual income
- Spending score
- Age
- Gender

This allows the marketing team to identify high-value customers, budget-conscious customers, and customers with growth potential.

## Dataset

Source: Kaggle Mall Customer Segmentation Dataset

The dataset contains 200 customer records with the following fields:

- CustomerID
- Gender
- Age
- Annual Income
- Spending Score

## Tools Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Methodology

1. Loaded and explored the dataset
2. Checked missing values and data types
3. Renamed columns for easier analysis
4. Visualised relationships between age, income, and spending score
5. Used the elbow method to identify the optimal number of clusters
6. Applied K-Means clustering
7. Interpreted the customer segments from a business perspective

## Key Findings

The model identified five customer segments:

1. Low Income, Low Spending  
2. Low Income, High Spending  
3. Moderate Income, Moderate Spending  
4. High Income, Low Spending  
5. High Income, High Spending  

The most valuable segment is the high income, high spending group. These customers are likely strong candidates for loyalty offers, premium campaigns, and personalised promotions.

The high income, low spending group may represent an opportunity for re-engagement campaigns.

## Business Recommendations

- Create loyalty campaigns for high-income, high-spending customers
- Test targeted promotions for high-income, low-spending customers
- Use budget-friendly offers for low-income, high-spending customers
- Maintain broad engagement campaigns for moderate-income, moderate-spending customers

## Next Steps

- Add purchase history data
- Include customer lifetime value
- Compare K-Means with other clustering methods
- Build an interactive dashboard using Power BI, Tableau, or Streamlit
