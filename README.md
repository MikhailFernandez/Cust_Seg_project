# Customer Segmentation and Revenue Forecasting Project

## CRISP-DM Process Overview

### 1. **Business Understanding**
The objective of this project was to answer key business questions regarding customer behavior, segmentation, and revenue generation, focusing on the following areas:
- Identifying key customer segments.
- Determining which customer segments are most valuable to the business.
- Analyzing which product categories are popular across customer segments.
- Understanding factors driving the highest revenue.
- Visualizing which countries and states contribute the most and least to revenue.

### 2. **Data Understanding**
The dataset comprised sales transactions, customer demographics, product categories, and geographic data. Key columns included:
- `Customer_Age`, `Order_Quantity`, `Revenue`, `Profit`, `Country`, `State`, `Product_Category`, and `Sub_Category`.
- The dataset was explored to identify patterns and trends, particularly focusing on customer segments and their revenue contributions.

### 3. **Data Preparation**
- The data was preprocessed using PostgreSQL to clean, aggregate, and merge relevant tables.
- Columns were normalized, missing values were handled, and the dataset was structured for analysis.
- The features were selected for clustering and prediction models, including `Customer_Age`, `Order_Quantity`, `Revenue`, and `Profit`.

### 4. **Modeling**
- **K-Means Clustering**: Applied to segment customers into four groups based on purchasing behavior. The segments were named as:
  - High-Value Shoppers
  - Frequent Shoppers
  - Occasional Shoppers
  - Low-Value Shoppers
- **Random Forest Regression**: Used to identify the key factors driving revenue, with `Order_Quantity` and `Product_Category` as the most important features.
- **Logistic Regression**: Implemented to predict the likelihood of a customer becoming a high-revenue customer based on past behavior.

### 5. **Evaluation**
- The K-Means clustering model successfully identified distinct customer segments, allowing for targeted marketing strategies.
- Random Forest and Logistic Regression models provided insights into the primary drivers of revenue, such as product category and order quantity, aiding in strategic decision-making.
- Visualizations in Tableau offered valuable insights into geographic revenue distribution, highlighting top-performing countries and states.

### 6. **Deployment**
- The project was visualized using Python’s plotting libraries and Tableau Public. 
- Key insights and interactive visualizations were created, including:
  - Cluster distribution analysis
  - Feature importance charts for revenue drivers
  - Geographic revenue heatmaps for countries and states.
- These outputs can be used to inform future marketing strategies and business decisions.

