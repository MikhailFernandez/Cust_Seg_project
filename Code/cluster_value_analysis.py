import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

file_path = '/Users/mikhailfernandez/Desktop/Personal Projects/Customer Segmentation and revenue forecasting/data/Sales.csv'
data = pd.read_csv(file_path)

X = data[['Customer_Age', 'Order_Quantity', 'Revenue', 'Profit']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_value = data.groupby('Cluster').agg({
    'Revenue': ['mean', 'sum'],
    'Profit': ['mean', 'sum'],
    'Customer_Age': 'count'
}).reset_index()

cluster_value.columns = ['Cluster', 'Avg_Revenue',
                         'Total_Revenue', 'Avg_Profit', 'Total_Profit', 'Customer_Count']

cluster_value_sorted = cluster_value.sort_values(
    by='Total_Revenue', ascending=False)

segment_names = ['High-Spenders', 'Moderate Spenders',
                 'Occasional Shoppers', 'Low-Value Customers']
cluster_value_sorted['Segment_Name'] = segment_names

cluster_name_map = dict(
    zip(cluster_value_sorted['Cluster'], cluster_value_sorted['Segment_Name']))
data['Segment_Name'] = data['Cluster'].map(cluster_name_map)

print(cluster_value_sorted)

plt.figure(figsize=(10, 6))

plt.bar(cluster_value_sorted['Segment_Name'], cluster_value_sorted['Total_Revenue'],
        color='blue', alpha=0.7, label='Total Revenue')

plt.bar(cluster_value_sorted['Segment_Name'], cluster_value_sorted['Total_Profit'],
        color='green', alpha=0.5, label='Total Profit')

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, p: format(int(x), ',')))
plt.title('Total Revenue and Profit by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Amount ($)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(cluster_value_sorted['Customer_Count'], labels=cluster_value_sorted['Segment_Name'],
        autopct='%1.1f%%', startangle=140, colors=['blue', 'green', 'red', 'purple'])
plt.title('Customer Distribution by Segment')
plt.show()
