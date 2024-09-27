import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/mikhailfernandez/Desktop/Personal Projects/Customer Segmentation and revenue forecasting/data/Sales.csv'
data = pd.read_csv(file_path)


X = data[['Customer_Age', 'Order_Quantity', 'Revenue', 'Profit']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

segment_names = ['High-Spenders', 'Moderate Spenders',
                 'Occasional Shoppers', 'Low-Value Customers']
cluster_value = data.groupby('Cluster').agg(
    {'Revenue': 'sum', 'Customer_Age': 'count'}).reset_index()
cluster_value_sorted = cluster_value.sort_values(by='Revenue', ascending=False)
cluster_value_sorted['Segment_Name'] = segment_names
cluster_name_map = dict(
    zip(cluster_value_sorted['Cluster'], cluster_value_sorted['Segment_Name']))
data['Segment_Name'] = data['Cluster'].map(cluster_name_map)

category_popularity = data.groupby(['Segment_Name', 'Product_Category']).agg({
    'Order_Quantity': 'sum',
    'Revenue': 'sum'
}).reset_index()

category_popularity_sorted_revenue = category_popularity.sort_values(
    by=['Segment_Name', 'Revenue'], ascending=False)
category_popularity_sorted_orders = category_popularity.sort_values(
    by=['Segment_Name', 'Order_Quantity'], ascending=False)


def annotate_bars(ax, orient='v'):
    for p in ax.patches:
        if orient == 'v':
            ax.annotate(f'{p.get_height():,.0f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9),
                        textcoords='offset points')
        else:
            ax.annotate(f'{p.get_width():,.0f}',
                        (p.get_width(), p.get_y() + p.get_height() / 2.),
                        ha='center', va='center', xytext=(9, 0),
                        textcoords='offset points')


plt.figure(figsize=(12, 8))
ax = sns.barplot(data=category_popularity_sorted_revenue,
                 x='Product_Category', y='Revenue', hue='Segment_Name', dodge=True)
plt.title('Most Popular Product Categories by Customer Segment (Based on Revenue)')
plt.xticks(rotation=90)
plt.ylabel('Total Revenue')
plt.xlabel('Product Category')
plt.legend(title='Customer Segment')
annotate_bars(ax, orient='v')
plt.show()

plt.figure(figsize=(12, 8))
ax = sns.barplot(data=category_popularity_sorted_orders, x='Product_Category',
                 y='Order_Quantity', hue='Segment_Name', dodge=True)
plt.title(
    'Most Popular Product Categories by Customer Segment (Based on Order Quantity)')
plt.xticks(rotation=90)
plt.ylabel('Total Order Quantity')
plt.xlabel('Product Category')
plt.legend(title='Customer Segment')

annotate_bars(ax, orient='v')
plt.show()
