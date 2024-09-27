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

kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=[
                          'Customer_Age', 'Order_Quantity', 'Revenue', 'Profit'])

sorted_clusters = cluster_df.sort_values(by='Revenue', ascending=False).index
cluster_names = {sorted_clusters[0]: 'High-Spenders',
                 sorted_clusters[1]: 'Moderate Spenders',
                 sorted_clusters[2]: 'Low Spenders',
                 sorted_clusters[3]: 'Occasional Shoppers'}

data['Segment_Name'] = data['Cluster'].map(cluster_names)

sns.set(style="ticks", palette="pastel")
plt.figure(figsize=(10, 6))

pair_plot = sns.pairplot(data[['Customer_Age', 'Order_Quantity', 'Revenue', 'Profit', 'Segment_Name']],
                         hue='Segment_Name', palette='deep', diag_kind='kde', markers=["o", "s", "D", "^"])

pair_plot.fig.suptitle('Pair Plot of Customer Segments', y=1.02)
pair_plot.set(xticklabels=[])
plt.show()

plt.figure(figsize=(10, 6))
cluster_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Cluster Centers (Average Values for Each Cluster)')
plt.xlabel('Cluster')
plt.ylabel('Feature Values')
plt.xticks(ticks=[0, 1, 2, 3], labels=['High-Spenders',
           'Moderate Spenders', 'Low Spenders', 'Occasional Shoppers'], rotation=0)
plt.show()

cluster_summary = data.groupby('Segment_Name').mean(
)[['Customer_Age', 'Order_Quantity', 'Revenue', 'Profit']]
print("Cluster Summary:\n", cluster_summary)
