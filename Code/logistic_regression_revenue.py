import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/mikhailfernandez/Desktop/Personal Projects/Customer Segmentation and revenue forecasting/data/Sales.csv'
data = pd.read_csv(file_path)

threshold = 5000
data['High_Revenue'] = (data['Revenue'] > threshold).astype(int)

segmentation_features = ['Customer_Age', 'Order_Quantity', 'Profit']
kmeans = KMeans(n_clusters=4, random_state=42)
data['Segment_Name'] = kmeans.fit_predict(data[segmentation_features])

features = ['Customer_Age', 'Order_Quantity',
            'Profit', 'Product_Category', 'Segment_Name']
data_encoded = pd.get_dummies(data[features], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_encoded)

target = data['High_Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, target, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

coefficients = pd.DataFrame({
    'Feature': data_encoded.columns,
    'Importance': abs(log_reg.coef_[0])
})

coefficients = coefficients.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=coefficients)
plt.title('Feature Importance in Predicting High Revenue Customers')
plt.show()
