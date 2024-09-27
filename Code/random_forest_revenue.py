import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/mikhailfernandez/Desktop/Personal Projects/Customer Segmentation and revenue forecasting/data/Sales.csv'
data = pd.read_csv(file_path)

print("Available columns in the dataset:", data.columns)

segmentation_features = ['Customer_Age', 'Order_Quantity', 'Profit']
kmeans = KMeans(n_clusters=4, random_state=42)
data['Segment_Name'] = kmeans.fit_predict(data[segmentation_features])

required_columns = ['Customer_Age', 'Order_Quantity',
                    'Product_Category', 'Segment_Name']

missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    print("All required columns are present.")

if not missing_columns:
    data_encoded = pd.get_dummies(data[required_columns], drop_first=True)

    target = data['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(
        data_encoded, target, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    feature_importance = rf.feature_importances_
    features_list = data_encoded.columns

    importance_df = pd.DataFrame(
        {'Feature': features_list, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance in Driving Revenue (Random Forest)')
    plt.show()
