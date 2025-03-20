import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

%matplotlib inline

df = pd.read_excel(r"C:\Users\kevin\Downloads\air+quality\AirQualityUCI.xlsx", na_values=-200)

print("Data Preview:")
print(df.head())

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("\nMissing Values Per Column:")
print(df.isnull().sum())

df['DateTime'] = pd.to_datetime(
    df['Date'].astype(str) + ' ' + df['Time'].astype(str),
    format="%Y-%m-%d %H:%M:%S"
)
df.set_index('DateTime', inplace=True)

df.drop(columns=['Date', 'Time'], inplace=True)

print("\nDescriptive Statistics:")
print(df.describe())

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['CO(GT)'], label='CO(GT)')
plt.xlabel('Date')
plt.ylabel('CO (GT)')
plt.title('Time Series of CO(GT)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=df['CO(GT)'])
plt.title('Distribution of CO(GT)')
plt.show()

features = [
    'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 
    'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 
    'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)'
]
target = 'CO(GT)'

model_df = df[features + [target]].dropna()
print("\nShape of data used for modeling:", model_df.shape)

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual CO(GT)')
plt.ylabel('Predicted CO(GT)')
plt.title('Actual vs. Predicted CO(GT)')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\kevin\Downloads\air+quality\AirQualityUCI.xlsx"
df = pd.read_excel(file_path, sheet_name="AirQualityUCI")
print("Data loaded. Shape:", df.shape)

methods = ["Linear Regression", "Forward Selection", "Backward Selection", "Lasso Regression", "Ridge Regression"]
mse = [0.345, 0.340, 0.338, 0.355, 0.350]   # Replace with actual MSE values
r2 = [0.805, 0.810, 0.812, 0.799, 0.802]      # Replace with actual R² values

results_df = pd.DataFrame({
    "Method": methods,
    "MSE": mse,
    "R^2": r2
})

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')

# Adjust layout and save the table as an image
plt.tight_layout()
plt.savefig("regression_results_table.png", bbox_inches='tight', dpi=300)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

file_path = r"C:\Users\kevin\Downloads\air+quality\AirQualityUCI.xlsx"
df = pd.read_excel(file_path, sheet_name="AirQualityUCI")
print("Data loaded. Shape:", df.shape)

df_clean = df.dropna()

features = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
X = df_clean[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method For Optimal k')
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_clean['Cluster'] = clusters

plt.figure(figsize=(8, 5))
scatter = plt.scatter(df_clean['CO(GT)'], df_clean['NO2(GT)'], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('CO(GT)')
plt.ylabel('NO2(GT)')
plt.title('K-Means Clustering (k=3) of Air Quality Data')
plt.colorbar(scatter, label='Cluster')
plt.show()

