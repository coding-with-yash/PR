import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset (replace 'path_to_dataset' with the actual path)
url = r"weatherHistory.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

# Select relevant columns (humidity and apparent temperature)
data = df[['Humidity', 'Apparent Temperature (C)']]

# Drop rows with missing values
data = data.dropna()

# Split the dataset into features (X) and target variable (y)
X = data[['Humidity']]
y = data['Apparent Temperature (C)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the performance of the regression model
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-Square: {r2:.4f}")

# Visualize the simple linear regression model
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Linear Regression Model')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature (C)')
plt.title('Simple Linear Regression Model')
plt.legend()
plt.show()

y = a + bx
b= (n∑xy) − (∑x)(∑y) / (n∑x2) − (∑x)2
a= ∑y − b(∑x) / n