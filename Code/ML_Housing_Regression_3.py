import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("/content/Housing.csv")  # Replace with your dataset's file path

data=data.dropna()

X = data[['area']]  # Feature(s)
y = data['price']   # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)
print(len(X_train))

reg = LinearRegression()
reg.fit(X_train, y_train)
y_predicted = reg.predict(X_test)

print("intercept \n",reg.intercept_)
print("slope \n",reg.coef_)

mse = mean_squared_error(y_test, y_predicted)
mae=mean_absolute_error(y_test,y_predicted)
r2=r2_score(y_test,y_predicted)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-square:",r2)

#print("predicted data by model \n",y_pred)
print("Actual data \n",y_test)

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_predicted, color='red')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Housing Price Prediction')
plt.show()

new_area = [[2000]]  # Replace with the new 'area' value you want to predict
predicted_price = reg.predict(new_area)
print("Predicted Price for New Area:", predicted_price)

y = a + bx
b= (n∑xy) − (∑x)(∑y) / (n∑x2) − (∑x)2
a= ∑y − b(∑x) / n