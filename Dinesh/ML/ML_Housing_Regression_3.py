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
          

#Theory :-


# 1. What is Simple Linear Regression?
# 
# **Answer:**
# Simple Linear Regression is a statistical method that allows us to summarize and study the relationship between two continuous variables. It assumes that there is a linear relationship between the independent variable (predictor) and the dependent variable (response). The method aims to find the best-fitting line through the data points that minimizes the sum of squared differences between the observed and predicted values.
# 
# 2. Why is Simple Linear Regression Used?
# 
# **Answer:**
# Simple Linear Regression is used to model the relationship between two variables when it is believed that one variable (independent) influences the other (dependent) in a linear way. It is particularly useful for predicting the value of the dependent variable based on the value of the independent variable.
# 
# 
# 3. **What is the Purpose of Splitting the Data into Training and Testing Sets?**
# 
# **Answer:**
# The purpose of splitting the data into training and testing sets is to assess the performance of the model on new, unseen data. The model is trained on the training set and then evaluated on the testing set to estimate how well it will generalize to new, unseen data. This helps to avoid overfitting, where the model performs well on the training data but fails to generalize to new data.
# 
# 4. **What are Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-Squared (R2) Metrics?**
# 
# **Answer:**
# - Mean Squared Error (MSE): It measures the average squared difference between the actual and predicted values. A lower MSE indicates better model performance.
#     
#           n
# MSE = 1/n ∑   (yi - y^)^2
#           i=1
#   
# - **Mean Absolute Error (MAE):** It measures the average absolute difference between the actual and predicted values. Like MSE, a lower MAE indicates better model performance.
# 
#     
#           n
# MAE = 1/n ∑   |yi - y^|
#           i=1
# 
# 
# - **R-Squared (R2):** It represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). R2 values range from 0 to 1, with 1 indicating a perfect fit.
#         
#           ∑ (yi - y^)^2
# R^2 = 1 - ___________
#           ∑ (yi - y-)^2
#           
#                                  ________
# (Root Mean Square Error RMSE) = / MSE
# 
# 
# 5. **Explain the Steps Involved in Visualizing a Simple Linear Regression Model.**
# 
# Answer:
# 1. Scatter Plot:** Plot the scatter plot of the independent variable (e.g., Humidity) against the dependent variable (e.g., Apparent Temperature) to visualize the data distribution.
# 
# 2. **Regression Line:** Plot the regression line based on the trained model's coefficients. The line represents the best fit through the data points.
# 
# 3. **Labels and Title:** Add labels to the x-axis and y-axis for clarity. Include a title that describes the purpose of the plot, such as "Simple Linear Regression Model."
# 
# 4. **Show the Plot:** Display the plot to visualize how well the regression line fits the data points.
