import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()

# Create a DataFrame from the iris dataset
data = pd.DataFrame({
    'sepallength': iris.data[:, 0],
    'sepalwidth': iris.data[:, 1],
    'petallength': iris.data[:, 2],
    'petalwidth': iris.data[:, 3],
    'variety': iris.target
})

# Split the dataset into features (X) and the target variable (y)
X = data.drop('variety', axis=1)
y = data['variety']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Perform predictions on the test dataset
y_pred = clf.predict(X_test)

# Calculate accuracy using metrics module
accuracy = metrics.accuracy_score(y_test, y_pred)
print("ACCURACY OF THE MODEL:", accuracy)

# Predicting the variety of a flower with specific features
prediction = clf.predict([[3, 3, 2, 2]])
print("Predicted Variety:", prediction)

# Using the feature importance variable
feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importance:")
print(feature_imp)

# Visualize feature importance with a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Importance")
plt.show()
