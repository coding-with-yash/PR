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


#Theory :- 



# Confusion Matrix is a table that is often used to describe the performance of a classification model on a set of data for which the true values are known. It helps in understanding the performance of a classification algorithm by breaking down the predictions into four categories: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

# Here's a breakdown of each component of the confusion matrix:

# True Positive (TP):

# Instances that are actually positive and are correctly predicted as positive by the model.
# True Negative (TN):

# Instances that are actually negative and are correctly predicted as negative by the model.
# False Positive (FP) - Type I Error:

# Instances that are actually negative but are incorrectly predicted as positive by the model.
# False Negative (FN) - Type II Error:

# Instances that are actually positive but are incorrectly predicted as negative by the model.
# The confusion matrix is usually presented in the following format:

# mathematica
# Copy code
#               | Predicted Positive | Predicted Negative |
# Actual Positive |        TP          |        FN          |
# Actual Negative |        FP          |        TN          |
# Here's how precision, recall, and F1-score are related to the confusion matrix:

# Precision:

# Precision is the ratio of TP to the total predicted positives. It is a measure of the accuracy of positive predictions.
# Formula: Precision=TP/TP+FP
 
# Recall (Sensitivity or True Positive Rate):
# Recall is the ratio of TP to the total actual positives. It measures the ability of the model to capture all the positive instances.
# Formula: Recall=TP/TP+FN

# F1-Score:

# The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall.
# Formula: F1-Score =2×Precision×Recall/Precision+Recall
 
# These metrics can be calculated directly from the values in the confusion matrix. Understanding the confusion matrix and associated metrics helps in assessing the strengths and weaknesses of a classification model and can guide model improvement efforts.

