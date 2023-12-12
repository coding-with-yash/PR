import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import category_encoders as ce

# Load the dataset
df = pd.read_csv(r"car_evaluation.csv")

# Split the dataset into features (X) and the target variable (y)
X = df.drop(['unacc'], axis=1)
y = df['unacc']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Encode categorical variables with ordinal encoding
encoder = ce.OrdinalEncoder()
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Create and fit Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy score: {accuracy:.4f}')

# Visualize feature importances
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)

# Classification report
print('Classification report:\n', classification_report(y_test, y_pred))




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

