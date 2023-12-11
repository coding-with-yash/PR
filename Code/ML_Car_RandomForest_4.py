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
