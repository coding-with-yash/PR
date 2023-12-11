import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

data = pd.read_csv("/content/sample_data/iris.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = LabelEncoder().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_predict = gnb.predict(x_test)
confusion_mat = confusion_matrix(y_test, y_predict)
print("Confusion matrix \n", confusion_mat)
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_predict_bin = label_binarize(y_predict, classes=np.unique(y))
print(y_test_bin)
print("\n",y_predict_bin)
#
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(np.unique(y))):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_predict_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_predict_bin.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(np.unique(y))):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--', linewidth=4)

# Plot random chance line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

# Add labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

#It is used to determine the conditional probability of event A when event B has already happened.
P(A|B) = P(B|A)P(A) / P(B)
