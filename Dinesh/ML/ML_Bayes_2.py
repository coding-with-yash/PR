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







#2nd code
import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/Admin/Desktop/ML/Final/2.Naive bayes/Iris.csv")

x=data.iloc[:,:-1].values
print(x)

y=data.iloc[:,-1].values
print(y)

from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
print(x_train)
print(y_train)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)

#Prediction using Baye
y_predict=gnb.predict(x_test)
print("Predicted Value by model ",y_predict)
print("Actual value in dataset  ",y_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,y_predict)
print("confusion matrix \n",confusion_mat)

#Classifier Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)*100
print("Accuracy of Gaussian Model",accuracy)

'''input=[5.1, 3.5 ,1.4, 0.2]
output=gnb.predict(input)
print("predicted output\n",output)'''




# Naive Bayes classification is a machine learning algorithm based on Bayes' theorem, which is a probability theory developed by Reverend Thomas Bayes. The "naive" in Naive Bayes comes from the assumption that features used to describe an observation are conditionally independent, given the class label. This assumption simplifies the computation and makes the algorithm computationally efficient.
# 
# Here's a step-by-step explanation of how Naive Bayes classification works:
# 
# Bayes' Theorem:
# The algorithm is based on Bayes' theorem, which calculates the probability of a hypothesis given the observed evidence. The formula for Bayes' theorem is as follows:
# P(A∣B)= P(B∣A)⋅P(A)/P(B)
# 
# P(A∣B) is the probability of event A given that event B has occurred.
# 
# P(B∣A) is the probability of event B given that event A has occurred.
# 
# P(A) and P(B) are the probabilities of events A and B occurring, respectively.
# 

# Naive Bayes classification has found application in various domains, particularly in natural language processing and text classification due to its simplicity and efficiency. Here are some common applications:
# 
# 1. **Spam Email Filtering:**
#    Naive Bayes is widely used for spam email filtering. By analyzing the content and features of emails, the algorithm can classify them as either spam or non-spam based on the likelihood of certain words or patterns occurring in spam emails.
# 
# 2. **Sentiment Analysis:**
#    In sentiment analysis, Naive Bayes can be used to determine the sentiment of a piece of text, such as a review or a social media post. It classifies the text as positive, negative, or neutral based on the occurrence of certain words or phrases associated with sentiment.
# 
# 3. **Document Classification:**
#    Naive Bayes is employed for categorizing documents into predefined categories. For instance, news articles can be classified into topics like sports, politics, or entertainment based on the words present in the articles.
# 
# 4. **Medical Diagnosis:**
#    In the medical field, Naive Bayes has been used for diagnostic purposes. By considering various symptoms and test results as features, the algorithm can help in predicting the likelihood of a patient having a particular medical condition.
# 
# 5. **Credit Scoring:**
#    Naive Bayes can be applied to credit scoring, where it helps evaluate the creditworthiness of individuals by considering various financial and personal features. This aids in deciding whether to approve or deny a loan application.
# 
# 6. **Fraud Detection:**
#    Naive Bayes is used in fraud detection systems to identify potentially fraudulent activities. By analyzing patterns and features in transactions, the algorithm can flag or block transactions that are likely to be fraudulent.
# 
# 7. **Authorship Attribution:**
#    Naive Bayes can be applied to determine the likely authorship of a document based on writing style and word choices. This is useful in forensic linguistics and literary studies.
# 
# 8. **Customer Support Ticket Classification:**
#    In customer support systems, Naive Bayes can help classify incoming support tickets into predefined categories, allowing for more efficient routing and resolution of customer issues.
# 
# 9. **Weather Prediction:**
#    In meteorology, Naive Bayes has been used for weather prediction by considering various atmospheric features. It helps in classifying weather conditions, such as predicting whether it will rain or not.
# 
# While Naive Bayes has its limitations, such as the assumption of feature independence, it remains a popular choice for certain applications, especially where interpretability and simplicity are important considerations.