# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
data=pd.read_csv("/content/Mall_Customers.csv")
print(data)
x=data.iloc[:,[3,4]].values
print(x)
#Training the K-means algorithm on the training dataset
from sklearn.cluster import KMeans
k=KMeans(n_clusters=5, init='k-means++', random_state= 42)
clusters=k.fit_predict(x)
print("number of clusters are \n",clusters)
#Visualizing the Clusters
mtp.scatter(x[clusters==0,0],x[clusters==0,1],s=100,c='blue',label='Cluster 1')
mtp.scatter(x[clusters==1,0],x[clusters==1,1],s=100,c='green',label='Cluster 2')
mtp.scatter(x[clusters==2,0],x[clusters==2,1],s=100,c='red',label='Cluster 3')
mtp.scatter(x[clusters==3,0],x[clusters==3,1],s=100,c='cyan',label='Cluster 4')
mtp.scatter(x[clusters==4,0],x[clusters==4,1],s=100,c='magenta',label='Cluster 5')
mtp.scatter(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
mtp.title('Clusters of customers')
mtp.xlabel('Annual Income (k$)')
mtp.ylabel('Spending Score (1-100)')
mtp.legend()
mtp.show()

#unsupervised,It allows us to cluster the data into different groups and a convenient way to discover the categories of groups in the unlabeled dataset on its own without the need for any training.



# # **K-means Clustering:**
# 
# K-means clustering is an unsupervised machine learning algorithm used for partitioning a dataset into a set of distinct, non-overlapping groups or clusters. The goal is to group similar data points together while keeping dissimilar points in different clusters. It is a widely used method for exploratory data analysis and is particularly useful in applications such as customer segmentation, image compression, and anomaly detection.
# 
# Here's a step-by-step explanation of the K-means clustering algorithm:
# 
# 1. **Initialization:**
#    - Choose the number of clusters (K) that you want to divide your data into.
#    - Randomly initialize K cluster centroids. These centroids are the initial mean values around which the clusters will form.
# 
# 2. **Assignment:**
#    - For each data point in the dataset, calculate the distance to each centroid.
#    - Assign the data point to the cluster whose centroid is closest (usually using Euclidean distance).
# 
# 3. **Update Centroids:**
#    - Recalculate the mean of the data points within each cluster.
#    - Update the cluster centroids to be the newly calculated means.
# 
# 4. **Repeat:**
#    - Repeat the assignment and update steps iteratively until convergence. Convergence occurs when the assignment of data points to clusters stabilizes, and centroids no longer change significantly.
# 
# 5. **Final Result:**
#    - The final result is a set of K clusters, where each cluster is represented by its centroid.
#    - Each data point belongs to the cluster with the nearest centroid.
# 
# **Key Concepts:**
# 
# - **Centroid:** The center of a cluster, defined as the mean of all data points in the cluster.
# - **Inertia/Within-Cluster Sum of Squares (WCSS):** The sum of squared distances between each data point and its assigned cluster centroid. It is often used to evaluate the quality of the clustering.
# 
# **Advantages:**
# - Simple and easy to implement.
# - Scales well to large datasets.
# 
# **Limitations:**
# - Requires the number of clusters (K) to be specified in advance.
# - Sensitive to the initial placement of centroids, and results may vary with different initializations.
# - Assumes clusters are spherical and equally sized, which may not always be the case.
# 
# K-means clustering is a foundational algorithm in machine learning and is widely used in various fields for grouping and analyzing data.




