#!/usr/bin/env python
# coding: utf-8

# # Market segmentation example

# ## Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# ## Load the data
data = pd.read_csv ('3.12.+Example.csv')
data

# ## Plot the data

plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')


# ## Select the features
x = data.copy()

# ## Clustering
kmeans = KMeans(2)
kmeans.fit(x)

# ## Clustering results
clusters = x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)
plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')


# ## Standardize the variables
from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
x_scaled


# ## Take advantage of the Elbow method
wcss =[]
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
    
wcss
plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# ###### Using elbow 2,3,4,5 are best choice of number of cluster in these 4 is optimal
kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)
clusters_new

plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

