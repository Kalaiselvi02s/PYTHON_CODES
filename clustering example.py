#!/usr/bin/env python
# coding: utf-8

# # Market segmentation example

# ## Import the relevant libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# ## Load the data

# In[3]:


data = pd.read_csv ('3.12.+Example.csv')


# In[4]:


data


# ## Plot the data

# In[5]:


plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')


# ## Select the features

# In[6]:


x = data.copy()


# ## Clustering

# In[7]:


kmeans = KMeans(2)
kmeans.fit(x)


# ## Clustering results

# In[8]:


clusters = x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)


# In[9]:


plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')


# ## Standardize the variables

# In[10]:


from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
x_scaled


# ## Take advantage of the Elbow method

# In[11]:


wcss =[]

for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
    
wcss


# In[12]:


plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# ###### Using elbow 2,3,4,5 are best choice of number of cluster in these 4 is optimal

# In[15]:


kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)


# In[16]:


clusters_new


# In[17]:


plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

