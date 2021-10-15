
# coding: utf-8

# In[124]:

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

CLUSTER_NUM = 15

#please update this to the location of the file on your device
filename = '/home/varadars/Research Mentorship (Cyclones)/cyclogenesis_init_WP_wpsh_enso.csv'
cyclone_data = pd.read_csv(filename)

lat_long_data = cyclone_data[['LAT', 'LON']]
lat_long_data.head()


# In[125]:

#Running k-means on the above-defined set of clusters
#printing the values for the cluster centers
kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=0).fit(lat_long_data)
print(kmeans.cluster_centers_)


# In[126]:

#predict the labels of clusters
label = kmeans.fit_predict(lat_long_data)
print(label)


# In[127]:

#displaying all the clusters as well as the respective cluster centers
import matplotlib.pyplot as plt

lat_long_data.plot.scatter('LAT', 'LON',c=label,colormap='viridis')

x = kmeans.cluster_centers_[:,0]
y = kmeans.cluster_centers_[:,1]
plt.scatter(x, y, color='red')


# In[128]:

#adding the cluster values to the dataframe
#ignore any errors that this produces
lat_long_data.loc[:, 'clusters'] = label
lat_long_data


# In[130]:

#prints the head of each cluster
#prints the cluster map with only the values from each cluster
for number in range(CLUSTER_NUM):
    first_cluster = lat_long_data[lat_long_data["clusters"] == number]
    print("Cluster")
    print(number + 1)
    print(first_cluster.head())
    first_cluster.plot.scatter('LAT', 'LON', c='clusters', colormap='viridis')


# In[ ]:



