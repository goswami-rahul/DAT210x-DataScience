#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 00:02:32 2018

@author: deadpool
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def clusterInfo(model):
    print("Cluster Analysis Inertia: ", model.inertia_)
    print('------------------------------------------')
    
    for i in range(len(model.cluster_centers_)):
        print("\n  Cluster ", i)
        print("    Centroid ", model.cluster_centers_[i])
        print("    #Samples ", (model.labels_==i).sum()) # NumPy Power
              
# Find the cluster with the least # attached nodes
def clusterWithFewestSamples(model):
    # Ensure there's at least on cluster...
    minSamples = len(model.labels_)
    minCluster = 0
    
    for i in range(len(model.cluster_centers_)):
        if minSamples > (model.labels_==i).sum():
            minCluster = i
            minSamples = (model.labels_==i).sum()

    print("\n  Cluster With Fewest Samples: ", minCluster)
    return (model.labels_==minCluster)

def get_unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

df = pd.read_csv('Datasets/CDR.csv')
df['CallDate'] = pd.to_datetime(df.CallDate, errors='coerce')
df['CallTime'] = pd.to_timedelta(df.CallTime, errors='coerce')
df['Duration'] = pd.to_timedelta(df.Duration, errors='coerce')

Inlist = get_unique(df.In)
user_coord = []
user_work_coord = []

for num in Inlist:
    user = df[df.In == num]
    user = user[~((user.DOW == 'Sat') | (user.DOW == 'Sun'))]
    user = user[user.CallTime < '17:00:00']
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(user["TowerLat TowerLon".split()])
    coord = kmeans.cluster_centers_
    user_coord.append(coord[1])
    user_work_coord.append(coord[0])
    
    user.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')
    plt.show()
    
    midWayClusterIndices = clusterWithFewestSamples(kmeans)
    midWaySamples = user[midWayClusterIndices]
    print("    Its Waypoint Time: ", midWaySamples.CallTime.mean())
    
    plt.scatter(kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,0], s=169, c='r', marker='x', alpha=0.8, linewidths=2)
    plt.title('Weekday Calls Centroids')
    plt.show()