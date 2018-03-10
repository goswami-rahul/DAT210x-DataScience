#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 18:23:58 2018

@author: deadpool
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

for num in Inlist:
    user = df[df.In == num]
    user = user[(user.DOW == 'Sat') | (user.DOW == 'Sun')]
    user = user[(user.CallTime < '06:00:00') | (user.CallTime > '22:00:00')]
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(user["TowerLat TowerLon".split()])
    coord = kmeans.cluster_centers_
    user_coord.extend(coord)