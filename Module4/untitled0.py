#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:07:28 2018

@author: deadpool
"""

import pandas as pd
from sklearn.manifold import Isomap

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')

folder = 'Datasets/ALOI/32'
samples = []
colors = []
for imgname in os.listdir(folder):
    img = misc.imread(os.path.join(folder, imgname))
    samples.append((img/255.0).reshape(-1))
    colors.append('b')

folder += 'i'
for imgname in os.listdir(folder):
    img = misc.imread(os.path.join(folder, imgname))
    samples.append((img/255.0).reshape(-1))
    colors.append('r')

df = pd.DataFrame(samples)

iso = Isomap(n_components=3, n_neighbors=6)
iso.fit(df)
T = iso.transform(df)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(T[:, 0], T[:, 1], c=colors)
plt.show()

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

ax.set_title('...')
ax.set_xlabel('component 0')
ax.set_ylabel('component 1')
ax.set_zlabel('component 2')
ax.scatter(T[:, 0], T[:, 1], T[:, 2], c=colors, marker='.', alpha=0.75)
plt.show()