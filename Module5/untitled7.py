#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 01:16:47 2018

@author: deadpool
"""

scaler = preprocessing.Normalizer()
d = scaler.fit_transform(df)
d = pd.DataFrame(d, columns=df.columns)
d.describe()
d.plot.hist(alpha=0.8)