# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 04:25:10 2019

@author: Luis Galaviz
"""

from sklearn.datasets import load_boston
import pandas as pd
import seaborn as sns
import numpy as np

dataset = load_boston()
df = pd.DataFrame(dataset.data, columns =dataset.feature_names)
cov = np.cov(df.values.T)

selected_features = ['LSTAT']
df = df[selected_features]

df['LSTAT2'] = df['LSTAT'].apply(np.square)
df['target'] = dataset.target
df['target_log'] = df['target'].apply(np.log)
#df = df.sorted(selected_features)


means = df.apply(np.mean, axis = 0)
corr = df.corr()



sns.pairplot(df)