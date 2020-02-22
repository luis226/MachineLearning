# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:07:17 2020

@author: Luis Galaviz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from data import get_normal_clouds, get_donut
from sortedcontainers import SortedList
from future.utils import iteritems

# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i,x in enumerate(X): # test points
            sl = SortedList() # stores (distance, class) tuples
            for j,xt in enumerate(self.X): # training points
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    # don't need to check, just add
                    sl.add( (d, self.y[j]) )
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add( (d, self.y[j]) )
            # print "input:", x
            # print "sl:", sl

            # vote
            votes = {}
            for _, v in sl:
                # print "v:", v
                votes[v] = votes.get(v,0) + 1
            # print "votes:", votes, "true:", Ytest[i]
            
            #max_votes = 0
            #max_votes_class = -1
            #for v,count in iteritems(votes):
                #if count > max_votes:
                #    max_votes = count
                #    max_votes_class = v
            #y[i] = max_votes_class
            
            votes_1 = votes.get(1, 0)
            y[i] = votes_1 / self.k
            
        return y
    
np.random.seed(40)
X, Y = get_normal_clouds()
# X, Y = get_donut()
X = X[: ,1:] 
k = 20
model = KNN(k = k)
model.fit(X, Y)


plt.scatter(X[:50, 0], X[:50, 1], c='green', marker= 's')
plt.scatter(X[50:, 0], X[50:, 1], c='red', marker= 'v')


xx, yy = np.meshgrid(np.arange(start = -10, stop = 10, step = 0.2),
                     np.arange(start = -10, stop = 10, step = 0.2))
x_grid = np.vstack((xx.ravel(), yy.ravel())).T
y_pred = model.predict(x_grid)

# Plotting inputs
plt.contourf(xx, yy, y_pred.reshape(xx.shape),
             alpha = 0.2, cmap = ListedColormap(('green', 'red')))
plt.xlim(0, 8)
plt.ylim(0, 8)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title(f'Binary Classification k={k}')
plt.legend(['Class 0', 'Class 1'])
plt.show()



