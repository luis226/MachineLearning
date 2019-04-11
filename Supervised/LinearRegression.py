# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:50:23 2019

@author: Luis Galaviz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegression():
    def __init__(self):
        self.X = None
        self.y = None
        self.a = None
        self.b = None
    
    def fit(self, X, y):      
        x_mean = X.mean()
        y_mean = y.mean()
        xy_mean = np.dot(X,y) / len(X)
        x_sqr_mean = np.dot(X, X) / len(X)
        x_mean_sqr = np.power(x_mean, 2)
        
        #print(x_mean, x_sqr_mean, x_mean_sqr, y_mean, xy_mean,)
        
        self.a = (xy_mean - x_mean * y_mean) / (x_sqr_mean - x_mean_sqr)
        self.b = (x_sqr_mean * y_mean - x_mean * xy_mean) / (x_sqr_mean - x_mean_sqr)
        
        y_hat = self.a * X + self.b
        #plt.plot(X, y_hat)
        
        diff1 = y - y_hat
        
        diff2 = y - y_mean
        
        #residuals_df = pd.DataFrame()
        sns.swarmplot(data=[diff1])
        sns.swarmplot(data=[diff2], color='red')
        RSS = diff1.dot(diff1)
        RST = diff2.dot(diff2)
        print(RSS, RST)
        R = 1 - RSS/RST
        print("R value is ", R)
        
        print(self.a, self.b)
        
        #denominator = X.dot(X) - X.mean() * X.sum()
        #print(denominator, x_sqr_mean - x_mean_sqr)
        #a = ( X.dot(y) - y.mean()*X.sum() ) / denominator
        #b = ( y.mean() * X.dot(X) - X.mean() * X.dot(y) ) / denominator
        #print(a, b)
        # let's calculate the predicted Y
        #Yhat = a*X + b
        

    def predict(self, X):
        x_0 = X[0]
        #y_0 = y[0]

        return self.a * x_0 + self.b

#df = pd.read_csv('data_1d.csv')

#X = df.iloc[:, 0].values
#y = df.iloc[:, 1].values

#model = LinearRegression()
#model.fit(X, y)

#plt.scatter(X, y)

#plt.plot([0, 100], [model.predict([0]),model.predict([100])])

#plt.show()








