# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:50:23 2019

@author: Luis Galaviz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from KNN import KNN

class LinearRegression():
    def __init__(self):
        self.X = None
        self.y = None
        self.slope = None
        self.intercept = None
    
    def fit(self, X, y):      
        x_mean = X.mean()
        y_mean = y.mean()
        xy_mean = np.dot(X,y) / len(X)
        x_sqr_mean = np.dot(X, X) / len(X)
        x_mean_sqr = np.power(x_mean, 2)
        
        #print(x_mean, x_sqr_mean, x_mean_sqr, y_mean, xy_mean,)
        
        self.slope = (xy_mean - x_mean * y_mean) / (x_sqr_mean - x_mean_sqr)
        self.intercept = (x_sqr_mean * y_mean - x_mean * xy_mean) / (x_sqr_mean - x_mean_sqr)
        
        y_hat = self.slope * X + self.intercept
        #plt.plot(X, y_hat)
        
        diff1 = y - y_hat
        
        diff2 = y - y_mean
        
        #residuals_df = pd.DataFrame()
        
        #sns.swarmplot(data=[diff1])
        #sns.swarmplot(data=[diff2], color='red')
        RSS = diff1.dot(diff1)
        RST = diff2.dot(diff2)
        #print(RSS, RST)
        R = 1 - RSS/RST
        #print("R value is ", R)
        
        #print(self.slope, self.intercept)
        
        #denominator = X.dot(X) - X.mean() * X.sum()
        #print(denominator, x_sqr_mean - x_mean_sqr)
        #a = ( X.dot(y) - y.mean()*X.sum() ) / denominator
        #b = ( y.mean() * X.dot(X) - X.mean() * X.dot(y) ) / denominator
        #print(a, b)
        # let's calculate the predicted Y
        #Yhat = a*X + b
        

    def predict(self, X):
        return self.slope * X + self.intercept

