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

np.random.seed(40)

X = np.arange(0, 30)
N = len(X)
f = lambda x: (2*X + 1) + np.random.normal(0, 2, size = N)
f(X)

def problem_1(X, f, model):
    # Problem #1  
    N = len(X)
    y_real = 2*X + 1
    y =  y_real + np.random.normal(0, 2, size = N)
    
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.4, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_all = model.predict(X)
    
    print('Slope= ',model.slope ,'Intercept= ', model.intercept)
    
    plt.figure(1)
    plt.plot(X, y_real, color = 'red', linestyle = ':', alpha = 1)
    plt.plot(X, y_all, color = 'green', alpha = 0.6)
    plt.scatter(X_train, y_train, color = 'blue', marker = '.')
    plt.scatter(X_test, y_test, color = 'orange', marker = 'x')
    lengends = ['f(x) = 2x + 1', 'f_pred(x)', 'Training Set', 'Test Set']
    plt.legend(lengends)
    plt.xlabel('x')
    plt.ylabel('y')

# Problem #1 
X = np.arange(0, 30)
N = len(X)
y_real = 2*X + 1
y =  y_real + np.random.normal(0, 2, size = N)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.4, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_all = model.predict(X)

print('Slope= ',model.slope ,'Intercept= ', model.intercept)

plt.figure(1)
plt.plot(X, y_real, color = 'red', linestyle = ':', alpha = 1)
plt.plot(X, y_all, color = 'green', alpha = 0.6)
plt.scatter(X_train, y_train, color = 'blue', marker = '.')
plt.scatter(X_test, y_test, color = 'orange', marker = 'x')
lengends = ['f(x) = 2x + 1', 'f_pred(x)', 'Training Set', 'Test Set']
plt.legend(lengends)
plt.xlabel('x')
plt.ylabel('y')


# Problem #2
X = np.linspace(0, 2*np.pi, num = 40)
N = len(X)
y_real = np.sin(X)
y =  y_real + np.random.normal(0, 0.1, size = N)


model = LinearRegression()
model.fit(X, y)
y_hat = model.predict(X)
print('Slope= ',model.slope ,' Intercept= ', model.intercept)

plt.figure(2)

plt.plot(X, y_real, color = 'red', linestyle = ':', alpha = 1)
plt.plot(X, y_hat, color = 'green', alpha = 0.6)
plt.scatter(X, y)
lengends = ['f(x) = Sin(x)', 'f_pred(x)', 'f(x) = Sin(x) + N(0, 0.1)' ]
plt.legend(lengends)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

