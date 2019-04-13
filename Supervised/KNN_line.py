# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 06:09:33 2019

@author: Luis Galaviz
"""
import matplotlib.pyplot as plt
import numpy as np
from . import KNN


np.random.seed(20)
X = np.arange(0, 5, 1)
X_test = np.array([0.5])
tolerance = 0.5
X_all = np.linspace(np.min(X) - tolerance, np.max(X) + tolerance, num = 1000)

N = len(X)
f = lambda X: 2*X + 1
y_real = f(X)
y = y_real + np.random.normal(0, 2, size= N)

plt.figure()
plt.grid(True)
plt.plot(X_all, f(X_all), color = 'red', linestyle = ':', alpha = 1)

model = KNN(2)
model.fit(X, y)
plt.plot(X_all, model.predict(X_all), color = 'green')
plt.scatter(X, y, color = 'blue')
plt.scatter(X_test, model.predict(X_test), color = 'orange')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = 2x + 1')
plt.legend(['f(x)', 'f_pred(x) with k = 2', 'f(x) + N(0, 2)', 'x_i'])