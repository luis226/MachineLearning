# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:05:29 2020

@author: Luis Galaviz
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
def gradient_descent(f, df, x_limits, learning_rate = 0.05):
    opt_x = np.random.uniform(x_limits[0], x_limits[1])
    optimization_curve = np.ndarray(shape=(100, 2))
    for i in range(100):
        opt_x -= df(opt_x) * learning_rate
        optimization_curve[i] = [i, f(opt_x)]
    plt.plot(optimization_curve[:,0], optimization_curve[:,1])
    plt.title('f(x) Optimization')
    plt.xlabel('Iterations')
    plt.ylabel('f(x)')
    print(optimization_curve)

f = lambda X:  (X + 1)*(X + 2)
df = lambda X: 2*X + 3

gradient_descent(f, df, (-20, 20), learning_rate = 0.05)

X = np.linspace(-5, 5, num=200)

plt.plot(X, f(X))