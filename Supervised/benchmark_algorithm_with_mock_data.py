import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from KNN import KNN
from LinearRegression import LinearRegression

def benchmark(X, f, error, model, fig_num = 1, tolerance = 0):
    X_all = np.linspace(np.min(X)- tolerance, np.max(X) + tolerance, num = 1000)

    # X_all and y_all are used to plot the non-linear model 
    # with a better resolution
    y_real = f(X_all)
    
    
    # We calculate observated noisy data
    y =  f(X) + error
    
    # We separate tests sets to avoid memorization
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.4, random_state=42)
    model.fit(X_train, y_train)
    
    # We predict over X_all the range of values to visualize better
    y_all = model.predict(X_all)
    
    # Plot results
    plt.figure(fig_num)
    plt.plot(X_all, y_real, color = 'red', linestyle = ':', alpha = 1)
    plt.plot(X_all, y_all, color = 'green', alpha = 0.6)
    plt.scatter(X_train, y_train, color = 'blue', marker = '.')
    plt.scatter(X_test, y_test, color = 'orange', marker = 'x')
    plt.legend(['f(x)', 'f_pred(x)', 'Training Set', 'Test Set'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

np.random.seed(40)

# Trying a noisy Linear function
X_line = np.arange(0, 30)
N_line = len(X_line)
f_line = lambda x: (2*x + 1) 
tolerance_line = 5
lreg = LinearRegression()
error_line = np.random.normal(0, 2, size = N_line)
benchmark(X_line, f_line, error_line, LinearRegression(), tolerance = tolerance_line)
benchmark(X_line, f_line, error_line, KNN(2), fig_num = 2, tolerance = tolerance_line)


# Trying a noisy Sine function
X_sin = np.linspace(0, 2*np.pi, num = 40)
N_sin = len(X_sin)
f_sin = lambda x: np.sin(x)
tolerance_sin = np.pi / 3
error_sin = np.random.normal(0, 0.1, size = N_sin)
benchmark(X_sin, f_sin, error_sin, LinearRegression(), fig_num = 3, tolerance = tolerance_sin)
benchmark(X_sin, f_sin, error_sin, KNN(2), fig_num = 4, tolerance = tolerance_sin)

