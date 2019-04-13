import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_real_and_pred_function(X_t, y_real, y_hat):
    # Plots funtion with and without noise
    plt.figure(0)
    plt.plot(X_t[:, 1], y_real, color = 'red', alpha = 0.7, linestyle = '--')
    plt.plot(X_t[:, 1], y_hat, color = 'green', alpha = 0.7)
    plt.scatter(X_t[:, 1], y, s = 8)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Comparision between predicted and real f(x)')
    plt.legend(["f(x)", 'f_pred(x)', "f(x) + N(mean, var)"])
    plt.show()

np.random.seed(20)

# Data parameters
real_slope = 1
real_intercept = 0
a = 0
b = 10
N = 10
mean = 0
var = 0.5

# Generating simulated data f(x) = ax + b + N(mean, var)
X =  np.linspace(a, b, num = N)
y_real = X * real_slope + real_intercept 
y = y_real + np.random.normal(mean, var, size = N)

# Reshape vectors to ensure operations consistency
y_real = y_real.reshape((N, 1))
y = y.reshape((N, 1))

# Adds a ones row to be able to write this on matrix notation
X = np.vstack((np.ones(shape = (N,)), X))
X_t = X.T

# Creating a vector to write the line model as matrix multiplication
intercept_pred = 0.8
slope_pred = 0.9
w_t = np.array([[intercept_pred, slope_pred]])
w = w_t.T

# Predicted values are calculated by w_T * X
y_hat = np.matmul(w_t, X).T


plot_real_and_pred_function(X_t, y_real, y_hat)



# Root mean squared error calculation
y_diff = y - y_hat
sqr_err = np.matmul(y_diff.T, y_diff) 
mean_sqr_err = sqr_err / len(y)
mean_root_sqr_err = np.sqrt(sqr_err / len(y)) 

abs_err = np.abs(y_diff)
sqr = np.square(y_diff)
table_err = np.concatenate((X_t[:, 1].reshape(N, 1), y, y_hat,
                            y_diff, sqr ,abs_err), 
                           axis = 1)
df_err = pd.DataFrame(table_err, columns = ['X', 'f(x)', 'f_pred(x)', 'diff', 'SE', 'AE'])

print("MSE: ", mean_sqr_err[0, 0])
print("RMSE: ", mean_root_sqr_err[0, 0])

#a_intervals = np.linspace(-1, 2, num = 200)
#b_intervals = np.linspace(1, 3, num = 200)

N = 100
#aa, bb = np.meshgrid(a_intervals, b_intervals)
aa, bb = np.mgrid[-10:10:100j, -10:10:100j]
ab = np.vstack((aa.flatten(), bb.flatten()))
ab_t = ab.T

y_hats = np.matmul(X_t, ab)
y_diffs = y - y_hats 


abs_err_matrix = np.sum(np.abs(y_diffs), axis = 0).reshape((N, N))
abs_err_matrix = abs_err_matrix / abs_err_matrix.max()

# Calculates and normalizes Squared Error
sqr_err_matrix = np.matmul(y_diffs.T, y_diffs) 
rmse = np.diag(sqr_err_matrix).reshape((N, N))
rmse = rmse / rmse.max()


import seaborn as sns


#
plt.figure(1)
levels = [0, 0.05 , 0.1, 0.3,  0.5, 1]
contour = plt.contour(aa, bb, abs_err_matrix, levels, colors='k')
plt.clabel(contour, colors = 'k', fmt = '%3.2f', fontsize=12)
contour_filled = plt.contourf(aa, bb, abs_err_matrix, levels, cmap="RdBu_r")
plt.colorbar(contour_filled)
plt.title('MAE Error')
plt.xlabel('a (intercept)')
plt.ylabel('b (slope)')
plt.show()


plt.figure(2)
levels = [0, 0.05 , 0.1, 0.3,  0.5, 1]
contour = plt.contour(aa, bb, rmse, levels, colors='k')
plt.clabel(contour, colors = 'k', fmt = '%3.2f', fontsize=12)
contour_filled = plt.contourf(aa, bb, rmse, levels, cmap="RdBu_r")
plt.colorbar(contour_filled)
plt.title('RMSE Error')
plt.xlabel('a (intercept)')
plt.ylabel('b (slope)')
plt.show()


'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
#
#
#plt.clf()
fig = plt.figure(3)
ax = fig.gca(projection='3d')
#
## Plot the surface.
surf = ax.plot_surface(aa, bb, abs_err_matrix, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

## Customize the z axis.
ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
## Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
#
plt.show()


#plt.clf()
fig = plt.figure(4)
ax = fig.gca(projection='3d')
#
## Plot the surface.
surf = ax.plot_surface(aa, bb, rmse, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

## Customize the z axis.
ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
## Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
#
plt.show()