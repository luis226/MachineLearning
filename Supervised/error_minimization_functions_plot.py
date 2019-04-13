import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from error_functions import *

# Print 3d Surfaces dependencys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Simulated Data parameters
real_slope = 1
real_intercept = 0
N = 10
mean = 0
var = 0.5

# Generating simulated data f(x) = ax + b + N(mean, var)
X =  np.arange(0, 10)
y_real = X * real_slope + real_intercept 
y = y_real + np.random.normal(mean, var, size = N)

# Reshape vectors to ensure operations consistency
y_real = y_real.reshape((N, 1))
y = y.reshape((N, 1))

# Adds a ones row to be able to write this on matrix notation
X = np.vstack((np.ones(shape = (N,)), X))
X_t = X.T

N = 100
a_intervals = np.linspace(-10, 10, num = N)
b_intervals = np.linspace(-10, 10, num = N)


aa, bb = np.meshgrid(a_intervals, b_intervals)
#aa, bb = np.mgrid[-10:10:100j, -10:10:100j]
ab = np.vstack((aa.flatten(), bb.flatten()))
ab_t = ab.T

y_hats = np.matmul(X_t, ab)
y_diffs = y - y_hats 


abs_err_matrix = np.sum(np.abs(y_diffs), axis = 0).reshape((N, N))
abs_err_matrix = abs_err_matrix / abs_err_matrix.max()

# Calculates and normalizes Squared Error
sqr_err_matrix = np.matmul(y_diffs.T, y_diffs) 
se = np.diag(sqr_err_matrix).reshape((N, N))
se = se / se.max()



fig, axs = plt.subplots(1, 2, constrained_layout=True)
# MAE Contour
levels = [0, 0.05 , 0.1, 0.3,  0.5, 1]
contour = axs[0].contour(aa, bb, abs_err_matrix, levels, colors='k')
axs[0].clabel(contour, colors = 'k', fmt = '%3.2f', fontsize=12)
contour_filled = axs[0].contourf(aa, bb, abs_err_matrix, levels, cmap="RdBu_r")
#axs[0].set_colorbar(contour_filled)
axs[0].set_title('MAE Error')
axs[0].set_xlabel('a (intercept)')
axs[0].set_ylabel('b (slope)')
#plt.show()

# MSE Contour
#plt.figure(2)
contour = axs[1].contour(aa, bb, se, levels, colors='k')
axs[1].clabel(contour, colors = 'k', fmt = '%3.2f', fontsize=12)

axs[1].set_title('MSE Error')
axs[1].set_xlabel('a (intercept)')
axs[1].set_ylabel('b (slope)')

contour_filled = axs[1].contourf(aa, bb, se, levels, cmap="RdBu_r")
fig.colorbar(contour_filled)
plt.show()


#Plot MAE Surface
fig = plt.figure(3)
ax = fig.gca(projection='3d')
fig.suptitle('MAE vs Slope,Intercept', fontsize=16)
surf = ax.plot_surface(aa, bb, abs_err_matrix, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('MAE')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# Print MSE surface
fig = plt.figure(4)
fig.suptitle('MSR vs Slope,Intercept', fontsize=16)
ax = fig.gca(projection='3d')

surf = ax.plot_surface(aa, bb, se, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('MSE')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()