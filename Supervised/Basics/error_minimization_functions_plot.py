import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data_1d.csv')

np.random.seed(20)

#def plot_a_mse():
a = 1 
b = 10
N = 10
mean = 0
var = 0.5

X =  np.linspace(a, b, num = N)
y_real = X * a + b 
y = y_real + np.random.normal(mean, var, size = N)
#X, y = get_line_gausserr(1, 10, N = 10, mean = 0, var = 2.5)
    
plt.plot(X, y_real, color = 'red', alpha = 0.7)

plt.scatter(X, y, s = 8)

plt.legend(["f(x)", "f(x + error)"])
#
#plt.xlim(0, 10)
#plt.ylim(10, 20)
#
#plt.show()

X = df.iloc[:, 0].values
X_t = np.array([X]).T
X_t = np.array([[1, 2, 3, 4, 3, 5]]).T


x_0 = np.ones((len(X_t), 1))
X_t = np.concatenate((x_0, X_t), axis = 1)
X = X_t.T

y = df.iloc[:, 1].values
y = np.array([y]).T

y = np.array([[1, 3, 3, 3, 5, 5]]).T


w_t = np.array([[0, 1]])
w = w_t.T
y_hat = np.matmul(w_t, X).T

#print(np.matmul(w_t, X ), np.matmul(w_t, X ).shape)
#print(np.matmul(X_t, w ), np.matmul(X_t, w ).shape)
#
#plt.scatter(X_t[:, 1], y)

#plt.plot(X_t[:, 1], y_hat)
#plt.title("x vs f(X)")
#plt.xlabel("x")
#plt.ylabel("f(x)")
#plt.show()



y_diff = y - y_hat
sqr_err = np.matmul(y_diff.T, y_diff) 
mean_sqr_err = sqr_err / len(y)
mean_root_sqr_err = np.sqrt(sqr_err) / len(y)

print("MSE: ", mean_sqr_err[0, 0])
print("MRSE: ", mean_root_sqr_err[0, 0])

#a_intervals = np.linspace(-1, 2, num = 200)
#b_intervals = np.linspace(1, 3, num = 200)

N = 100
#aa, bb = np.meshgrid(a_intervals, b_intervals)
aa, bb = np.mgrid[-10:10:100j, -10:10:100j]



#xy = np.array([[0, 1]
#            , [1, 0.5]
#            , [1, 1]])
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



#import numpy as np
#import matplotlib.pyplot as plt
#xlist = np.linspace(-3.0, 3.0, 100)
#ylist = np.linspace(-3.0, 3.0, 100)

#plt.figure()
##levels = [0.0, 0.025, 0.05, 0.075, 0.1, 0.2]
##cp = plt.contour(aa, bb, rmse, levels = levels, colors='k')
##plt.clabel(cp, colors = 'k', fmt = '%2.1f', fontsize=12)
#contour_filled = plt.contourf(aa, bb, rmse)
#plt.colorbar(contour_filled)
#plt.title('Root mean squared error')
#plt.xlabel('w_0')
#plt.ylabel('w_1')
#plt.show()

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
##import numpy as np
#import matplotlib.pyplot as plt
#xlist = np.linspace(-3.0, 3.0, 100)
#ylist = np.linspace(-3.0, 3.0, 100)
#X, Y = np.meshgrid(xlist, ylist)
#Z = np.sqrt(X**2 + Y**2)
#plt.show()

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