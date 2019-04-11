# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:02 2019

@author: Luis Galaviz
"""

def Sum(u):
    result = 0
    for u_i in u:
        result += u_i
    return result

def test_sum_function():
    assert Sum([1, 1, 1, 1]) == 4, "Sum of [1, 1, 1, 1] should be 4"
    assert Sum([]) == 0, "Sum of [] should be 0"
    assert Sum([2, -1, 3, 5, 0]) == 9, "Sum of [] should be 9"
    assert Sum([3]) == 3, "Sum of [] should be 13"
    
test_sum_function()


import numpy as np
import time
import pandas as pd


np.random.seed(40)
def compare_with_numpy(intervals):
    
    time_np = np.ndarray(shape = (len(intervals),), dtype = float)
    time_sum = np.ndarray(shape = (len(intervals),), dtype = float)
    
    
    for i, N in enumerate(intervals):
        u = np.random.randint(0, 100, size=(N,))
        
        # Measure the time for numpy
        time1 = time.time()
        np.sum(u)
        time2 = time.time()
        time_np[i] = (time2-time1) * 1000
        
            
        # Measure the time for our Sum method
        time1 = time.time()
        Sum(u)
        time2 = time.time()
        time_sum[i] = (time2-time1) * 1000
    
    return time_np, time_sum

intervals = np.array([10000, 50000, 100000, 200000, 300000, 400000])
t_np, t_sum = compare_with_numpy(intervals)
table = np.vstack( (intervals, t_np, t_sum))
df_time = pd.DataFrame(data=table.T, columns = ['N', 'time_np', 'time_sum'] )
print(df_time) 
    
#time1 = time.time()
#time2 = time.time()
#print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)