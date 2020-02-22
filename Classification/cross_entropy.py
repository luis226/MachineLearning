# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:30:09 2020

@author: Luis Galaviz
"""
import numpy as np
import pandas as pd

def negative_cross_entropy_table(y, y_pred):
    # c value is used to avoid Ln(0) which is undefined
    c = 0.0000000000001
    err = y * np.log(y_pred + c) + (1 - y) * np.log(1 - y_pred + c)
    return err

y = np.array([0, 0, 0, 0, 0,
              1, 1, 1, 1, 1])

y_pred = np.array([0.2, 0, 0.6, 0.3, 0.5,
                   0.8, 0.1, 1, 0.65, 0.3])


ce = np.round(negative_cross_entropy_table(y, y_pred), decimals=4)

table = np.vstack((y, y_pred, ce)).T

df = pd.DataFrame(table, columns=['y','y_pred','l'])
print(df)