# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:30:53 2019

@author: Luis Galaviz
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

calculated_value = 100.0
red = np.array([104, 103, 97, 103, 97, 102, 98, 98, 100, 101])
yellow = np.array([100, 106, 95, 104, 100, 94, 101, 100, 99, 100])

red_diff = red - calculated_value
red_sqr_err = np.power(red_diff, 2)
red_abs_err = np.abs(red_diff)
print("Red Squared Error: ", np.sum(red_sqr_err), 
      "Red Absolute Error: ", np.sum(red_abs_err))


yellow_diff = yellow - calculated_value
yellow_sqr_err = np.power(yellow_diff, 2)
yellow_abs_err = np.abs(np.abs(yellow_diff))
print("Yellow Squared Error: ", np.sum(yellow_sqr_err), 
      "Yellow Absolute Error: ", np.sum(yellow_abs_err))

table = np.vstack((red, red_sqr_err, red_abs_err, 
                   yellow, yellow_sqr_err, yellow_abs_err)).T
df_err = pd.DataFrame(table, columns = ['Red','Red_SqrErr', 'Red_AbsErr',
                                        'Yellow', 'Yel_SqrErr', 'Yel_AbsErr'])

print(df_err)

swarm_red = np.vstack((np.full((10,), 1), red)).T
swarm_yel = np.vstack((np.full((10,), 2), yellow)).T
swarm_data = np.concatenate((swarm_red, swarm_yel))

df_swarm = pd.DataFrame(data = swarm_data, columns = ['color_code', 'Volts'])
df_swarm['Volt_Color'] = df_swarm['color_code'].apply(lambda c: 'Red' if c == 1 else 'Yellow')

_ = sns.swarmplot(x = 'Volts', y = 'Volt_Color', data = df_swarm, 
              orient = 'h',  size = 8.0)
_.set_title('Volts Measured by each Voltmeter')
_.title.set_fontsize(18)

plt.show()