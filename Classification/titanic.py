# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

d = pd.read_csv('train.csv')

survived = d[d.Survived == 1]['Survived'].count()
not_survived = d[d.Survived == 0]['Survived'].count()
total = survived + not_survived

p_survived = survived / total
p_not_survived = not_survived / total

plt.bar(["Survived", "Not Survived"], [p_survived, p_not_survived], color= ['blue', 'red'])
plt.title("Survived and Not Survived")
plt.ylabel("Probability")
plt.show()

p_male = d[d.Sex == 'male']['Survived'].count() / total
p_female =  d[d.Sex == 'female']['Survived'].count() / total


survived_and_male = d[(d.Sex == 'male') & (d.Survived == 1)]['Survived'].count()
survived_and_female = d[(d.Sex == 'female') & (d.Survived == 1)]['Survived'].count()
p_survived_and_male = survived_and_male / total
p_surived_and_female = survived_and_female / total

print('Observe that sum of probabilities of male and female survivors' +
      f' {survived_and_male + survived_and_female} is equals to survived probability {p_survived}')
sns.barplot(x='Sex', y='Survived', data= d)

pd.crosstab(d['Survived'], d['Sex'], margins = True, normalize = 'all')


df = pd.crosstab(d['Survived'], [d.Sex, d.Pclass], normalize = 'all')
pd.crosstab(d['Survived'], d.Pclass, margins= True, normalize = 'all')
sns.barplot(x='Pclass', y='Survived', data= d)
sns.barplot(x='Sex', y='Survived', hue='Pclass', data= d)
pd.crosstab(d['Survived'], d['Sex'], normalize = 'columns')
