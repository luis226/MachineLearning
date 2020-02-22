# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:40:19 2020

@author: Luis Galaviz
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class Bayes:
    def __init__(self):
        self.joint_distribution_table = None
    
    def train(self, X, y):
        # These lines objective is to avoid errors if you send a pandas Series
        # or dataframe, just to ensure compatibility
        x_series = [X]
        if type(X) == pd.DataFrame:
            x_series = []
            for col in X.columns:
                x_series.append(X[col])
        y_series = y[y.columns[0]]
        
        # Calculates joint probability table
        jdt = pd.crosstab(y_series, x_series, normalize= 'all')
        self.joint_distribution_table = jdt
    
    def predict(self, X, round_prob = True):
        # Create an array of zeros as a initial response
        y_pred = np.zeros(shape=(X.shape[0],))
        
        # We need to convert X column to tuples cause they help us to index our table
        for i, t in enumerate(X.itertuples(index=False, name= None)):
            # We index our table by our X values
            sub_table = self.joint_distribution_table.loc(axis=1)[t]
            # P(xy)
            prob_xy = sub_table.iloc[1]
            # P(x)
            prob_x = sub_table.sum()
            
            # P(y|x) = P(xy)/P(x)
            y_pred[i] = prob_xy / prob_x
        
        # If round, numbers with prob > 0.5 will be class 1, 0 otherwise
        if round_prob:
            y_pred = np.round(y_pred)
            
        return y_pred

class NaiveBayes:
    def __init__(self):
        self.conditional_tables = None
        self.priors = None
    
    def train(self, X, y):
        self.conditional_tables = []
        # Creating condition probability tables
        if type(X) == pd.DataFrame:
            for col in X.columns:
                crosstable = pd.crosstab(X[col], y, normalize = 'columns')
                self.conditional_tables.append(crosstable)
        else:
            crosstable = pd.crosstab(X, y, normalize = 'columns')
            self.conditional_tables.append(crosstable)
        
        # Filling priors
        class_1 = y.mean()
        class_0 = 1 - class_1
        self.priors = [class_0, class_1]
        
    def predict(self, X, round_prob = True):
        y_pred = np.zeros(shape=(X.shape[0]))
        
        # Itter each row
        for i, (index, row) in enumerate(X.iterrows()):
            # Calculates numerator of probability
            map_0 = self.priors[0]
            map_1 = self.priors[1]
            for j, col in enumerate(row):
                map_0 = map_0 * self.conditional_tables[j].loc[col, 0]
                map_1 = map_1 * self.conditional_tables[j].loc[col, 1]
            
            # Normalizes the results
            p_1 = map_1 / (map_0 + map_1)
            if round_prob:
                p_1 = np.round(p_1)
            y_pred[i] = p_1
        return y_pred
        

def print_results(y_train, y_pred_train, y_test, y_pred_test):
    print("Training confusion matrix")
    cf_train = confusion_matrix(y_train,y_pred_train) 
    print(cf_train)
    print("Test confusion matrix")
    cf_test = confusion_matrix(y_test,y_pred_test) 
    print(cf_test)
    print('\n')
    

# Reading data
df = pd.read_csv('train.csv')
X = df[['Sex','Pclass']]
y = df[['Survived']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
n_train, n_test = X_train.shape[0],  X_test.shape[0]

# Creating one model with sex as predictor
model_sex = Bayes()
model_sex.train(X_train[['Sex']], y_train)
print("Sex only model")
print_results(y_train, model_sex.predict(X_train[['Sex']]),
              y_test, model_sex.predict(X_test[['Sex']]))

# Creating one model with pclass as predictor
model_pclass = Bayes()
model_pclass.train(X_train[['Pclass']], y_train)
print("Pclass only model")
print_results(y_train, model_pclass.predict(X_train[['Pclass']]),
              y_test, model_pclass.predict(X_test[['Pclass']]))

# Creating one model with pclass and sex
model_pclass_sex = Bayes()
model_pclass_sex.train(X_train[['Pclass','Sex']], y_train)
print("Pclass and sex model")
print_results(y_train, model_pclass_sex.predict(X_train[['Pclass','Sex']]),
              y_test, model_pclass_sex.predict(X_test[['Pclass', 'Sex']]))

# Creating a naive bayes model with pclass and sex
model_nb = NaiveBayes()
model_nb.train(X_train, y_train.Survived)
print("Naive bayes model Pclass and Sex")
print_results(y_train, model_nb.predict(X_train),
              y_test, model_nb.predict(X_test))






