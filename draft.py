#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:48:08 2020

@author: kshitij
"""
import numpy as np
import pandas as pd

df = pd.read_csv('Life Expectancy Data.csv')
df.head()

df.dropna(axis=0,inplace = True)

y = df.iloc[:,3]
X = df.drop(df.columns[[0,3]],axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Status'] = le.fit_transform(X['Status'])

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))
sns.heatmap(df.corr(),annot = True)
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, verbose = 1)
rf.fit(X_train,y_train)

rf_pred = rf.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error
mse(y_test,rf_pred)
r2_score(y_test,rf_pred)

import pickle

# Saving model to disk
pickle.dump(rf, open('draft.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('draft.pkl','rb'))
#print(model.predict([[2, 9, 6]]))