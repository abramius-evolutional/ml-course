from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

df = pd.read_csv('abalone.csv')
df['Sex'] = df['Sex'].map({'M': 1, 'I': 0, 'F': -1})

X = df[df.keys()[:-1]]
y = df[df.keys()[-1]]

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
grid = {'n_estimators': range(1, 51)}
forest = RandomForestRegressor(n_estimators=100, random_state=1)
gs = GridSearchCV(forest, grid, scoring='r2', cv=kfold)
gs.fit(X, y)

mean_test_score = gs.cv_results_['mean_test_score']

print(gs.cv_results_['mean_test_score'])