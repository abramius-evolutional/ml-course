import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale


df = pd.read_csv('wine.scv', header=None)
n = len(df.keys())
print('n =', n)
x = df[df.keys()[1:]].values
y = df[df.keys()[0]].values

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

accurs = []
ks = range(1, 51)

for k in ks:
    neigh = KNeighborsClassifier(n_neighbors=k)
    accur = np.array(cross_val_score(neigh, x, y, cv=kfold))
    mean_accuracy = np.mean(accur)
    accurs.append(mean_accuracy)

argmax = np.argmax(accurs)
print('\nbefore scaling:')
print('k[argmax(accuracy)] =', ks[argmax], 'value =', accurs[argmax])

x = scale(x)

accurs = []

for k in ks:
    neigh = KNeighborsClassifier(n_neighbors=k)
    accur = np.array(cross_val_score(neigh, x, y, cv=kfold))
    mean_accuracy = np.mean(accur)
    accurs.append(mean_accuracy)

argmax = np.argmax(accurs)
print('\nafter scaling:')
print('k[argmax(accuracy)] =', ks[argmax], 'value =', accurs[argmax])


colors = {1: 'r', 2: 'g', 3: 'b'}
for j in range(n-1):
    x1, x2 = x[:,0], x[:,j]
    plt.subplot(4, 4, j+1, title='comparison %i with %s' % (0, j))
    plt.plot(x1[np.where(y==1)], x2[np.where(y==1)], 'r.')
    plt.plot(x1[np.where(y==2)], x2[np.where(y==2)], 'g.')
    plt.plot(x1[np.where(y==3)], x2[np.where(y==3)], 'b.')
plt.show()

