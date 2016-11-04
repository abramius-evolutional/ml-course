import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

names = ['name%i' % i for i in range(14)]
df = pd.read_csv('wine.scv', header=None, names=names)
n = len(df.keys())
print('n =', n)

x = df[['name1', 'name11']].values
y = df['name0'].values

stds = np.std(x, axis=0)
means = np.mean(x, axis=0)
norm_x = scale(x)

neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(norm_x, y)

max_1 = np.max(x[:,0])
min_1 = np.min(x[:,0])
max_2 = np.max(x[:,1])
min_2 = np.min(x[:,1])

points = []
for i in np.linspace(min_1, max_1, 30):
	for j in np.linspace(min_2, max_2, 30):
		norm_i = (i - means[0]) / stds[0]
		norm_j = (j - means[1]) / stds[1]
		points.append([norm_i, norm_j])

prediction = neigh.predict(points)
x1, x2 = np.array(points).T
plt.plot(x1[np.where(prediction==1)], x2[np.where(prediction==1)], 'r.', alpha=0.5)
plt.plot(x1[np.where(prediction==2)], x2[np.where(prediction==2)], 'g.', alpha=0.5)
plt.plot(x1[np.where(prediction==3)], x2[np.where(prediction==3)], 'b.', alpha=0.5)

x1, x2 = norm_x[:,0], norm_x[:,1]
plt.plot(x1[np.where(y==1)], x2[np.where(y==1)], 'ro')
plt.plot(x1[np.where(y==2)], x2[np.where(y==2)], 'go')
plt.plot(x1[np.where(y==3)], x2[np.where(y==3)], 'bo')

plt.show()

