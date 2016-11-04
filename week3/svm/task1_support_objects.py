import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC

df = pd.read_csv('svm-data.csv', header=None)

df[0] = df[0].map({0: -1, 1: 1})

X_train = df[[1, 2]].values
y_train = df[0].values

clf = SVC(C=10000000, random_state=241, kernel='linear')
clf.fit(X_train, y_train)

print(list(np.array(clf.support_) + 1))

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 1)
yy = a * xx - (clf.intercept_[0]) / w[1]

i1 = np.where(y_train < 0)
i2 = np.where(y_train > 0)
isupport = np.array(clf.support_)
plt.plot(X_train.T[0][i1], X_train.T[1][i1], 'ro')
plt.plot(X_train.T[0][i2], X_train.T[1][i2], 'go')
plt.plot(X_train.T[0][isupport], X_train.T[1][isupport], 'bo')
plt.plot(xx, yy, '--')
plt.show()