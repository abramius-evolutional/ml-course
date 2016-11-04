import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data-logistic.csv', header=None)

y = df[0].values
x = df[[1, 2]].values

def P(x, w):
    return 1. / (1. + np.exp(-1 * np.dot(x, w)))

l = len(y)
C = 10.
step = 0.1
w = np.array([0., 0.])

for p in range(10000):
    e = np.exp(-1 * y * (np.dot(x, w)))
    e2 = 1. - 1. / (1 + e)
    s = np.sum((x.T * e2 * y).T, axis=0)
    dw = step / l * s - step * C * w
    if np.sum(np.abs(dw)) < 1e-9:
        break
    w = w + dw

print('iteration number =', p)
print('w =', w)

est_y = P(x, w)

score = roc_auc_score(y, est_y)
print('score =', round(score, 3))

space = [1., -w[0] / w[1]]
plt.subplot(aspect='equal')
plt.plot([-2 * space[0], 2 * space[0]], [-2 * space[1], 2 * space[1]], 'r--')
plt.plot(x.T[0][np.where(y > 0)], x.T[1][np.where(y > 0)], 'g.')
plt.plot(x.T[0][np.where(y < 0)], x.T[1][np.where(y < 0)], 'r.')
plt.plot(x.T[0][np.where(y != np.sign(est_y-0.5))], x.T[1][np.where(y != np.sign(est_y-0.5))], 'b+')
plt.grid()
plt.show()
