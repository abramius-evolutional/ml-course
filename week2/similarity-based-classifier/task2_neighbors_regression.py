import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn.datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

dataset = sklearn.datasets.load_boston()

data, target = dataset['data'], dataset['target']

norm_data = scale(data)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

errs = []
ps = []
for p in np.linspace(1, 10, 200):
	neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
	accur = np.array(cross_val_score(neigh, norm_data, target, cv=kfold, scoring='neg_mean_squared_error'))
	ps.append(p)
	errs.append(np.mean(accur))
	print(p)

argmax = np.argmax(errs)
print('==> p =', ps[argmax], 'error =', errs[argmax])
