from datetime import datetime
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


close_prices = pd.read_csv('close_prices.csv')
close_prices['date'] = [datetime.strptime(d, '%Y-%m-%d') for d in close_prices['date']]

X_train = close_prices[[key for key in close_prices.keys()[1:]]]
pca = PCA(n_components=10)
pca.fit_transform(X_train)
print('==> explained_variance_ratio_ values are:')
print(pca.explained_variance_ratio_)
print('\t', 'and cummulative:')
print('\t', [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1,len(pca.explained_variance_ratio_))])

X_ = pca.transform(X_train)
first_component = X_.T[0]

djia_index = pd.read_csv('djia_index.csv')
djia = djia_index['^DJI'].values

corr = np.corrcoef([first_component, djia])
print('==>')
print(corr)

print('==> company with max significance for the first component:')
idx_of_max = np.argmax(pca.components_[0])
print(list(X_train.keys()))
print(X_train.keys()[idx_of_max], '(Verizon Communications Inc.)', pca.components_[0][idx_of_max])
