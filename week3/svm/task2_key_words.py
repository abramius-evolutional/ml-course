from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np


newsgroups = datasets.fetch_20newsgroups(
    subset='all', 
    categories=['alt.atheism', 'sci.space']
)

data, target = newsgroups.data, newsgroups.target

print("len(data) =", len(data), "len(target) =", len(target))

vectorizer = TfidfVectorizer()
vdata = vectorizer.fit_transform(data)
feature_mapping = vectorizer.get_feature_names()

def search_optimum_C():
    grid = {'C': np.power(10.0, np.arange(-5, 6))}

    kfold = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kfold)
    gs.fit(vdata, target)

    for a in gs.grid_scores_:
        print(a)

    # mean: 0.55263, std: 0.02812, params: {'C': 1.0000000000000001e-05}
    # mean: 0.55263, std: 0.02812, params: {'C': 0.0001}
    # mean: 0.55263, std: 0.02812, params: {'C': 0.001}
    # mean: 0.55263, std: 0.02812, params: {'C': 0.01}
    # mean: 0.95017, std: 0.00822, params: {'C': 0.10000000000000001}
    # mean: 0.99328, std: 0.00455, params: {'C': 1.0}
    # mean: 0.99328, std: 0.00455, params: {'C': 10.0}
    # mean: 0.99328, std: 0.00455, params: {'C': 100.0}
    # mean: 0.99328, std: 0.00455, params: {'C': 1000.0}
    # mean: 0.99328, std: 0.00455, params: {'C': 10000.0}
    # mean: 0.99328, std: 0.00455, params: {'C': 100000.0}

# search_optimum_C()
optimum_C = 10.0
clf = SVC(kernel='linear', random_state=241, C=optimum_C)
clf.fit(vdata, target)

print(clf.coef_.indices)
indeces = clf.coef_.indices
values = np.abs(clf.coef_.data)

s_i = np.argsort(values)

names = []
for i in s_i[-10:]:
    idx = indeces[i]
    value = values[i]
    name = feature_mapping[idx]
    names.append(name)

print(names)
print(','.join(sorted(names)))


