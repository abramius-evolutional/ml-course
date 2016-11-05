from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('gbm-data.csv')
data = df.values

X = data[:,1:]
y = data[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

def sigmoid(arr):
    return 1./(1. + np.exp(-arr))

# for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
for learning_rate in [0.2, 0.1]:
    clf = GradientBoostingClassifier(n_estimators=250,
        learning_rate=learning_rate,
        # verbose=True,
        random_state=241)
    clf.fit(X_train, y_train)

    predict_train_by_iter = clf.staged_decision_function(X_train)
    predict_test_by_iter = clf.staged_decision_function(X_test)

    loss_train_by_iter = []
    loss_test_by_iter = []
    
    for predict in predict_train_by_iter:
        loss_value = log_loss(y_train, sigmoid(predict))
        loss_train_by_iter.append(loss_value)

    for predict in predict_test_by_iter:
        loss_value = log_loss(y_test, sigmoid(predict))
        loss_test_by_iter.append(loss_value)

    min_loss_index = np.argmin(loss_test_by_iter)
    print('learning_rate=%s, min_loss_value=%s, iteration(from 1)=%s' % (
        learning_rate,
        loss_test_by_iter[min_loss_index],
        min_loss_index + 1
    ))

    plt.title(learning_rate)
    plt.plot(loss_train_by_iter)
    plt.plot(loss_test_by_iter)
    plt.show()

clf = RandomForestClassifier(n_estimators=37, random_state=241)
clf.fit(X_train, y_train)
prediction = clf.predict_proba(X_test)
loss_value = log_loss(y_test, prediction)
print('Random forest classifier min loss value = ', loss_value)

