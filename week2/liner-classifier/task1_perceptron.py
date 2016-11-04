import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('perceptron-train.csv', header=None)
df_test = pd.read_csv('perceptron-test.csv', header=None)

X_train, X_test = df_train[[1, 2]].values, df_test[[1, 2]].values
y_train, y_test = df_train[0].values, df_test[0].values

# i1 = np.where(y_train < 0)
# i2 = np.where(y_train >= 0)
# plt.plot(X_train.T[0][i1], X_train.T[1][i1], 'r.')
# plt.plot(X_train.T[0][i2], X_train.T[1][i2], 'g.')
# plt.grid()
# plt.plot(X_test.T[0], X_test.T[1], 'b.')
# plt.show()

classifier = Perceptron(random_state=241)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

true_number = np.sum((predictions==y_test).astype(int))

print("true number %s of %s" % (true_number, len(y_test)))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = Perceptron(random_state=241)
classifier.fit(X_train_scaled, y_train)
predictions = classifier.predict(X_test_scaled)
true_number_scaled = np.sum((predictions==y_test).astype(int))

print("true number %s of %s" % (true_number_scaled, len(y_test)))

n = float(len(y_test))
print("%s - %s = %s" % (true_number_scaled/n, true_number/n, true_number_scaled/n-true_number/n))