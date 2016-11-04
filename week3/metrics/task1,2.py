import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

df_cl = pd.read_csv('classification.csv')

y_true, y_pred = df_cl.values.T

TP = len(np.where((y_true==1) & (y_pred == 1))[0])
FP = len(np.where((y_true==0) & (y_pred == 1))[0])
FN = len(np.where((y_true==1) & (y_pred == 0))[0])
TN = len(np.where((y_true==0) & (y_pred == 0))[0])
print("TP\tFP\tFN\tTN")
print("%s\t%s\t%s\t%s" % (TP, FP, FN, TN))
print(np.array([[TP, FP], [FN, TN]]))

accuracy_score = sklearn.metrics.accuracy_score(y_true, y_pred)
precision_score = sklearn.metrics.precision_score(y_true, y_pred)
recall_score = sklearn.metrics.recall_score(y_true, y_pred)
f1_score = sklearn.metrics.f1_score(y_true, y_pred)

print("accuracy_score", accuracy_score)
print("precision_score", precision_score)
print("recall_score", recall_score)
print("f1_score", f1_score)