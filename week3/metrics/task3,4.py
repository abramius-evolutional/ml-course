# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

df = pd.read_csv('scores.csv')

y_true = df['true'].values

for name in ['score_logreg', 'score_svm', 'score_knn', 'score_tree']:
    score_clf = df[name].values
    roc = sklearn.metrics.roc_auc_score(y_true, score_clf)

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, score_clf)

    max_precision_recall_70 = np.max(precision[np.where(recall >= 0.7)])

    # plt.subplot(aspect=True)
    # plt.plot(recall, precision, 'ro-')
    # plt.grid()
    # plt.show()

    print(name)
    print("\troc_auc_score =", roc)
    print("\tmax_precision (recall>=0.7)", max_precision_recall_70)