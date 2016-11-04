import pandas as pd
import numpy as np
import os
import subprocess
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree

def visualize_tree(tree, feature_names):
    with open("tree.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    # dot -Tpng tree.dot -o tree.png
    command = ["dot", "-Tpng", "tree.dot", "-o", "tree.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

df = pd.read_csv('titanic.csv', index_col='PassengerId')

df = df[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
df = df.dropna()

df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

x, y = df[['Pclass', 'Fare', 'Age', 'Sex']], df['Survived']
clf = DecisionTreeClassifier(random_state=241)
clf.fit(x, y)

importances = clf.feature_importances_
print(importances)

visualize_tree(clf, ['Pclass', 'Fare', 'Age', 'Sex'])

