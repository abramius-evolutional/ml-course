# coding: utf-8
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack, vstack
from tqdm import tqdm
import pandas as pd
import numpy as np

# data_train = pd.read_csv('salary-train.csv')
# data_test = pd.read_csv('salary-test-mini.csv')

# getting lowercase of the fullDescription feature
# for i in tqdm(range(len(data_train))):
#     data_train['FullDescription'][i] = data_train['FullDescription'][i].lower()
# data_train.to_csv('salary-train-lower.csv', index=False)

# for i in range(len(data_test)):
#     data_test['FullDescription'][i] = data_test['FullDescription'][i].lower()
# data_test.to_csv('salary-test-mini-lower.csv', index=False)

data_train = pd.read_csv('salary-train-lower.csv')[:]
data_test = pd.read_csv('salary-test-mini-lower.csv')

y_train = data_train['SalaryNormalized'].values

# replace nan value to 'nan' string value
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

print('stage 1')

data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

vectorizer = TfidfVectorizer(min_df=5)
vdata_train = vectorizer.fit_transform(data_train['FullDescription'])
vdata_test = vectorizer.transform(data_test['FullDescription'])
feature_mapping = vectorizer.get_feature_names()

print('stage 2')

print(vdata_train.shape)
print(X_train_categ.shape)
features_train = hstack([vdata_train, X_train_categ])
features_test = hstack([vdata_test, X_test_categ])

print('stage 3')

rgs = Ridge(alpha=1, random_state=241)
rgs.fit(features_train, y_train)
print('regression prediction')
print(rgs.predict(features_test))

