# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# считываем тренировочные данные
df = pd.read_csv('features.csv', index_col='match_id')

# формируем список полей, которые относятся к результату
# эти поля, которые должны быть исключены из тренировочных данных
result_keys = ['duration', 'radiant_win', 'tower_status_radiant',
    'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']

# формируем список полей для тренировочных данных
train_keys = [key for key in df.keys() if key not in result_keys]

# формируем список полей, тип которых - числовой, а не категориальный
numeric_keys = [key for key in train_keys if (key[-4:]!='hero') and (key!='lobby_type')]

# формируем список полей связанных с героями
hero_keys = [key for key in train_keys if (key[-4:]=='hero')]

# формируем датафрейм со всеми полями (числоыми и категориальными)
# но исключая поля результатов
df_train = df[train_keys]

# определяем размер выборки
n_train = df_train.values.shape[0]

# выводим информацию о том, какие поля имеют пропуски
# также выводим процент пропусков
print('\n\n\n==> Empty values')
for key in train_keys:
    count = df_train[key].count()
    length = df_train[key].values.shape[0]
    empty_count = length - count
    if empty_count > 0:
        print('Empty values', 
            '%s %%' % round(100.*empty_count / length, 1), 
            '(%s of %s)' % (empty_count, length), key)

# заполняем пропуски нулями
df_train = df_train.fillna(0)

# формируем матрицу признаков и вектор ответов
X_train = df_train.values
y_train = df['radiant_win'].values

# -------------------
# Градиентный бустинг
# -------------------

n_estimators_list = []#[10, 20, 30, 40, 50]

# пробегаем параметром n_estimators по списку проверяемых значений
for n_estimators in n_estimators_list:
    start = datetime.now()

    # формируем генератор для кроссвалидации
    kfold = KFold(n_splits=5, shuffle=True)

    # формируем классификатор
    gbc = GradientBoostingClassifier(n_estimators=n_estimators, verbose=False)

    # запускаем кроссвалидацию
    accur = np.array(cross_val_score(gbc, X_train, y_train, cv=kfold, scoring='roc_auc'))

    # принтуем значения
    print('\nn_estimators =', n_estimators)
    print('accur =', list(accur))
    print('mean accur =', np.mean(accur))
    print('iteration time =', (datetime.now() - start).total_seconds())

# -----------------------
# Логистическая регрессия
# -----------------------

# определяем функцию, которая будет пробегаться по списку параметров
# регуляризации L2, для каждого значения с помощью кроссвалидации оценивать
# качество классификатора и выводить значения
def logistic_regression_estimator(X, y, cs):
    accurs = []
    for C in cs:
        start = datetime.now()
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        lgc = LogisticRegression(C=C)
        accur = np.array(cross_val_score(lgc, X, y, cv=kfold, scoring='roc_auc'))
        accurs.append(np.mean(accur))
        print('iteration time =', (datetime.now() - start).total_seconds())
    print('==> Logistic Regression')
    i = np.argmax(accurs)
    print('the best C', cs[i])
    print('the best accur', accurs[i])
    plt.plot(cs, accurs)
    plt.plot([cs[i]], [accurs[i]], 'ro')
    plt.grid()
    plt.ylabel('Mean accur')
    plt.xlabel('C (L2 regularizer)')
    plt.show()

# приводим ответы в удобоворимый для логистической регрессии вид
# хотя кажется, что это делать не обязательно, возможно она делает это сама
y_train[np.where(y_train == 0)] = -1

# нормируем признаки (X_train содержатся все признаки, и категариальные и числовые)
scaler = StandardScaler()
X_train_normed = scaler.fit_transform(X_train)

# запускаем перебор значений С (регуляризатора) для логистической регрессии
# также данный метод вызывался с более широким диапазоном перебора
logistic_regression_estimator(X_train_normed, y_train, cs=range(50, 60, 3))

# вытаскиваем только те поля, которые имеют числовой тип
X_train_numeric = df_train[numeric_keys].values

# нормируем числовые поля
scaler = StandardScaler()
X_train_numeric_normed = scaler.fit_transform(X_train_numeric)

# запускаем перебор значений для параметра регуляризатора для логистической регрессии
# также данный метод вызывался с более широким диапазоном перебора
# на этот раз запускает перебор для матрицы с отсутствием категариальных признаков
logistic_regression_estimator(X_train_numeric_normed, y_train, cs=range(80, 90, 3))

# далее кодируем категориальные признаки по методу "мешка слов"
unique_hero_values = []
hero_values = df_train[hero_keys].values

# определяем уникальный набор героев, сортируем для удобства
# и считаем количество n_hero
unique_hero_values = np.append([], hero_values)
unique_hero_values = sorted(list(set(unique_hero_values)))
n_hero = len(unique_hero_values)
print('==> Unique heros number =', n_hero)

# достаем связанные с героями признаки
df_hero = df[hero_keys]

# формируем матрицу дополнительных признаков
X_pick = np.zeros([n_train, n_hero])

# для каждой записи из выборки
for i in tqdm(range(n_train)):
    # пробегаем по количеству героев в команде
    for p in [1, 2, 3, 4, 5]:
        # формируем имена команд с двух сторон
        r_field_name = 'r%d_hero' % p
        d_field_name = 'd%d_hero' % p
        # далее получаем id данных героев
        r_hero = df[r_field_name].values[i]
        d_hero = df[d_field_name].values[i]
        # на основании id героев получаем индекс из уникального списка героев
        r_j = unique_hero_values.index(r_hero)
        d_j = unique_hero_values.index(d_hero)
        # устанавливаем значения 1 и -1 в матрицу дополнительных признаков
        X_pick[i, r_j] = 1
        X_pick[i, d_j] = -1

# расширяем матрицу числовых признаков "мешком слов"
X_train_numeric_normed = np.append(X_train_numeric_normed, X_pick, axis=1)

# запускаем логистическую регрессию на расширенной матрице
# также данный метод запускался с большим диапазоном перебора параметра C
# с дискретом 0.1. Оптимальным оказался C=0.1
logistic_regression_estimator(X_train_numeric_normed[:,:], y_train[:], cs=[0.1])

# после перебора параметров и определения того, какой метод является наилучшем
# (в нашем случае это логистическая регрессия с параметром C=0.1 и матрицей признаков
#  включающей категориальные признаки в виде "мешка слов")
# мы запускаем данный метод на всей тренировчной выборке
# (т. е. ранее лог. регрессия запускалась только на части выборки при кроссвалидации)
lgc = LogisticRegression(C=0.1)
lgc.fit(X_train_numeric_normed, y_train)

# классификатор обучен, далее необходимо получить предсказания для тестовой выборки
df_test_total = pd.read_csv('features_test.csv', index_col='match_id')

# по аналогии формируем датафрейм и заполняем пропуски
df_test = df_test_total[train_keys]
df_test = df_test.fillna(0)

# определяем размер выборки и определяем заготовку для матрицы 
# дополнительных параметров "мешка слов"
n_test = df_test.values.shape[0]
X_pick = np.zeros([n_test, n_hero])

# по аналогии заполняем матрицу "мешка слов"
for i, u in tqdm(list(enumerate(range(n_test)))):
    for p in [1, 2, 3, 4, 5]:
        r_field_name = 'r%d_hero' % p
        d_field_name = 'd%d_hero' % p
        r_hero = df_test[r_field_name].values[i]
        d_hero = df_test[d_field_name].values[i]
        r_j = unique_hero_values.index(r_hero)
        d_j = unique_hero_values.index(d_hero)
        X_pick[i, r_j] = 1
        X_pick[i, d_j] = -1

# формируем матрицу числовых признаков и нормируем ее
# нормировка происходит не по параметрам тестовой выборки
# а с помощью обученного преобразователя по тренировочной выборке
X_test_numeric = df_test[numeric_keys].values
X_test_numeric_normed = scaler.transform(X_test_numeric)

# формируем расширенную матрицу признаков (числовые признаки + "мешок слов")
X_test_numeric_normed = np.append(X_test_numeric_normed, X_pick, axis=1)

# получаем предсказания для тестовой выборки
y_predict  = lgc.predict_proba(X_test_numeric_normed)[:,1]

# формируем итоговый csv файл с предсказаниями
result_csv_string = 'match_id,radiant_win\n'
for i in range(n_test):
    result_csv_string += '%s,%s\n' % (df_test_total.index[i], y_predict[i])
with open('result.csv', 'w') as f:
    f.write(result_csv_string)
    f.close()










