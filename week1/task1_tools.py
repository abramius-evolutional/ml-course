import pandas
import numpy as np
import re

df = pandas.read_csv('titanic.csv', index_col='PassengerId')

print(df.keys())
# for i, key in enumerate(df.keys()):
#     if df.dtypes[i] == np.object:
#         print('--> key:', key)
#         unique_keys = df[key].unique()
#         print('values:', len(df[key]))
#         print('unique:', len(unique_keys))

male_number = (df['Sex'].values == 'male').astype(int).sum()
female_number = (df['Sex'].values == 'female').astype(int).sum()

print('# 1')
print('==> male number', male_number)
print('==> female number', female_number)

print('# 2')
survived_pecent = df['Survived'].sum() * 1. / len(df['Survived'])
print('==> Survived:', round(survived_pecent * 100, 2))

print('# 3')
class1_rate = (df['Pclass'] == 1).astype(int).sum() * 1. / len(df['Pclass'])
print('==> fist class:', round(class1_rate * 100, 2))

print('# 4')
mean_age = df['Age'].mean()
median_age = df['Age'].median()
mode_age = df['Age'].mode()[0]
print('==> Age mean:', mean_age, 'median:', median_age, 'mode:', mode_age)

print('# 5')
sibsp = df['SibSp'].values
parch = df['Parch'].values
print('==> corrcoef SibSp and Parch:', np.corrcoef([sibsp, parch])[0, 1])

print('# 6')
sex = df['Sex'].values
name = df['Name'].values
female_names = name[np.where(sex == 'female')]
b = {}
re_first_name = re.compile(r'.+(Mrs|Mr|Miss|Mme|Ms|Lady|Mlle|Dr)\. {0,1}(\w+)(.+\((\w+) .+\)){0,1}')
for n in female_names:
    # print(n)
    try:
        f_name = re_first_name.match(n).group(4) if re_first_name.match(n).group(4) else re_first_name.match(n).group(2)
        # print('\t', f_name)
        b[f_name] = b.get(f_name, 0) + 1
    except:
        pass

sorted_keys = sorted(b.keys(), key=lambda k: -b[k])
print(list(zip(sorted_keys, [b[key] for key in sorted_keys]))[0])

