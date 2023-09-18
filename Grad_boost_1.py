#------------------------------------------
# Градиентный бустинг
#------------------------------------------

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt


x, y = make_regression(n_samples=10, n_features=3)

df = pd.DataFrame(x)
df['y_true'] = y
df['y_pred_0'] = df['y_true'].mean()
mae = mean_absolute_error(df['y_true'], df['y_pred_0'])  # в среднем модель ошибается на это число

# Перед тем как делать град. бустинг - считаем остатки (residuals)
# На их основе обучаются последующие алгоритмы
df['residual_0'] = df['y_true'] - df['y_pred_0']
tree_1 = DecisionTreeRegressor(max_depth=1)
tree_1.fit(df[[0,1,2]], df['residual_0'])
df['tree_pred_1'] = tree_1.predict(df[[0,1,2]])

# plot_tree(tree_1)
# plt.show()

nu = 0.1
pd.set_option('display.max_columns', None)
df['y_pred_1'] = df['y_pred_0'] + nu * df['tree_pred_1']

mae1 = mean_absolute_error(df['y_true'], df['y_pred_1']) # ошибка для первого предсказания
df['residual_1'] = df['y_true'] - df['y_pred_1']

####################################################
# Далее создем второй алгоритм (второе дерево), на ошибках предыдущего и тд

tree_2 = DecisionTreeRegressor(max_depth=1)
tree_2.fit(df[[0,1,2]], df['residual_1'])

df['y_pred_2'] = df['y_pred_1'] + nu * tree_2.predict(df[[0,1,2]])
mae2 = mean_absolute_error(df['y_true'], df['y_pred_2'])

# print(f'mae - {mae}')
# print(f'mae1 - {mae1}')
# print(f'mae2 - {mae2}')

##############################################################################
##############################################################################
# реализация через цикл

df = df[[0, 1, 2, 'y_true']].copy()

n = 40  # число алгоритмов в композиции
max_depth = 2
lm = 0.1
algs = []
df['y_pred'] = df['y_true'].mean()

for i in range(n):
    df['residual'] = df['y_true'] - df['y_pred']
    algs.append(DecisionTreeRegressor(max_depth=max_depth))
    algs[-1].fit(df[[0,1,2]], df['residual'])
    df['y_pred'] = df['y_pred'] + lm * algs[-1].predict(df[[0,1,2]])
    print(mean_absolute_error(df['y_true'], df['y_pred']))


# Предсказание по обученным алгоритмам
test = df[[0, 1, 2]].copy()  # допустим, что это тестовая выборка
test['pred'] = df['y_true'].mean()

for alg in algs:
    test['pred'] += lm * alg.predict(test[[0, 1, 2]])

print(test)
print()
print(df)