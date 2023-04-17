# Алгоритм регрессии AdaBoost на решающих деревьях

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


x = np.arange(0, np.pi/2, 0.1).reshape(-1, 1)
y = np.cos(x) + np.random.normal(0, 0.1, x.shape)

T = 1           # число алгоритмов в композиции
max_depth = 2   # максимальная глубина решающих деревьев
algs = []       # список полученных алгоритмов
s = np.array(y.ravel()) # остатки, изначально берем как значения ф-ии "у"

for n in range(T):
    algs.append(DecisionTreeRegressor(max_depth=max_depth))
    algs[-1].fit(x, s)
    s -= algs[-1].predict(x)    # пересчитываем остатки

# восстанавливаем исходный сигнал по набору полученных деревьев
yy = algs[0].predict(x)
for n in range(1, T):
    yy += algs[n].predict(x)

# визуализация
f, ax = plt.subplots(2, 1, figsize=(9, 5))
ax[0].plot(x, y, label='sin(x)')
ax[0].plot(x, yy, label='Апроксимация')
ax[0].legend()
ax[0].grid()

ax[1].plot(x, s, label='Разница') # разница между исходным графиком и апроксимацией
plt.legend()
plt.grid()
plt.show()