#-----------------------------------
# Стохастический градиентный спуск
#-----------------------------------

import numpy as np
import matplotlib.pyplot as plt


# сигмоидная функция потерь
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))

# производная функции потерь по вектору w
def df(w, x, y):
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y


x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
x_train = [x + [1] for x in x_train]
x_train = np.array(x_train)
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

n_train = len(x_train)  # размер обучающей выборки
w = [0.0, 0.0, 0.0]     # начальные весовые коэф.
nt = 0.0005  # шаг сходимости
lm = 0.01    # скорость забывания для Q
N = 500      # число итераций

Q = np.mean([loss(x, w, y) for x, y in zip(x_train, y_train)])
Q_plot = [Q]

for i in range(N):
    k = np.random.randint(0, n_train - 1)
    ek = loss(w, x_train[k], y_train[k])
    w = w - nt * df(w, x_train[k], y_train[k])
    Q = lm * ek + (1 - lm) * Q
    Q_plot.append(Q)

print(f'Вектор w = {w}')
# print(Q_plot)

line_x = list(range(max(x_train[:, 0])))  # формирование графика разделяющей линии
line_y = [-x * w[0] / w[1] - w[2] for x in line_x]

# формирование точек 1-го и 2-го классов
x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

# формирование точек двух классов на графике
plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')

plt.plot(line_x, line_y, color='green')

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel('Длина')
plt.xlabel('Ширина')
plt.grid()
plt.show()