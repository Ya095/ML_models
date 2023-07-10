#------------------------------
# Решающее дерево (регрессия)
#------------------------------

from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, np.pi, 0.1).reshape(-1, 1)
y = np.cos(x)

clf = tree.DecisionTreeRegressor(max_depth=3)
clf.fit(x, y) # обучаем дерево (строим его) по обучающим данным
yy = clf.predict(x) # делаем прогнозы
k = np.array([5]).reshape(1,-1)

# tree.plot_tree(clf) # строит изображение дерева
plt.plot(x, y, label='cos(x)')
plt.plot(x, yy, label='DT regression')
plt.grid()
plt.legend()
plt.title('max_depth=3')
plt.show()
