#--------------------------------------------
# Алгоритм Random Forest для задач регрессии
#--------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


x = np.arange(0, np.pi, 0.1)
n_samples = len(x)
y = np.cos(x) + np.random.normal(0.0, 0.1, n_samples)
x = x.reshape(-1, 1)

clf = RandomForestRegressor(n_estimators=4, max_depth=2, random_state=1)
clf.fit(x, y)
yy = clf.predict(x)

plt.plot(x, y, label='cos(x)')
plt.plot(x, yy, label='RF Regressor')
plt.title("4 дерева глубиной 2")
plt.legend()
plt.grid()
plt.show()