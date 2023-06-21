#---------------------------------
# Алгоритм кластеризации DBSCAN
#---------------------------------

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd


x = np.array([[98, 62], [80, 95], [71, 130], [89, 164], [137, 115], [107, 155], [109, 105], [174, 62], [183, 115],
              [164,153], [142, 174], [140, 80], [308, 123], [229, 171], [195, 237], [180, 298], [179, 340],
              [251, 262], [300, 176], [346, 178], [311, 237], [291, 283], [254, 340], [215, 308], [239, 223],
              [281, 207], [283, 156]])


cls = DBSCAN(eps=55, min_samples=3, metric='euclidean')
cls.fit(x)

labels = cls.labels_

# вывод результатов в таблицу
df = pd.DataFrame(x)
df['Class'] = labels
print(df)

# вывод результатов на графике
colors = ['green', 'blue', 'brown', 'red', 'yellow']
set_labels = [x for x in set(labels)]

for i in set_labels:
    if i == -1:
        a = x[labels == i]
        plt.scatter(a[:, 0], a[:, 1], s=30, c='black') # выбросы/шумы

    a = x[labels == i]
    plt.scatter(a[:,0], a[:,1], s=30, c=colors[i])

plt.grid()
plt.show()