#---------------------------------------------
# Агломеративная иерархическая кластеризация
#---------------------------------------------

from itertools import cycle
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt


# функция для отображения дендограммы (из репозитория sklearn)
def plot_dendrogram(model, **kwargs):

    # Дочерние элементы иерархической кластеризации
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


x = [(89, 151), (114, 120), (156, 110), (163, 153), (148, 215), (170, 229), (319, 166), (290, 178), (282, 222)]
x = np.array(x)

NC = 3  # максимальное число кластеров

clustering = AgglomerativeClustering(n_clusters=NC, linkage="ward", metric='euclidean')
x_pr = clustering.fit_predict(x)

f, ax = plt.subplots(1, 2)
kk = 0
for c, n in zip(cycle('bgrcmykgrcmykgrcmykgrcmykgrcmykgrcmyk'), range(NC)):
    clst = x[x_pr == n].T
    print(clst)
    print()
    ax[0].scatter(clst[0], clst[1], s=10, color=c)

    count = 0
    for _ in clst[0]:
        if count < len(clst[0]):
            ax[0].annotate(f'x{kk}', xy=(clst[0][count], clst[1][count]), xytext=(clst[0][count] - 3, clst[1][count] +
                                                                                  2))
        kk += 1
        count += 1

plot_dendrogram(clustering, ax=ax[1])
plt.show()
