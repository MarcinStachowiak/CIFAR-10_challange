import random

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

__author__ = "Marcin Stachowiak"
__version__ = "1.0"
__email__ = "marcin.stachowiak.ms@gmail.com"

def reduce_dim_PCA(X, Y, class_names, n_components=2, visualise=False):
    if visualise:
        print('Features size reduction using PCA')
    pca = PCA(n_components)
    x_reduced = pca.fit_transform(X)
    if visualise:
        plot_first_two_features(x_reduced, Y, class_names, 'PCA of CIFAR 10 dataset')
    return x_reduced


def reduce_dim_TSNE(X, Y, class_names, n_components=2, visualise=False):
    if visualise:
        print('Features size reduction using TSNE')
    nr_samples = random.sample(range(len(X)), int(len(X) / 20))  # samples number reduction
    X = X[nr_samples]
    Y = Y[nr_samples]
    tsne = TSNE(n_components)
    x_reduced = tsne.fit_transform(X)
    if visualise:
        plot_first_two_features(x_reduced, Y, class_names, 'TSNE of CIFAR 10 dataset')
    return x_reduced


def plot_first_two_features(X_reduced, Y, class_names, title):
    plt.figure()
    colors = ['black', 'red', 'sandybrown', 'gold', 'darkgreen', 'mediumspringgreen', 'blue', 'm', 'slategray', 'tan']
    lw = 2

    for i in range(len(class_names)):
        plt.scatter(X_reduced[Y == i, 0], X_reduced[Y == i, 1], color=colors[i], alpha=.8, lw=lw,
                    label=class_names[i], s=1)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()
