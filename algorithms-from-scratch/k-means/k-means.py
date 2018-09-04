import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()


class k_means:
    def __init__(self, k, tol, max_iter):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, dataset):
        self.centroids = []
        for i in range(self.k):
            self.centroids.append(dataset[i])

        for i in range(self.max_iter):
            self.classifications = []

            for i in range(self.k):
                self.classifications.append([])
