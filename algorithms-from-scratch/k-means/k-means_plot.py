import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


class k_means:

    def __init__(self, k, tol, max_iter):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter


    def fit(self, dataset):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = dataset[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in dataset:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break


    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


X = np.array([[1, 2],
              [4, 3],
              [5, 8 ],
              [8, 6],
              [1, 0.8],
              [7, 9]])

colors = 10*["g", "r", "c", "b", "k"]

clf = k_means(4, 0.001, 300)
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()