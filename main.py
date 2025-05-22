
import numpy as np
import random as rand


def distance(point, data):

    return np.sqrt(np.sum((point - data)**2, axis=1))


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    def fit(self, X_train):
        minimum, maximum = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [rand.uniform(minimum, maximum) for _ in range(self.n_clusters)]

        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dist = distance(x, self.centroids)
                centroid_id = np.argmin(dist)
                sorted_points[centroid_id-1].append(x)

            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_ids = []
        for x in X:
            dist = distance(x, self.centroids)
            centroid_id = np.argmin(dist)
            centroids.append(self.centroids[centroid_id])
            centroid_ids.append(centroid_id)

        return centroids, centroid_ids

    def fit_predict(self, X):
        self.fit(X)
        _, cluster_indices = self.evaluate(X)
        return np.array(cluster_indices)

