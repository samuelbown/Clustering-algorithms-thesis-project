import numpy as np
from numba import njit, prange

def initialize_centroids_kmeans_pp(X, k):
    nsamples, _ = X.shape
    centroids = np.empty((k, X.shape[1]))
    c1 = np.random.randint(nsamples)
    centroids[0] = X[c1]
    closestdistSq = np.full(nsamples, np.inf)
    for c_id in range(1, k):
        distSq = np.sum((X - centroids[c_id - 1]) ** 2, axis=1)
        closestdistSq = np.minimum(closestdistSq, distSq)
        probs = closestdistSq / closestdistSq.sum()
        next_idx = np.random.choice(nsamples, p=probs)
        centroids[c_id] = X[next_idx]
    return centroids

@njit(parallel=True, fastmath=True)
def elkanFit(X, initial_centroids, labels, max_iter=100, tol=1e-4):
    n_samples, n_features = X.shape
    k = initial_centroids.shape[0]

    centroids = initial_centroids.copy()
    prev_centroids = np.empty_like(centroids)

    upper_bounds = np.empty(n_samples, dtype=np.float64)
    lower_bounds = np.zeros((n_samples, k), dtype=np.float64)

    centroid_dists = np.zeros((k, k), dtype=np.float64)
    s = np.zeros(k, dtype=np.float64)

    for i in prange(n_samples):
        c = labels[i]
        dist = 0.0
        for f in range(n_features):
            diff = X[i, f] - centroids[c, f]
            dist += diff * diff
        upper_bounds[i] = np.sqrt(dist)

    for it in range(max_iter):
        for i in range(k):
            for j in range(k):
                if i == j:
                    centroid_dists[i, j] = np.inf
                else:
                    dist = 0.0
                    for f in range(n_features):
                        diff = centroids[i, f] - centroids[j, f]
                        dist += diff * diff
                    centroid_dists[i, j] = np.sqrt(dist)
            s[i] = 0.5 * np.min(centroid_dists[i])

        for i in prange(n_samples):
            c = labels[i]
            if upper_bounds[i] <= s[c]:
                continue

            for j in range(k):
                if j == c:
                    continue
                if upper_bounds[i] <= lower_bounds[i, j]:
                    continue
                if upper_bounds[i] <= 0.5 * centroid_dists[c, j]:
                    continue

                d = 0.0
                for f in range(n_features):
                    diff = X[i, f] - centroids[j, f]
                    d += diff * diff
                d = np.sqrt(d)

                lower_bounds[i, j] = d

                if d < upper_bounds[i]:
                    c = j
                    upper_bounds[i] = d

            labels[i] = c

        prev_centroids[:] = centroids[:]

        counts = np.zeros(k, dtype=np.int32)
        centroids[:] = 0.0

        for i in prange(n_samples):
            c = labels[i]
            for f in range(n_features):
                centroids[c, f] += X[i, f]
            counts[c] += 1

        for j in range(k):
            if counts[j] > 0:
                for f in range(n_features):
                    centroids[j, f] /= counts[j]

        shifts = np.zeros(k, dtype=np.float64)
        for j in range(k):
            shift = 0.0
            for f in range(n_features):
                diff = centroids[j, f] - prev_centroids[j, f]
                shift += diff * diff
            shifts[j] = np.sqrt(shift)

        maxShift = np.max(shifts)

        for i in prange(n_samples):
            upper_bounds[i] += maxShift
            for j in range(k):
                lower_bounds[i, j] = max(0.0, lower_bounds[i, j] - maxShift)

        if maxShift <= tol:
            break

    return centroids, labels



class KMeans:
    def __init__(self, n_clusters=3, max_iter=500, tol=1e-4, init='kmeans++', alg='lloyd'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.alg = alg
        self.centroids = None
        self.labels = None

    def fit(self, X_train):
        X_train = np.ascontiguousarray(X_train, dtype=np.float64)
        nsamples, nfeatures = X_train.shape

        if nsamples == 0:
            self.centroids = None
            self.labels = None
            return

        if self.init == 'kmeans++':
            centroids = initialize_centroids_kmeans_pp(X_train, self.n_clusters)
        else:
            idx = np.random.choice(nsamples, self.n_clusters, replace=False)
            centroids = X_train[idx]


        distances = np.linalg.norm(X_train[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        centroids, labels = elkanFit(
            X_train, centroids, labels, max_iter=self.max_iter, tol=self.tol
        )
        self.centroids = centroids
        self.labels = labels



    def evaluate(self, X):
        centroids = []
        centroidz = []
        for x in X:
            dist = np.linalg.norm(x - self.centroids, axis=1)
            centroid = np.argmin(dist)
            centroids.append(self.centroids[centroid])
            centroidz.append(centroid)
        return centroids, centroidz


    def fit_predict(self, X):
        self.fit(X)
        return self.labels
