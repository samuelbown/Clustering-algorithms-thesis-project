import numpy as np

def distance(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

def distance_matrix(X, Y):
    return np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)

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

class KMeans:
    def __init__(self, n_clusters=3, max_iter=500, tol=1e-10, init='kmeans++', alg='lloyd'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init  # 'random' or 'kmeans++'
        self.alg = alg

    def fit(self, X_train):
        nsamples, nfeatures = X_train.shape

        if self.alg == 'lloyd':

            if self.init == 'kmeans++':
                self.centroids = initialize_centroids_kmeans_pp(X_train, self.n_clusters)
            else:
                minimum, maximum = np.min(X_train, axis=0), np.max(X_train, axis=0)
                self.centroids = np.array([np.random.uniform(minimum, maximum) for _ in range(self.n_clusters)])

            for iteration in range(self.max_iter):
                prevcentroids = self.centroids.copy()

                distances = distance_matrix(X_train, self.centroids)
                labels = np.argmin(distances, axis=1)

                for i in range(self.n_clusters):
                    points = X_train[labels == i]
                    if len(points) > 0:
                        self.centroids[i] = points.mean(axis=0)

                shifts = np.linalg.norm(self.centroids - prevcentroids, axis=1)
                if np.all(shifts <= self.tol):
                    print(f"Lloyd's converged in {iteration + 1} iterations")
                    break


        if self.alg == 'UB':
            if self.init == 'kmeans++':
                self.centroids = initialize_centroids_kmeans_pp(X_train,self.n_clusters)
            else:
                self.centroids = X_train[np.random.choice(nsamples, self.n_clusters, replace=False)]

            labels = np.zeros(nsamples, dtype=int)
            upper_bounds = np.full(nsamples, np.inf)


            for it in range(self.max_iter):
                prevcentroids = self.centroids.copy()


                centdist = distance_matrix(self.centroids, self.centroids)
                np.fill_diagonal(centdist, np.inf)
                s = 0.5 * np.min(centdist, axis=1)



                toupdate = (upper_bounds > s[labels])
                if np.any(toupdate):
                    distances = distance_matrix(X_train[toupdate], self.centroids)
                    newLabels = np.argmin(distances, axis=1)
                    upper_bounds[toupdate] = distances[np.arange(len(newLabels)), newLabels]


                    labels[toupdate] = newLabels

                for k in range(self.n_clusters):
                    points = X_train[labels == k]
                    if len(points) > 0:
                        self.centroids[k] = np.mean(points, axis=0)


                centroid_shifts = np.linalg.norm(self.centroids - prevcentroids, axis=1)
                upper_bounds += centroid_shifts[labels]

                if np.max(centroid_shifts) <= self.tol :
                    print(f"Converged in {it + 1} iterations")
                    break

        if self.alg == 'elkan':
            if self.init == 'kmeans++':
                self.centroids = initialize_centroids_kmeans_pp(X_train, self.n_clusters)
            else:
                minimum, maximum = np.min(X_train, axis=0), np.max(X_train, axis=0)
                self.centroids = np.array([np.random.uniform(minimum, maximum) for _ in range(self.n_clusters)])

            labels = np.zeros(nsamples, dtype=int)
            upper_bounds = np.full(nsamples, np.inf)
            lower_bounds = np.zeros((nsamples, self.n_clusters))

            for it in range(self.max_iter):
                prevcentroids = self.centroids.copy()

                centroid_dists = distance_matrix(self.centroids, self.centroids)
                np.fill_diagonal(centroid_dists, np.inf)
                s = 0.5 * np.min(centroid_dists, axis=1)

                for i in range(nsamples):
                    c = labels[i]

                    if upper_bounds[i] <= s[c]:
                        continue


                    for j in range(self.n_clusters):
                        if j == c:
                            continue
                        if upper_bounds[i] <= lower_bounds[i, j]:
                            continue
                        if upper_bounds[i] <= 0.5 * centroid_dists[c, j]:
                            continue


                        dist = np.linalg.norm(X_train[i] - self.centroids[j])
                        lower_bounds[i, j] = dist

                        if dist < upper_bounds[i]:
                            c = j
                            upper_bounds[i] = dist

                    labels[i] = c

                for j in range(self.n_clusters):
                    points = X_train[labels == j]
                    if len(points) > 0:
                        self.centroids[j] = points.mean(axis=0)

                centroid_shifts = np.linalg.norm(self.centroids - prevcentroids, axis=1)
                for i in range(nsamples):
                    c = labels[i]
                    upper_bounds[i] = np.linalg.norm(X_train[i] - self.centroids[c]) + centroid_shifts[c]
                    lower_bounds[i] = np.maximum(0.0, lower_bounds[i] - centroid_shifts)

                if np.max(centroid_shifts) <= self.tol:
                    print(it)
                    break



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
        _, cluster_indices = self.evaluate(X)
        return np.array(cluster_indices)