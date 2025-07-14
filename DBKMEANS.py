import math
import numpy as np
from kmeans import KMeans as kme


class DBKMEANS:
    def fit(self, data_points, eps, minpts, kclust):
        k = len(data_points[0])
        root = self.setup_tree(data_points, k)
        labels = [-1] * len(data_points)
        clusters = []
        noise = []
        visited = set()

        for index, point in enumerate(data_points):
            if index in visited:
                continue
            visited.add(index)

            neighbours = self.nearest_neighbours(root, point, eps, k)

            if len(neighbours) + 1 < minpts:
                noise.append(point)
            else:
                cluster = []
                self.expand_neighbours(index, data_points, point, eps, minpts, cluster, clusters, neighbours, labels,
                                       visited, root, k)
                clusters.append(cluster)

        clusters = [np.array(cluster) for cluster in clusters]

        clusterCenters = []
        cluster_sizes = []
        m = len(clusters)
        k = kclust

        for cluster in clusters:
            center = cluster.mean(axis=0)
            size = len(cluster)
            clusterCenters.append(center)
            cluster_sizes.append(size)
        print(len(clusterCenters))
        if m > k:
            clusters, clusterCenters = merge_clusters(clusters, clusterCenters, k)
        elif m < k:
            clusters, clusterCenters = split_clusters(clusters, clusterCenters, k)

        print(len(clusterCenters))
        return np.array(clusterCenters)


    def expand_neighbours(self, index, data_points, point, eps, minpts, cluster, clusters, neighbours, labels, visited,
                          root, k):
        cluster.append(data_points[index])
        labels[index] = len(clusters)
        queue = neighbours.copy()
        while queue:
            neighbour_index = queue.pop(0)
            if neighbour_index not in visited:
                visited.add(neighbour_index)
                new_neighbours = self.nearest_neighbours(root, data_points[neighbour_index], eps, k)
                if len(new_neighbours) + 1 >= minpts:
                    temp_list = []
                    for x in new_neighbours:
                        if x not in queue:
                            temp_list.append(x)
                    queue.extend(temp_list)

            if labels[neighbour_index] == -1:
                labels[neighbour_index] = len(clusters)
                cluster.append(data_points[neighbour_index])

    def setup_tree(self, data_points, k):
        root = None
        for index, x in enumerate(data_points):
            root = insert(root, x, index, k)
        return root

    def nearest_neighbours(self, root, point, eps, k, depth=0, neighbours=None):
        if root is None:
            return neighbours

        if neighbours is None:
            neighbours = []

        d = self.distance(point, root.point)
        if d <= eps:
            neighbours.append(root.index)
        cd = depth % k  # cd - current dimension, each layer of the tree the dimension changes to narrow down on the point quicker

        if point[cd] < root.point[
            cd]:  # if x/y of the point is less/more than the x/y of the current node then go left/right down the tree
            self.nearest_neighbours(root.left, point, eps, k, depth + 1, neighbours)
            if abs(point[cd] - root.point[
                cd]) <= eps:  # if the current node is in eps distance of the point, there may be a closer point on the other side of the tree, obscured by the dimension cutoff
                self.nearest_neighbours(root.right, point, eps, k, depth + 1, neighbours)
        else:
            self.nearest_neighbours(root.right, point, eps, k, depth + 1, neighbours)
            if abs(point[cd] - root.point[cd]) <= eps:
                self.nearest_neighbours(root.left, point, eps, k, depth + 1, neighbours)
        return neighbours

    def distance(self, point1=[], point2=[]):
        total = 0
        for n in range(0, len(point1)):
            total += (point1[n] - point2[n]) ** 2
        return math.sqrt(total)


class Node:
    def __init__(self, point, index):
        self.point = point
        self.left = None
        self.right = None
        self.index = index


def newNode(point, index):
    return Node(point, index)


def insert(root, point, index, k, depth=0):
    if not root:
        return newNode(point, index)

    cd = depth % k

    if point[cd] < root.point[cd]:
        root.left = insert(root.left, point, index, k, depth + 1)
    else:
        root.right = insert(root.right, point, index, k, depth + 1)
    return root

def compute_cluster_center(cluster):
    return np.mean(cluster, axis=0)

def getDense(cluster):
    center = compute_cluster_center(cluster)
    distances = np.linalg.norm(cluster - center, axis=1)

    return len(cluster) / (np.mean(distances) + 1e-10)

def distancee(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))
def merge_clusters(clusters,clusterCenters, k):
    iterations = 0
    while len(clusters) > k:
        iterations += 1
        print(iterations, "number of iterations")

        sizes = [len(c) for c in clusters]

        c1 = np.argmin(sizes)
        t1 = clusters[c1]

        clusters.pop(c1)
        sizes.pop(c1)
        t2 = clusterCenters.pop(c1)
        dists = distancee(t2,clusterCenters)

        c2 = np.argmin(dists)
        t2 = clusters[c2]
        clusters.pop(c2)
        sizes.pop(c2)
        clusterCenters.pop(c2)


        merged = np.vstack((t1, t2))
        merged_points = np.vstack((t1,t2))
        new_center = merged_points.mean(axis=0)
        clusters.append(merged)
        clusterCenters.append(new_center)
    return clusters,clusterCenters

def split_clusters(clusters, clusterCenters, k=2):
    densities = [getDense(c) for c in clusters]
    idx = np.argmin(densities)
    clustToSplit = clusters.pop(idx)
    clusterCenters.pop(idx)

    if len(clustToSplit) < k:
        return clusters, clusterCenters

    kmeans = kme(n_clusters=k)
    labels = kmeans.fit_predict(clustToSplit)

    newClsuters = []
    newCenters = []
    for i in range(k):
        points = clustToSplit[labels == i]
        if len(points) == 0:
            continue
        newClsuters.append(points)
        newCenters.append(points.mean(axis=0))

    return newClsuters, newCenters

class KMeans:
    def __init__(self, n_clusters=3, max_iter=500, tol=1e-10, init='kmeans++', alg='lloyd',eps=8.5, minpts=120):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init  # 'random' or 'kmeans++'
        self.alg = alg
        self.eps = eps
        self.minpts = minpts

    def fit(self, X_train):
        uncap = False
        n_samples, n_features = X_train.shape
        if self.alg == 'lloyd':
            self.centroids = DBKMEANS.fit(data_points=X_train, eps=self.eps, minpts=self.minpts,kclust=self.n_clusters)

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


        else:
            dbk = DBKMEANS()
            self.centroids = dbk.fit(data_points=X_train,eps=3.4,minpts=5,kclust=self.n_clusters)
            labels = np.zeros(n_samples, dtype=int)
            upper_bounds = np.full(n_samples, np.inf)

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

                # Step 3: Update bounds
                centroid_shifts = np.linalg.norm(self.centroids - prevcentroids, axis=1)
                upper_bounds += centroid_shifts[labels]

                if np.max(centroid_shifts) <= self.tol:
                    print(f"Converged in {it + 1} iterations")
                    break

    def evaluate(self, X):
        centroids = []
        centroidz = []
        for x in X:
            dist = distance(x, self.centroids)
            centroid = np.argmin(dist)
            centroids.append(self.centroids[centroid])
            centroidz.append(centroid)
        return centroids, centroidz

    def fit_predict(self, X):
        self.fit(X)
        _, cluster_indices = self.evaluate(X)
        return np.array(cluster_indices)
def distance(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

def distance_matrix(X, Y):
    # Fast Euclidean distance matrix: ||X - Y||^2
    return np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)