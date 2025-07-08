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

        cluster_centers = []
        cluster_sizes = []
        m = len(clusters)
        k = kclust

        for cluster in clusters:
            center = cluster.mean(axis=0)
            size = len(cluster)
            cluster_centers.append(center)
            cluster_sizes.append(size)
        print(len(cluster_centers))
        if m > k:
            clusters, cluster_centers = merge_clusters(clusters, cluster_centers, k)
        elif m < k:
            clusters, cluster_centers = split_clusters(clusters, cluster_centers, k)

        print(len(cluster_centers))
        return np.array(cluster_centers)


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

def compute_cluster_density(cluster):
    center = compute_cluster_center(cluster)
    distances = np.linalg.norm(cluster - center, axis=1)

    return len(cluster) / (np.mean(distances) + 1e-10)

def distancee(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))
def merge_clusters(clusters,cluster_centers, k):
    iterations = 0
    while len(clusters) > k:
        iterations += 1
        print(iterations, "number of iterations")

        densities = [compute_cluster_density(c) for c in clusters]
        sizes = [len(c) for c in clusters]

        c1 = np.argmin(sizes)
        t1 = clusters[c1]

        clusters.pop(c1)
        sizes.pop(c1)
        t2 = cluster_centers.pop(c1)
        dists = distancee(t2,cluster_centers)

        c2 = np.argmin(dists)
        t2 = clusters[c2]
        clusters.pop(c2)
        sizes.pop(c2)
        cluster_centers.pop(c2)


        merged = np.vstack((t1, t2))
        merged_points = np.vstack((t1,t2))
        new_center = merged_points.mean(axis=0)
        print(len(clusters), "cluster length")
        print(c1, "c1")
        print(c2, "c2")
        print(sizes, "sizes")
        print(densities, "densities")

        clusters.append(merged)
        cluster_centers.append(new_center)
    return clusters,cluster_centers

def split_clusters(clusters, cluster_centers, k=2):
    densities = [compute_cluster_density(c) for c in clusters]
    idx = np.argmin(densities)
    cluster_to_split = clusters.pop(idx)
    cluster_centers.pop(idx)

    if len(cluster_to_split) < k:
        print("Not enough points to split this cluster")
        return clusters, cluster_centers

    kmeans = kme(n_clusters=k)
    labels = kmeans.fit_predict(cluster_to_split)

    # Form new clusters
    new_clusters = []
    new_centers = []
    for i in range(k):
        points = cluster_to_split[labels == i]
        if len(points) == 0:
            continue
        new_clusters.append(points)
        new_centers.append(points.mean(axis=0))

    # Add new splits
    print("hi")
    print(len(cluster_centers))
    print(len(clusters))
    clusters.extend(new_clusters)
    cluster_centers.extend(new_centers)
    print(len(cluster_centers))

    return new_clusters, new_centers

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
                prev_centroids = self.centroids.copy()

                # Vectorized distance computation
                distances = np.linalg.norm(X_train[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
                labels = np.argmin(distances, axis=1)

                for i in range(self.n_clusters):
                    points_in_cluster = X_train[labels == i]
                    if len(points_in_cluster) > 0:
                        self.centroids[i] = points_in_cluster.mean(axis=0)

                shifts = np.linalg.norm(self.centroids - prev_centroids, axis=1)
                if np.all(shifts <= self.tol) and  uncap == False:
                    print("it = ",iteration)
                    break
            print("maxx")
        else:
            ddd = DBKMEANS()
            self.centroids = ddd.fit(data_points=X_train,eps=8.4,minpts=120,kclust=self.n_clusters)
            labels = np.zeros(n_samples, dtype=int)
            upper_bounds = np.full(n_samples, np.inf)
            lower_bounds = np.zeros((n_samples, self.n_clusters))

            for it in range(self.max_iter):
                prev_centroids = self.centroids.copy()


                centroid_distances = distance_matrix(self.centroids, self.centroids)
                np.fill_diagonal(centroid_distances, np.inf)
                s = 0.5 * np.min(centroid_distances, axis=1)



                needs_update = (upper_bounds > s[labels])
                if np.any(needs_update):
                    distances = distance_matrix(X_train[needs_update], self.centroids)
                    new_labels = np.argmin(distances, axis=1)
                    upper_bounds[needs_update] = distances[np.arange(len(new_labels)), new_labels]
                    lower_bounds[needs_update] = distances

                    labels[needs_update] = new_labels

                for k in range(self.n_clusters):
                    points = X_train[labels == k]
                    if len(points) > 0:
                        self.centroids[k] = np.mean(points, axis=0)

                centroid_shifts = np.linalg.norm(self.centroids - prev_centroids, axis=1)
                upper_bounds += centroid_shifts[labels]
                lower_bounds -= centroid_shifts[np.newaxis, :]
                lower_bounds = np.maximum(lower_bounds, 0)

                if np.max(centroid_shifts) <= self.tol and uncap == False:
                    print(f"Converged in {it + 1} iterations")
                    break



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

def distance(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

def distance_matrix(X, Y):
    # Fast Euclidean distance matrix: ||X - Y||^2
    return np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)