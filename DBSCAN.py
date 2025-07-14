import math


class DBSCAN:
    def fit(self, data_points, eps, minpts):
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
                self.expand_neighbours(index, data_points, point, eps, minpts, cluster, clusters, neighbours, labels, visited, root, k)
                clusters.append(cluster)

        return clusters, noise, labels


    def expand_neighbours(self, index, data_points, point, eps, minpts, cluster, clusters, neighbours, labels, visited, root, k):
        cluster.append(data_points[index])
        labels[index] = len(clusters)
        queue = neighbours.copy()
        while queue:
            neighbour_index = queue.pop(0) #get the first neighbour index
            if neighbour_index not in visited:
                visited.add(neighbour_index)
                new_neighbours = self.nearest_neighbours(root, data_points[neighbour_index], eps, k) # check all its neighbours
                if len(new_neighbours) + 1 >= minpts: #if new neighbours are more than minpts
                    temp_list = []
                    for x in new_neighbours:
                        if x not in queue:
                            temp_list.append(x)
                    queue.extend(temp_list) #add new neighbours to queue
            
            if labels[neighbour_index] == -1: #if not labeled
                labels[neighbour_index] = len(clusters) #add to most recent cluster
                cluster.append(data_points[neighbour_index])

    def setup_tree(self, data_points, k):
        root = None
        for index,x in enumerate(data_points): 
            root = insert(root, x, index, k)
        return root

    def nearest_neighbours(self, root, point, eps, k, depth=0, neighbours = None):
        if root is None:
            return neighbours
        
        if neighbours is None:
            neighbours = []

        d = self.distance(point, root.point)
        if d <= eps:
            neighbours.append(root.index) 
        cd = depth % k #cd - current dimension, each layer of the tree the dimension changes to narrow down on the point quicker
        
        if point[cd] < root.point[cd]: #if x/y of the point is less/more than the x/y of the current node then go left/right down the tree 
            self.nearest_neighbours(root.left, point, eps, k, depth + 1, neighbours) 
            if abs(point[cd] - root.point[cd]) <= eps: #if the current node is in eps distance of the point, there may be another point within eps on the other side of the tree, obscured by the dimension cutoff
                self.nearest_neighbours(root.right, point, eps, k, depth + 1, neighbours)
        else:
            self.nearest_neighbours(root.right, point, eps, k, depth + 1, neighbours)
            if abs(point[cd] - root.point[cd]) <= eps:
                self.nearest_neighbours(root.left, point, eps, k, depth + 1, neighbours)
        return neighbours

    def distance(self, point1 = [], point2 = []):
        total = 0
        for n in range(0, len(point1)):
            total += (point1[n] - point2[n])**2
        return math.sqrt(total)

class Node:
    def __init__(self, point, index):
        self.point = point
        self.left = None
        self.right = None
        self.index = index

def newNode(point, index):
    return Node(point, index)

def insert(root, point, index, k, depth = 0): 
    if not root:
        return newNode(point, index)

    cd = depth % k
    
    if point[cd] < root.point[cd]: 
        root.left = insert(root.left, point, index, k, depth + 1)
    else:
        root.right = insert(root.right, point, index, k, depth + 1)
    return root
