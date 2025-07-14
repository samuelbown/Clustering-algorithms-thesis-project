import math
import numpy as np

class DBSCANBF:
    def fit(self, data_points, eps, minpts):
        labels = [-1] * len(data_points)
        clusters = []
        noise = []
        visited = set()

        for index, point in enumerate(data_points):
            if index in visited:
                continue
            visited.add(index)

            neighbours = self.get_neighbours(data_points, eps, point)
                
            if len(neighbours) + 1 < minpts:
                noise.append(point)
            else:
                cluster = []
                self.expand_neighbours(index, data_points, point, eps, minpts, cluster, clusters, neighbours, labels, visited)
                clusters.append(cluster)

        return clusters, noise, labels


    def expand_neighbours(self, index, data_points, point, eps, minpts, cluster, clusters, neighbours, labels, visited):
        cluster.append(data_points[index])
        labels[index] = len(clusters)
        queue = neighbours.copy()
        while queue:
            neighbour_index = queue.pop(0) #get the first neighbour index
            if neighbour_index not in visited:
                visited.add(neighbour_index) 
                new_neighbours = self.get_neighbours(data_points, eps, data_points[neighbour_index]) # check all its neighbours
                if len(new_neighbours) + 1 >= minpts: #if new neighbours are more than minpts
                    temp_list = []
                    for x in new_neighbours:
                        if x not in queue:
                            temp_list.append(x)
                    queue.extend(temp_list) #add new neighbours to queue
            
            if labels[neighbour_index] == -1: #if not labeled
                labels[neighbour_index] = len(clusters) #add to most recent cluster
                cluster.append(data_points[neighbour_index])

    def get_neighbours(self, data_points, eps, point):
        neighbors = []
        for index, point1 in enumerate(data_points):
            d = distance(point, point1)
            if d < eps and not np.array_equal(point, point1):
                neighbors.append(index)
        return neighbors
    
def distance(point1 = [], point2 = []):
    total = 0
    for n in range(0, len(point1)):
        total += (point1[n] - point2[n])**2
    return math.sqrt(total)