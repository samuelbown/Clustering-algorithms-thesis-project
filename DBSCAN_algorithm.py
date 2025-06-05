import random
import math
import numpy as np

def DBSCAN(data_points, eps = 0.5, minpts=3):
    noise = [] 
    unmarked = data_points
    marked = []
    clusters = []
    while len(unmarked) > 0:
        point = random.choice(unmarked) 
        marked.append(point) #pick random point and mark it 
        unmarked.remove(point)
        d = distance(point, unmarked) #check distances between it and all the other points
        neighbours = []
        for p in d:
            if p[1] < eps:
                neighbours.append(p[0]) #add all points below the epsilon distance to neighbours
        if len(neighbours) < minpts:
            noise.append(point) #if the neighbours are too few to become a cluster they all become noise
        else:
            cluster = [point]
            for neighbour in neighbours: #for each neighbour, their neighbours are checked and added to the current set of neighbours if they fit the requirements
                if neighbour not in marked:
                    new_d = distance(neighbour, data_points)
                    marked.append(neighbour)
                    unmarked.remove(neighbour)
                    new_neighbours = []
                    for p in new_d:
                        if p[1] < eps:
                            new_neighbours.append(p[0])
                    if len(new_neighbours) > minpts:
                        neighbours += new_neighbours
                isIn = False
                for c in clusters: #check if the neighbour exists in any clusters
                    if neighbour in c:
                        isIn = True
                if not isIn: #if not then append to this cluster
                    cluster.append(neighbour)
            clusters.append(cluster)  #append cluster to list of clusters
    return clusters, noise
        

def distance(point1, data):
    distances = []
    for point in data:
        d = math.sqrt((point[0] - point1[0])**2 + (point[1] - point1[1])**2) #sqrt((x2-x1)^2 + (y2-y1)^2)
        if d != 0:
            distances.append([point, d])
    return np.array(distances, dtype=object)
