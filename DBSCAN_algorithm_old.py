import random
import math
import numpy as np

class DBSCAN:
    def fit(data_points, eps, minpts):
        noise = [] 
        unmarked = list(data_points)
        marked = []
        clusters = []
        labels = [-1] * len(data_points)
        while len(unmarked) > 0:
            point = random.choice(unmarked) #pick a random point
            point_index = get_index(data_points, point) #find point index
            marked.append(point) # mark point
            unmarked = remove_from_array(unmarked, point)
            d = distance(point, unmarked) #check distances between it and all the other points
            neighbours = []
            for p in d:
                if p[1] < eps:
                    neighbours.append(p[0]) #add all points below the epsilon distance to neighbours
            if len(neighbours) < minpts:
                noise.append(point) #if neighbours are too few set point to noise
                labels[point_index] = -1
                for x in neighbours: #all neighbours become noise too
                    temp_index = get_index(data_points, x) 
                    labels[temp_index] = -1
            else:
                cluster = [point]
                cluster_index = len(clusters) #gets most recent cluster
                labels[point_index] = cluster_index
                for neighbour in neighbours: #for each neighbour, their neighbours are checked and added to the current set of neighbours if they fit the requirements
                    if not check_in_array(neighbour, marked):
                        new_d = distance(neighbour, data_points) 
                        marked.append(neighbour)
                        unmarked = remove_from_array(unmarked, neighbour)
                        new_neighbours = []
                        for p in new_d:
                            if p[1] < eps:
                                new_neighbours.append(p[0])
                        if len(new_neighbours) > minpts:
                            for new_point in new_neighbours:
                                already_in_neighbours = False
                                for x in neighbours:
                                    if np.array_equal(new_point, x):
                                        already_in_neighbours =True
                                        break
                                if not already_in_neighbours:
                                    neighbours.append(new_point)
                    isIn = False
                    for c in clusters: #check if the neighbour exists in any clusters
                        if check_in_array(neighbour, c):
                            isIn = True
                    if not isIn: #if not then append to this cluster
                        cluster.append(neighbour)
                    neighbour_index = get_index(data_points, neighbour)
                    if labels[neighbour_index] == -1:
                        labels[neighbour_index] = cluster_index
                clusters.append(cluster)  #append cluster to list of clusters
        return clusters, noise, labels
        
def remove_from_array(pointarray, point): #these functions are necessary because these operations don't work on np arrays easily
    newarray = []
    for x in pointarray:
        if not np.array_equal(x, point):
            newarray.append(x)
    return newarray

def check_in_array(pointarray, point):
    for x in pointarray:
        if np.array_equal(point, x):
            return True
    return False

def get_index(pointarray, point):
    for index, x in enumerate(pointarray):
        if np.array_equal(x, point):
            return index
    return -1

def distance(point1, data):
    distances = []
    for point in data:
        d = math.sqrt((point[0] - point1[0])**2 + (point[1] - point1[1])**2) #sqrt((x2-x1)^2 + (y2-y1)^2)
        if d != 0:
            distances.append([point, d])
    return np.array(distances, dtype=object)
