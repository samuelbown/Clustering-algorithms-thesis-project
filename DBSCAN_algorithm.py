import random
import math
import numpy as np

class DBSCAN:
    def fit(data_points, eps, minpts):
        unmarked = list(data_points)
        clusters = []
        labels = [-1] * len(data_points)
        noise = []

        core_points = [] 
        for point in unmarked: #finding all core points (points with at least minpts nodes within an eps radius)
            d = distance(point, unmarked) 
            neighbour_count = 0
            for p in d:
                if p[1] <= eps:
                    neighbour_count +=1
            if neighbour_count >= minpts:
                core_points.append(point)
        for point in core_points:
            unmarked = remove_from_array(unmarked, point)
        while len(core_points) > 0:
            point = random.choice(core_points) #pick a random core point
            cluster = [point]
            point_index = get_index(data_points, point) #find point index
            cluster_index = len(clusters)
            labels[point_index] = cluster_index # mark point
            core_points = remove_from_array(core_points, point)
            d = distance(point, core_points) #check distances between it and all the other core points
            neighbours = []
            for p in d:
                if p[1] < eps:
                    core_points = remove_from_array(core_points, p[0])
                    neighbours.append(p[0]) #add all core points below the epsilon distance to neighbours
            i = 0
            while i < len(neighbours): #checking the neighbours of the neigbours
                neighbour = neighbours[i]
                d = distance(neighbour, core_points)
                for p in d:
                    if p[1] < eps and not check_in_array(neighbours, p[0]): #append new neighbour if it is new and within epsilon of an existing neighbour
                        core_points = remove_from_array(core_points, p[0])
                        neighbours.append(p[0]) #
                        break
                i += 1
            for neighbour in neighbours: #add each all neighbours to a cluster
                if not check_in_array(cluster, neighbour):
                    point_index = get_index(data_points, neighbour) 
                    labels[point_index] = cluster_index
                    cluster.append(neighbour)
            clusters.append(cluster)
        while len(unmarked) > 0: #check all non-core points to see if they belong to a specific cluster or are noise
            point = random.choice(unmarked)
            point_index = get_index(data_points, point)
            isNoise = True
            for cluster in clusters:
                d = distance(point, cluster)
                for p in d:
                    if p[1] <= eps:
                        isNoise = False
                        point_index = get_index(data_points, point)
                        labels[point_index] = get_index(clusters, cluster)
                        cluster.append(point)
                        unmarked = remove_from_array(unmarked, point)
                        break
            if isNoise == True:
                noise.append(point)
                unmarked = remove_from_array(unmarked, point)
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
