# -*- coding: utf-8 -*-
# Name: Ding Wang
# Email: diw005@ucsd.edu
# PID: A53089251
from pyspark import SparkContext
sc = SparkContext()
# coding: utf-8

# ## K-means++
# 
# In this notebook, we are going to implement [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) algorithm with multiple initial sets. The original k-means++ algorithm will just sample one set of initial centroid points and iterate until the result converges. The only difference **in this implementation is that we will sample `RUNS` sets of initial centroid points and update them in parallel**. The procedure will finish when all centroid sets are converged.

# In[1]:

### Definition of some global parameters.
K = 5  # Number of centroids
RUNS = 25  # Number of K-means runs that are executed in parallel. Equivalently, number of sets of initial points
RANDOM_SEED = 60295531
converge_dist = 0.1 # The K-means algorithm is terminated when the change in the location 
                    # of the centroids is smaller than 0.1


# In[47]:

import numpy as np
import pickle
import sys
from numpy.linalg import norm
from matplotlib import pyplot as plt


def print_log(s):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def parse_data(row):
    '''
    Parse each pandas row into a tuple of (station_name, feature_vec),
    where feature_vec is the concatenation of the projection vectors
    of TAVG, TRANGE, and SNWD.
    '''
    return (row[0],
            np.concatenate([row[1], row[2], row[3]]))


def compute_entropy(d):
    '''
    Compute the entropy given the frequency vector `d`
    '''
    d = np.array(d)
    d = 1.0 * d / d.sum()
    return -np.sum(d * np.log2(d))


def choice(p):
    '''
    Generates a random sample from [0, len(p)),
    where p[i] is the probability associated with i. 
    '''
    random = np.random.random()
    r = 0.0
    for idx in range(len(p)):
        r = r + p[idx]
        if r > random:
            return idx
    assert(False)


def kmeans_init(rdd, K, RUNS, seed):
    '''
    Select `RUNS` sets of initial points for `K`-means++
    Represented by centers
    '''
    # the `centers` variable is what we want to return
    n_data = rdd.count()
    shape = rdd.take(1)[0][1].shape[0]   # == 9

    centers = np.zeros((RUNS, K, shape))   #`RUN` sets of K-means, each one dimension of K * shape

    def update_dist(vec, dist, k):
        new_dist = norm(vec - centers[:, k], axis=1)**2
        return np.min([dist, new_dist], axis=0)

    # The second element `dist` in the tuple below is the closest distance from
    # each data point to the selected points in the initial set, where `dist[i]`
    # is the closest distance to the points in the i-th initial set.
    # Should be a list of `RUN` points stands for `RUN` norm distances.
    data = rdd.map(lambda p: (p, [np.inf] * RUNS))               .cache()
    # Collect the feature vectors of all data points beforehand, might be
    # useful in the following for-loop
    # Just the feature vectors
    local_data = rdd.map(lambda (name, vec): vec).collect()
    
    # Randomly select the first point for every run of k-means++,
    # i.e. randomly select `RUNS` points and add it to the `centers` variable
    sample = [local_data[k] for k in np.random.randint(0, len(local_data), RUNS)]

    centers[:, 0] = sample

    for idx in range(K - 1):
        ##############################################################################
        # Insert your code here:
        ##############################################################################
        # In each iteration, you need to select one point for each set
        # of initial points (so select `RUNS` points in total).
        # For each data point x, let D_i(x) be the distance between x and
        # the nearest center that has already been added to the i-th set.
        # Choose a new data point for i-th set using a weighted probability
        # where point x is chosen with probability proportional to D_i(x)^2
        ##############################################################################
        #-------------------------Added Code-----------------------------
        data = data.map(lambda (x,y): (x, update_dist(x[1], y, idx)))
        m_distance = np.array(data.values().collect())
        m_distance_p = m_distance/m_distance.sum(axis = 0)
        for i in range(RUNS):
            centers[i,idx+1] = local_data[choice(m_distance_p[:,i])]
        #-------------------------Added Code-----------------------------
        #removed code: pass
    return centers


def get_closest(p, centers):
    '''
    Return the indices the nearest centroids of `p`.
    `centers` contains sets of centroids, where `centers[i]` is
    the i-th set of centroids.
    '''
    #centers = RUN * K * Num_of_Features
    best = [0] * len(centers)
    closest = [np.inf] * len(centers)
    for idx in range(len(centers)):   #0 - (RUN - 1)
        for j in range(len(centers[0])):   #0 - (K - 1)
            temp_dist = norm(p - centers[idx][j])
            if temp_dist < closest[idx]:
                closest[idx] = temp_dist
                best[idx] = j
    return best   # 1 * RUN list


def kmeans(rdd, K, RUNS, converge_dist, seed):
    '''
    Run K-means++ algorithm on `rdd`, where `RUNS` is the number of
    initial sets to use.
    '''
    k_points = kmeans_init(rdd, K, RUNS, seed)   #(25, 5, 9), <type 'numpy.ndarray'>
    print_log("Initialized.")
    temp_dist = 1.0
    
    #My variables
    local_data = np.array(rdd.map(lambda (name, vec): vec).collect())   #Num of points * 9
    new_points = {}
    
    iters = 0
    st = time.time()
    while temp_dist > converge_dist:
        ##############################################################################
        # INSERT YOUR CODE HERE
        ##############################################################################
        
        # Update all `RUNS` sets of centroids using standard k-means algorithm
        # Outline:
        #   - For each point x, select its nearest centroid in i-th centroids set
        #   - Average all points that are assigned to the same centroid
        #   - Update the centroid with the average of all points that are assigned to it
        
        # Insert your code here
        #npa_dataClosest = np.array(map(lambda x: get_closest(x, k_points),local_data))
        lRDD_dataClosest = rdd.map(lambda p: get_closest(p[1], k_points))         .cache()
        npa_dataClosest = np.array(lRDD_dataClosest.collect())   #Num of points * RUNS

        for idx in range(RUNS):
            for j in range(K):
                new_points[(idx,j)] = local_data[npa_dataClosest[:,idx] == j].mean(axis = 0)
        # You can modify this statement as long as `temp_dist` equals to
        # max( sum( l2_norm of the movement of j-th centroid in each centroids set ))
        ##############################################################################

        temp_dist = np.max([
                np.sum([norm(k_points[idx][j] - new_points[(idx, j)]) for j in range(K)])
                    for idx in range(RUNS)])

        iters = iters + 1
        if iters % 5 == 0:
            print_log("Iteration %d max shift: %.2f (time: %.2f)" %
                      (iters, temp_dist, time.time() - st))
            st = time.time()

        # update old centroids
        # You modify this for-loop to meet your need
        for ((idx, j), p) in new_points.items():
            k_points[idx][j] = p

    return k_points


# In[12]:

## Read data
data = pickle.load(open("../Data/Weather/stations_projections.pickle", "rb"))
rdd = sc.parallelize([parse_data(row[1]) for row in data.iterrows()])
rdd.take(1)


# In[48]:

# main code
rdd.take(1)
import time

st = time.time()

np.random.seed(RANDOM_SEED)
centroids = kmeans(rdd, K, RUNS, converge_dist, np.random.randint(1000))
group = rdd.mapValues(lambda p: get_closest(p, centroids))            .collect()

print "Time takes to converge:", time.time() - st


# In[49]:

## Verify your results

# In[50]:

def get_cost(rdd, centers):
    '''
    Compute the square of l2 norm from each data point in `rdd`
    to the centroids in `centers`
    '''
    def _get_cost(p, centers):
        best = [0] * len(centers)
        closest = [np.inf] * len(centers)
        for idx in range(len(centers)):
            for j in range(len(centers[0])):
                temp_dist = norm(p - centers[idx][j])
                if temp_dist < closest[idx]:
                    closest[idx] = temp_dist
                    best[idx] = j
        return np.array(closest)**2
    
    cost = rdd.map(lambda (name, v): _get_cost(v, centroids)).collect()
    return np.array(cost).sum(axis=0)

cost = get_cost(rdd, centroids)


# In[51]:

log2 = np.log2

print log2(np.max(cost)), log2(np.min(cost)), log2(np.mean(cost))


# ## Plot the increase of entropy after multiple runs of k-means++

# In[52]:

entropy = []

for i in range(RUNS):
    count = {}
    for g, sig in group:
        _s = ','.join(map(str, sig[:(i + 1)]))
        count[_s] = count.get(_s, 0) + 1
    entropy.append(compute_entropy(count.values()))


# **Note:** Remove this cell before submitting to PyBolt (PyBolt does not fully support matplotlib)

# ## Print the final results

# In[54]:

print 'entropy=',entropy
best = np.argmin(cost)
print 'best_centers=',list(centroids[best])
