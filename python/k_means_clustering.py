""" Implementation of k-means clustering algorithm
    Put the implementation into a class
"""

import os
import sys
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from collections import defaultdict

class KMeansClustering:
    def __init__(self, X, K, max_iters, plotProgress=False):
        self.X = X
        self.K = K
        self.max_iters = max_iters
        self.plotProgress = plotProgress

    def runKMeans(self):
        initial_centroids = self.kMeansInitCentroids(self.X, self.K)
        self.centroids, self.idx = self.k_means_clust(
                                          self.X, initial_centroids,
                                          self.max_iters,
                                          self.plotProgress)

        return self.centroids, self.idx

    def k_means_clust(self, X, initial_centroids,
                      max_iters, plotProgress=False):
        
        """ Cluster vehicles based on event data
        and visualize the final clusters on a graph
        Parameters
        ---------------
        X : numpy.ndarray
               Data matrix that stores features of vehicles
               Each row store features of a car
        initial_centroids : numpy.ndarray
        max_iters : int

        Returns
        ---------------
        centroids : numpy.ndarray
               Final centroids learned from the data
               
        idx : numpy.ndarray
               Centroid assignment for examples in X
               idx[i] is the centroid of example i-th in X
        """

        centroids = initial_centroids
        previous_centroids = centroids
        k = centroids.shape[0]
        prev_idx = np.zeros(X.shape[0])

        for i in range(max_iters):
            print('K-means iteration %d/%d...' % ( (i+1), max_iters) )
            idx = self.findClosestCentroids(X, centroids)

            if ( np.array_equal(idx, prev_idx) ):
                break
            else:
                prev_idx = idx

            if plotProgress:
                plotProgresskMeans(X, centroids, previous_centroids, idx, k, i)
                previous_centroids = centroids
                input('Press enter to continue.')
            centroids = self.computeCentroids(X, idx, k)

        return centroids, idx

    def computeCentroids(self, X, idx, K):
        """ Compute centroids from current assignments of examples
        
        Parameters
        --------------
        X : numpy.ndarray
            Data matrix
        idx : numpy.ndarray
            centroid assignment for examples in X
            idx[i] is the centroid of example i-th in X
        K : int
            The number of centroids
            
        Return
        --------------
        centroids : numpy.ndarray
            new centroids (2D array) (shape: (K,2))
        """

        centroids = np.zeros( (K, X.shape[1]) )
        for k in range(K):
            Ck = [i for i in range(idx.size) if idx[i] == k]
            centroids[k,:] = np.mean( X[Ck,:], axis = 0 )

        return centroids

    def findClosestCentroids(self, X, centroids):
        """ Find closet centroids for examples in X

        Parameters
        ------------
        X : numpy.ndarray
            Data matrix
        centroids : numpy.ndarray
            centroids in k-means algorithm

        Return
        ------------
        idx : numpy.ndarray (K elements)
            centroid assignment for examples in X
            idx[i] is the centroid of example i-th in X
        """

        k = centroids.shape[0]
        idx = np.zeros(X.shape[0], dtype=np.int)

        for i in range(X.shape[0]):
            dist = np.sum( (X[i] - centroids) ** 2, axis=1 )
            idx[i] = np.argmin(dist)
            
        return idx

    def plot_data_points(self, X, idx, k):
        """ Plot data points, coloring them so that points with the same
        centroid assignments have the same color

        Parameters
        ------------
        idx : numpy.ndarray
            Centroid assignment, idx[i] is the centroid of
            the example i-th in X
        k : int
            The number of centroids
        """

        plt.scatter(X[:,0], X[:,1], c=idx, marker='^', s=30)
        
            
    def plotProgresskMeans(self, X, centroids, previous_centroids, idx, k, i):
        """ Plot the progress when running k-means

        Parameters
        ---------------
        X : numpy.ndarray
            centroid, previous_centroids : numpy.ndarray
        idx : numpy.ndarray
            centroid assignment of examples
        k : int
            number of centroids
        i : iteration number

        Return
        ---------------
            None
        """

        plot_data_points(X, idx, k)
        plt.scatter(centroids[:,0], centroids[:,1], marker='x',
                    s = 70, c='k')

        for j in range( centroids.shape[0] ):
            plt.plot([ centroids[j,0], previous_centroids[j,0] ],
                     [ centroids[j,1], previous_centroids[j,1] ], c='b')

        plt.title('Iteration %d' % (i+1))
        plt.draw()
        plt.pause(0.01)

    def kMeansInitCentroids(self, X, K):
        """ Random initialization for k-means algorithms
        Just return a randomly sample of k examples in X
        """

        np.random.seed(99999)
        randidx = np.random.permutation( range(X.shape[0]) )
        
        return X[randidx[0:K], :]


    
    

